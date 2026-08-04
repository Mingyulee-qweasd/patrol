"""그래프 기반 실내 순찰 러너 (v3) — 병원 위상 그래프 위에서 이종 3대 순찰.

판단 축: is_task + 능력 조합(caps). WiFi 공유 상황판(단일 board — 조우 병합 불필요).
발견→누적→게이트(문턱+근접)→능력 매칭 파견→실행. 이동은 그래프 최단경로.
합성 VLM은 P0 error_model 오면 교체 (동일 인터페이스).
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import networkx as nx

from .env_graph import load_graph_env

CAP_LIST = ["report", "manip", "clean"]
DT = 1.0                      # 1스텝 = 1초
THETA = 0.80                 # 확신 문턱 (보정 확률)
MATCH_R = 2.0                # 같은 개체 판정 반경(m)


# ── 임무 (그래프 세계) ─────────────────────────────────────────
@dataclass
class Task:
    tid: int
    kind: str
    gt_caps: frozenset       # 정답 능력 조합 (report=보고만/비개입)
    gt_class: str            # 'task' | 'nontask' | 'report'
    xy: np.ndarray
    n: int                   # 필요 로봇 수 (조합=2, 보고/비개입=0)
    u: int
    c: float                 # 처리 시간
    t_spawn: float
    t_found: float = None
    t_done: float = None
    arrived: set = field(default_factory=set)
    work_until: float = None
    @property
    def active(self): return self.t_done is None


# ── 공유 상황판 후보 ───────────────────────────────────────────
@dataclass
class Cand:
    cid: int
    xy: np.ndarray
    s_logodds: float
    caps_hat: frozenset       # 근접 판정으로 추정된 능력 조합
    u_hat: int
    gt_tid: int
    near_confirmed: bool = False
    committed: bool = False
    assigned: set = field(default_factory=set)
    @property
    def s(self): return 1.0 / (1.0 + np.exp(-self.s_logodds))


def _logit(p): return np.log(np.clip(p, 1e-4, 1-1e-4) / (1 - np.clip(p, 1e-4, 1-1e-4)))


# ── 합성 VLM (P0 전 임시) ──────────────────────────────────────
class SynthVLM:
    """정답 caps에서 판정 추첨. 거리밴드별 정확도 차등. P0 오면 교체."""
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def observe(self, task: Task, band: str):
        p_detect = 0.85 if band == "far" else 0.98
        if self.rng.random() > p_detect:
            return None                       # 미탐
        near = band == "near"
        if task.gt_class == "task":
            correct = 0.9 if near else 0.72
            is_task = True
            if self.rng.random() < correct:
                caps = task.gt_caps
                conf = self.rng.normal(88 if near else 74, 8)
            else:                             # 오판: 조합을 단일로 축소 등
                caps = frozenset([next(iter(task.gt_caps))]) if task.gt_caps else frozenset()
                conf = self.rng.normal(76, 10)
        else:                                 # nontask/report: 개입 대상 아님
            correct = 0.92 if near else 0.80
            if self.rng.random() < correct:
                is_task = task.gt_class == "report"   # report는 is_task지만 caps=report
                caps = frozenset(["report"]) if task.gt_class == "report" else frozenset()
                conf = self.rng.normal(85, 8)
            else:                             # 오검: 개입 대상으로 착각
                is_task = True
                caps = frozenset([self.rng.choice(["manip", "clean"])])
                conf = self.rng.normal(72, 12)
        return {"is_task": is_task, "caps": caps,
                "conf": float(np.clip(conf, 1, 100))}


# ── 로봇 (그래프 순찰) ─────────────────────────────────────────
@dataclass
class GRobot:
    id: int
    role: str
    caps: frozenset
    v: float
    sense_r: float
    reliable_r: float
    xy: np.ndarray
    node: int                 # 현재/직전 노드
    path: list = field(default_factory=list)   # 남은 이동 노드열(월드좌표)
    mode: str = "patrol"      # patrol | detour | working
    target_cid: int = -1
    work_until: float = None


class GraphEpisode:
    def __init__(self, cfg_path, seed=0, horizon_s=6*3600, warmup_s=600,
                 error_model=None):
        self.env = load_graph_env(cfg_path)
        self.rng = np.random.default_rng(seed)
        self.vlm = error_model or SynthVLM(seed)
        self.horizon = horizon_s; self.warmup = warmup_s
        self.board: list[Cand] = []
        self._ncid = 0
        # 노드 방문 시각(idleness)
        self.node_last = {n: 0.0 for n in self.env.G.nodes}
        # 로봇 배치 (그래프 상 고르게 분산된 시작 노드)
        nodes = list(self.env.G.nodes)
        starts = [nodes[i * len(nodes)//len(self.env.robots)] for i in range(len(self.env.robots))]
        self.robots = {}
        for rb, sn in zip(self.env.robots, starts):
            self.robots[rb.id] = GRobot(rb.id, rb.role, rb.caps, rb.speed,
                                        rb.sense_r, rb.reliable_r,
                                        self.env.node_xy[sn].copy(), sn)
        self.tasks = self._make_stream(seed)
        self.log = []
        self.mis = 0

    # 임무 스트림 (푸아송)
    def _make_stream(self, seed):
        rng = np.random.default_rng(seed + 999)
        types = self.env.task_cfg["types"]
        names = list(types); shares = np.array([types[k]["share"] for k in names]); shares /= shares.sum()
        # 부하율 → λ (건당 평균 처리 로봇초 근사)
        rho = self.env.task_cfg.get("rho", 0.3)
        mean_c = np.mean([types[k]["c"] or 120 for k in names])
        lam = rho * len(self.robots) / max(mean_c, 1) * 0.5
        tasks = []; tid = 0; t = self.warmup
        while t < self.horizon:
            t += rng.exponential(1.0 / lam)
            if t >= self.horizon: break
            kind = str(rng.choice(names, p=shares)); spec = types[kind]
            caps = frozenset(spec["caps"])
            gcl = "report" if caps == frozenset(["report"]) else "task"
            xy = self.env.free_points(rng, 1)[0]
            tasks.append(Task(tid, kind, caps, gcl, xy, spec["n"], spec["u"],
                              spec["c"], t)); tid += 1
        # 비개입(함정) 개체
        n_amb = int(len(tasks) * self.env.task_cfg.get("nontask_ratio", 0.5))
        for _ in range(n_amb):
            ts = self.warmup + rng.uniform(0, self.horizon - self.warmup)
            xy = self.env.free_points(rng, 1)[0]
            tasks.append(Task(tid, "ambiguous", frozenset(), "nontask", xy, 0, 0, 0, ts)); tid += 1
        tasks.sort(key=lambda x: x.t_spawn)
        return tasks

    # ── 다음 순찰 노드: 가장 오래 안 본 이웃 방향 ──
    def _next_patrol_node(self, r: GRobot):
        # 현재 노드에서 idleness 큰 노드로 최단경로
        cand = sorted(self.env.G.nodes, key=lambda n: self.node_last[n])[:8]
        best = min(cand, key=lambda n: nx.shortest_path_length(
            self.env.G, r.node, n, weight="length") + self.node_last[n]*0.01)
        try:
            npath = nx.shortest_path(self.env.G, r.node, best, weight="length")
        except Exception:
            npath = [r.node]
        return [self.env.node_xy[n] for n in npath[1:]] or [self.env.node_xy[r.node]]

    def _move(self, r: GRobot, now):
        if not r.path:
            return
        tgt = r.path[0]
        d = np.linalg.norm(tgt - r.xy)
        step = r.v * DT
        if d <= step:
            r.xy = tgt.copy()
            # 도착 노드 idleness 갱신
            nn = self.env.nearest_node(r.xy)
            if np.linalg.norm(self.env.node_xy[nn] - r.xy) < 0.5:
                self.node_last[nn] = now; r.node = nn
            r.path.pop(0)
        else:
            r.xy = r.xy + (tgt - r.xy) / d * step

    # ── 감지 + 공유판 누적 ──
    def _sense(self, r: GRobot, now):
        for task in self.tasks:
            if task.t_spawn > now or not task.active:
                continue
            d = np.linalg.norm(task.xy - r.xy)
            if d > r.sense_r:
                continue
            if self.env.line_blocked(r.xy, task.xy):
                continue                       # 벽 뒤 안 보임
            band = "near" if d <= r.reliable_r else "far"
            j = self.vlm.observe(task, band)
            if j is None:
                continue
            self._update_board(task, j, band, now)

    def _update_board(self, task, j, band, now):
        delta = _logit(j["conf"]/100) * (1 if j["is_task"] else -1)
        if band == "far":
            delta *= 0.5
        for c in self.board:
            if np.linalg.norm(c.xy - task.xy) < MATCH_R:
                c.s_logodds = float(np.clip(c.s_logodds + delta, -30, 30))
                if band == "near":
                    c.near_confirmed = True
                    if j["is_task"]:
                        c.caps_hat = j["caps"]
                return
        self.board.append(Cand(self._ncid, task.xy.copy(), delta,
                               j.get("caps", frozenset()), task.u, task.tid,
                               near_confirmed=(band == "near")))
        self._ncid += 1

    # ── 파견: 능력 매칭 (WiFi 중앙 규칙) ──
    def _dispatch(self, now):
        for c in self.board:
            if c.committed or not (c.s >= THETA and c.near_confirmed):
                continue
            need = c.caps_hat
            if not need or need == frozenset(["report"]):
                c.committed = True            # 보고/비개입 — 이동 없음
                task = self._task(c.gt_tid)
                if task: task.t_found = task.t_found or now
                continue
            # 필요 능력 가진 유휴 로봇 찾기
            free = [r for r in self.robots.values() if r.mode == "patrol"]
            chosen = []
            for cap in need:
                cand = [r for r in free if cap in r.caps and r not in chosen]
                if cand:
                    chosen.append(min(cand, key=lambda r: np.linalg.norm(r.xy - c.xy)))
            if len(chosen) >= len(need):        # 필요 능력 다 충족
                c.committed = True
                task = self._task(c.gt_tid)
                if task: task.t_found = task.t_found or now
                for r in chosen:
                    r.mode = "detour"; r.target_cid = c.cid
                    nn = self.env.nearest_node(c.xy)
                    try:
                        p = nx.shortest_path(self.env.G, r.node, nn, weight="length")
                        r.path = [self.env.node_xy[x] for x in p[1:]] + [c.xy.copy()]
                    except Exception:
                        r.path = [c.xy.copy()]

    def _task(self, tid):
        return next((t for t in self.tasks if t.tid == tid), None)

    # ── 도착 처리 + 실행 ──
    def _arrive(self, r: GRobot, now):
        c = next((x for x in self.board if x.cid == r.target_cid), None)
        if c is None:
            r.mode = "patrol"; return
        task = self._task(c.gt_tid)
        if task is None or not task.active:
            r.mode = "patrol"; return
        if np.linalg.norm(r.xy - task.xy) > 1.0:
            return                             # 아직 도착 전
        # 오개입 체크: 실제로 비개입/보고 대상인데 개입하러 옴
        if task.gt_class in ("nontask", "report"):
            self.mis += 1
            self.log.append((now, "misintervention", task.tid, task.gt_class))
            task.t_done = now; r.mode = "patrol"; return
        task.arrived.add(r.id)
        if len(task.arrived) >= max(1, task.n):
            if task.work_until is None:
                task.work_until = now + task.c
            r.mode = "working"; r.work_until = task.work_until

    # ── 메인 루프 ──
    def run(self):
        for now in np.arange(0, self.horizon, DT):
            for r in self.robots.values():
                if r.mode == "working":
                    if now >= r.work_until:
                        task = next((t for t in self.tasks
                                     if r.id in t.arrived and t.active), None)
                        if task and now >= task.work_until:
                            task.t_done = now
                            self.log.append((now, "done", task.tid, task.kind))
                        r.mode = "patrol"; r.work_until = None
                    continue
                if r.mode == "patrol" and not r.path:
                    r.path = self._next_patrol_node(r)
                self._move(r, now)
                if r.mode == "detour" and not r.path:
                    self._arrive(r, now)
                self._sense(r, now)
            self._dispatch(now)
        return self.metrics()

    def metrics(self):
        done = [t for t in self.tasks if t.gt_class == "task" and t.t_done]
        real = [t for t in self.tasks if t.gt_class == "task"]
        found = [t for t in self.tasks if t.t_found]
        idle = np.mean(list(self.node_last.values()))
        return {
            "tasks_real": len(real),
            "completed": len(done),
            "completion_rate": round(len(done)/max(1, len(real)), 3),
            "found": len(found),
            "misintervention": self.mis,
            "mean_node_lastvisit": round(float(idle), 1),
            "board_candidates": len(self.board),
        }


if __name__ == "__main__":
    import sys, time
    t0 = time.time()
    ep = GraphEpisode(sys.argv[1] if len(sys.argv) > 1 else "exp/configs/hospital.yaml",
                      seed=1, horizon_s=6*3600, warmup_s=600)
    m = ep.run()
    print(f"6시간 병원 순찰 에피소드 — {time.time()-t0:.1f}초")
    for k, v in m.items():
        print(f"  {k}: {v}")
