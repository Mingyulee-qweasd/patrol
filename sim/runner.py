"""에피소드 러너 — full arm의 상태기계. 판단 1~5와 랑데뷰 프로토콜을 엮는다.

이벤트 로그(trace)가 지표·시각화의 단일 원천 (그림 역산 스키마).
"""
from dataclasses import dataclass, field

import numpy as np

from .env import load_env
from .geometry import path_length_world
from .memory import RobotMemory, MATCH_RADIUS
from .metrics import IdlenessMap
from .policy import Params, gate, dispatch, early_convoke, next_interval, auction
from .rendezvous import RendezvousManager
from .robot import Robot, advance_hub, advance_flanker, nearest_sweep_d
from .tasks import TaskStream
from .vlm import VLMSampler, synthetic_model
from .arms import Arm, ARMS

ARRIVE_R = 3.0     # 도착 판정 반경
MEET_R = 10.0      # 랑데뷰 집결 판정 반경
COMM_R = 30.0      # 근접 통신 반경 [m] — 조우 조정의 물리 전제 (근거리 무선 보수 추정)
WAIT_TIMEOUT_S = 1800.0  # 현장 대기 시한 (끝→끝 이동 ~1000s + 앞 작업 최대 600s + 여유) — 교착 방지


@dataclass
class Trace:
    events: list = field(default_factory=list)

    def log(self, t, kind, **kw):
        self.events.append({"t": t, "e": kind, **kw})


def kinds_from_env(env) -> dict:
    kinds = {k: {"n": v["n"], "u": v["u"], "gt_class": "task"}
             for k, v in env.task_cfg["types"].items()}
    kinds["ambiguous"] = {"n": 0, "u": 0, "gt_class": "nontask"}
    kinds["hazard"] = {"n": 0, "u": 0, "gt_class": "hazard"}
    return kinds


class Episode:
    def __init__(self, cfg_path: str, seed: int, horizon_s: float = 10800,
                 warmup_s: float = 1200, rho: float = 0.5,
                 params: Params | None = None, error_model: dict | None = None,
                 arm: Arm | str = "full", lambda_calib: float | None = None,
                 c_mult: float = 1.0, formation: str = "rail"):
        self.arm = ARMS[arm] if isinstance(arm, str) else arm
        self.env = load_env(cfg_path)
        self.p = params or Params()
        self.horizon = horizon_s
        if lambda_calib is not None:  # 부하 스윕용 오버라이드 (yaml 기본값 대체)
            self.env.task_cfg["lambda_calib"] = lambda_calib
        if c_mult != 1.0:             # 처리 시간 배율 스윕 (조정 가치의 교차점 탐색)
            for spec in self.env.task_cfg["types"].values():
                spec["c"] = spec.get("c", 0.0) * c_mult
        self.stream = TaskStream(self.env, rho, seed, horizon_s, warmup_s)
        self.vlm = {rid: VLMSampler(error_model or synthetic_model(kinds_from_env(self.env)),
                                    seed * 7919 + rid, world_seed=seed)
                    for rid in range(3)}
        self.mem = {rid: RobotMemory(rid) for rid in range(3)}
        self.trace = Trace()

        e = self.env
        self.robots = {
            0: Robot(0, "hub", e.route.point_at(0.0).copy(), e.hub_v),
            1: Robot(1, "left", e.route.frame_to_world(0, 2.0).copy(), e.v_sweep),
            2: Robot(2, "right", e.route.frame_to_world(0, -2.0).copy(), e.v_sweep),
        }
        self.wpath, self.wcum = {}, {}
        for rid, side in [(1, "left"), (2, "right")]:
            pts = np.array([e.route.frame_to_world(s, t) for s, t in e.sweep_paths[side]])
            self.wpath[rid] = pts
            self.wcum[rid] = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])

        # v2 이종 수거: env.collect 존재 시 — 로봇0 = 대형 집하(레일·자가수거), 1·2 = 소형(용량)
        self.v2 = bool(self.env.collect)
        if self.v2:
            self.cap = int(self.env.collect.get("small_capacity", 5))
            self.dump_s = float(self.env.collect.get("dump_s", 30))
            self.load_s = float(self.env.collect.get("load_s", 60))
            self.arm_r = float(self.env.collect.get("truck_arm_r", 8))
            self.env.hub_v = float(self.env.collect.get("truck_v", 1.5))  # 대형은 자체 속도
            self.robots[0].v = self.env.hub_v
        if self.v2 and self.arm.fixed_bin:
            formation = "three_sweep"  # 대형 없음 — 셋 다 소형 (폭 3분할)
            self.robots[0].v = e.v_sweep
            self.bin_xy = e.route.point_at(e.route.length / 2)  # 고정 수거함 = 노선 중앙
        else:
            self.bin_xy = None
        # 대형 비교: "rail"(현행: 가운데 직선) vs "three_sweep"(전원 톱니 — 폭 3분할)
        self.three_sweep = (formation == "three_sweep")
        if self.three_sweep:
            def band_path(t_lo, t_hi, spacing=25.0):  # spacing은 yaml과 동일 값
                ss = np.arange(0.0, e.route.length + 1e-6, spacing)
                wp = []
                for k, sv in enumerate(ss):
                    pair = [(sv, t_lo), (sv, t_hi)] if k % 2 == 0 else [(sv, t_hi), (sv, t_lo)]
                    wp.extend(pair)
                return np.array([e.route.frame_to_world(sv, tv) for sv, tv in wp])
            w = max(e.corridor_width.values())
            for rid, (lo, hi) in {0: (-15.0, 15.0), 1: (15.0, w), 2: (-w, -15.0)}.items():
                pts = band_path(lo, hi)
                self.wpath[rid] = pts
                self.wcum[rid] = np.concatenate(
                    [[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
            self.robots[0].v = e.v_sweep  # 가운데도 순찰 속도로 톱니

        d0_s = e.route.length / e.hub_v  # 기본 간격 = 스윕 1패스 시간
        self.rdv = RendezvousManager(e, d0_s, e.grace_frac)
        self.rdv.schedule(0.0, 0.0, +1, d0_s)
        self.d0_s = d0_s
        self.coalition = {}   # cid -> {"xy", "crew": set 배정, "arrived": set}
        self.wait_since = {}  # rid -> wait_site 진입 시각 (교착 방지 시한용)
        self.enc_active = {}  # (a,b) -> 조우 중 여부 (조우당 협의 1회)
        self._dump_until = {}  # rid -> 하역 완료 시각 (v2)
        self._last_dump = {}   # rid -> 마지막 하역 시각 (timer-dump 비교군용)
        self.snaps = []       # 동작 확인 동영상용 장면 기록
        self.idle_map = IdlenessMap(self.env)
        self.tasks_by_id = {x.tid: x for x in self.stream.tasks}

    # ── 감지·판단 사이클 ─────────────────────────────────────────
    def sense(self, rid: int, now: float):
        r = self.robots[rid]
        for task in self.stream.spawned(now):
            d = float(np.linalg.norm(task.xy - r.xy))
            if d > self.env.sense_r:
                continue
            band = "near" if d <= self.env.reliable_r else "far"
            j = self.vlm[rid].observe(task.kind, band, oid=task.tid)
            if j is None:
                continue
            c = self.mem[rid].update(task.xy, j, band, now, task.tid)
            if task.t_found is None and c.s >= self.p.theta_act:
                task.t_found = now
                self.trace.log(now, "found", tid=task.tid, rid=rid, tkind=task.kind)
            self.decide(rid, c, now)

    def decide(self, rid: int, c, now: float):
        r = self.robots[rid]
        if c.committed or r.mode in ("detour", "wait_site", "working", "dumping", "staged_wait"):
            return  # 이동·대기·작업 중에는 새 결정으로 이탈 금지
        if self.v2:
            if rid == 0 and not self.arm.fixed_bin:
                return  # 대형은 자체 정책 (레일+자가수거+상차 대응) — decide 미사용
            task = self.tasks_by_id.get(c.gt_tid)
            if task is not None:
                if self.arm.fixed_bin and task.carry == "loadable":
                    return  # 트럭 없인 소파 수거 물리적 불가 (이 팀의 구조적 한계 — 정직 기재)
                if task.carry == "portable" and r.cargo >= self.cap:
                    return  # 만적 — 새 수거 불가 (판단6이 하역을 유도)
        if not self.arm.use_gate:
            g = "act" if c.s_logodds > 0 else "drop"
        else:
            g = gate(c, self.p)
        if g == "reobserve":
            c.reobserve = True
            return
        if g == "drop":
            return
        # 판단2 — 즉시/안건
        t_to_rdv = max(self.rdv.next.t - now, 0.0)
        can_return = True  # v1: 유예·레일 회복이 안전망이라 완화
        if not self.arm.use_sizing:
            decision = "now"   # 반응식: n̂ 무시하고 혼자 출동 (부족하면 현장에서 판명)
        elif not self.arm.use_agenda and c.n_hat >= 2:
            decision = "skip"  # solo-only: 다수 임무는 다룰 수단 없음
        elif self.arm.encounter_coord and c.near_confirmed and c.n_hat >= 2:
            c.agenda = True
            decision = "now"  # 조우 약속이 생기기 전까진 반응식처럼 재시도 (겹침 채널 유지)
                              # — 조우가 나면 2자 약속이 이 후보를 선점(committed)
        else:
            decision = dispatch(c, r.xy, r.v, t_to_rdv, can_return, self.p)
            if decision == "agenda" and not c.near_confirmed:
                # 접근 우선 조정 (경로 3): 원거리 n̂은 무정보(P0 실측)이므로 그걸로 안건화하지
                # 않는다 — 먼저 접근해 근접 확정 후, n̂≥2로 판명되면 그때 안건화 (도착 처리에서)
                decision = "now"
        if decision == "skip":
            return
        if decision == "now":
            c.committed = True
            r.mode = "detour"; r.target = c.xy.copy(); r.target_cid = c.cid
            self.trace.log(now, "dispatch_now", rid=rid, cid=c.cid)
        else:
            if not c.agenda:
                c.agenda = True
                self.trace.log(now, "agenda", rid=rid, cid=c.cid, n_hat=c.n_hat, u_hat=c.u_hat)
            if self.arm.adaptive_rdv and not c.convoked and early_convoke(c, t_to_rdv, self.p):
                c.convoked = True
                new_t = now + 0.5 * self.d0_s
                if new_t < self.rdv.next.t:  # 앞당기기만 허용 (뒤로 밀기 금지)
                    self.rdv.schedule(now, self._hub().hub_s, self._hub().hub_dir,
                                      0.5 * self.d0_s)
                    self.trace.log(now, "convoke", rid=rid, cid=c.cid)

    # ── 도착 처리 (개입·회송·오개입) ─────────────────────────────
    def _hub(self):
        return next(r for r in self.robots.values() if r.role == "hub")

    def _maybe_swap_role(self, r, now: float):
        """role-adaptive: 임무에서 복귀할 때, 자리를 비운 로봇들의 자리(역할+경로+진행도)가
        내 원래 자리보다 가까우면 통째로 맞바꿈 — 코얼리션 복귀 시 좌우 교차 이동 낭비 제거."""
        if not self.arm.adaptive_roles:
            return
        away = [o for o in self.robots.values()
                if o.rid != r.rid and o.mode in ("detour", "wait_site") and o.target_cid >= 0]

        def slot_dist(owner):  # 그 자리의 현 진행 지점까지의 거리
            if owner.role == "hub":
                return float(np.linalg.norm(self.env.route.point_at(owner.hub_s) - r.xy))
            pts, cum = self.wpath[owner.rid], self.wcum[owner.rid]
            i = int(np.argmin(np.abs(cum - owner.sweep_d)))
            return float(np.linalg.norm(pts[i] - r.xy))

        best = min(away, key=slot_dist, default=None)
        if best is None or slot_dist(best) + 20.0 >= slot_dist(r):  # 20m 이상 이득일 때만
            return
        # 스윕 경로는 역할에 붙는다 — 교환 후 rid 재귀속
        paths = {o.role: (self.wpath.pop(o.rid), self.wcum.pop(o.rid))
                 for o in (r, best) if o.role != "hub"}
        r.role, best.role = best.role, r.role
        for f in ("sweep_d", "sweep_dir", "hub_s", "hub_dir", "v"):
            a, b = getattr(r, f), getattr(best, f)
            setattr(r, f, b); setattr(best, f, a)
        for o in (r, best):
            if o.role != "hub":
                self.wpath[o.rid], self.wcum[o.rid] = paths[o.role]
        self.trace.log(now, "role_swap", rid=r.rid, took=r.role, other=best.rid)

    def _finish_semantics(self, task, now: float):
        """v2: 작업 c 완료가 곧 임무 종료가 아닌 경우 처리. True = 종료 지연(적재 대기)."""
        if not self.v2:
            return False
        if task.carry == "portable":
            for orid in task.arrived:
                if orid != 0:
                    self.robots[orid].cargo += 1  # 소형이 실음 (대형 자가수거는 무제한 통)
            return False
        if task.carry == "loadable" and task.staged_at is None:
            # 소형 2가 들어올려 도로변에 둠 → 둘 다 현장에서 대형을 기다림 (전용 모드)
            task.staged_at = now
            task.xy = self.env.route.point_at(self.env.route.project(task.xy))
            for orid in list(task.arrived):
                if orid != 0:
                    o = self.robots[orid]
                    o.mode = "staged_wait"; o.work_tid = task.tid
                    o.xy = task.xy.copy()  # 함께 도로변으로 끌고 온 것
                    self.wait_since[orid] = now
            self.trace.log(now, "staged", tid=task.tid)
            return True
        return False

    def _start_work(self, task, rid: int, now: float):
        """필요 인원 집결 → 처리 작업 개시: n대가 c초 동안 현장에 묶임 (c=0이면 즉시 완료)."""
        if self.v2 and task.carry == "loadable" and task.staged_at is not None:
            return  # 이미 들어올려 적재 대기 중 — 대형 상차만 남음 (재작업 금지)
        if task.c <= 0:
            task.t_done = now
            task.served_by = tuple(task.arrived)
            self.trace.log(now, "complete", tid=task.tid, rid=rid, tkind=task.kind,
                           delay=now - task.t_spawn)
            return
        task.work_until = now + task.c
        self.trace.log(now, "work_start", tid=task.tid, crew=tuple(task.arrived), c=task.c)
        for orid in task.arrived:
            o = self.robots[orid]
            o.mode = "working"; o.work_tid = task.tid
            o.target = None; o.target_cid = -1

    def _resume_at_idle(self, rid: int, now: float = 0.0):
        """복귀 지점 = 현 위치 주변(±250m 스윕창)에서 방치 최대 지점 — 순간이동 없이 걸어서 (판단3)."""
        r = self.robots[rid]
        self._maybe_swap_role(r, now)
        if (r.role == "hub" and not self.three_sweep) or self.arm.sebs_patrol:
            if r.role == "hub":
                # 이탈 지점의 옛 진행도가 남아 순간이동성 복귀가 되는 것 방지 — 현 위치를 레일에 투영
                r.hub_s = self.env.route.project(r.xy)
            return
        pts, cum = self.wpath[rid], self.wcum[rid]
        d_near = float(cum[int(np.argmin(np.linalg.norm(pts - r.xy, axis=1)))])
        win = (cum > d_near - 250) & (cum < d_near + 250)
        if not win.any():
            r.sweep_d = d_near
            return
        cells, last = self.idle_map.cells, self.idle_map.last_seen
        best_i, best_idle = None, -1.0
        for i in np.where(win)[0][::4]:  # 4점 간격 샘플
            ci = int(np.argmin(np.linalg.norm(self.idle_map.cells - pts[i], axis=1)))
            idle = -last[ci]
            if idle > best_idle:
                best_idle, best_i = idle, i
        if best_i is None:
            r.sweep_d = d_near
            return
        r.sweep_d = float(cum[best_i])
        # 그 지점까지는 걸어가서 재개 (순간이동 금지)
        if np.linalg.norm(pts[best_i] - r.xy) > 15:
            r.mode = "detour"; r.target = pts[best_i].copy(); r.target_cid = -2  # -2 = 순찰 복귀 지점

    def on_arrival(self, rid: int, cid: int, now: float):
        r = self.robots[rid]
        if cid == -2:  # 순찰 복귀 지점 도착
            r.mode = "patrol"; r.target = None; r.target_cid = -1
            return
        mem = self.mem[rid]
        # 후보 식별은 '현재 서 있는 자리' 기준 — cid는 로봇별 일련번호라 배정 경유 도착에서
        # 남의 번호가 내 다른 후보를 가리키는 혼선이 있었음 (#26). 번호+위치 일치 우선, 위치 차선.
        c = next((x for x in mem.items
                  if x.cid == cid and np.linalg.norm(x.xy - r.xy) < MATCH_RADIUS), None)
        if c is None:
            c = next((x for x in mem.items
                      if np.linalg.norm(x.xy - r.xy) < MATCH_RADIUS), None)
        r.mode = "patrol"; r.target = None; r.target_cid = -1
        self._resume_at_idle(rid, now)
        if c is None:
            return
        task = self.tasks_by_id.get(c.gt_tid)
        if task is None or not task.active:
            return
        # 근접 최종 관측 (오류 모델 near 행)
        j = self.vlm[rid].observe(task.kind, "near", oid=task.tid)
        if not c.committed:
            # 재관측 방문 — 목적이 개입이 아니라 관측: 기억만 갱신하고 판단 재실행 (개입은
            # 문턱+판단2를 통과해 커밋된 도착만 가능 — 방문이 오개입 경로가 되던 결함 수리)
            if j is not None:
                self.mem[rid].update(task.xy, j, "near", now, task.tid)
                self.decide(rid, c, now)
            return
        if j is None or not j["is_task"]:
            self.trace.log(now, "abort_nontask", rid=rid, tid=task.tid)
            return  # 헛걸음 — 개입 안 함
        # 근접 판정으로 기억 갱신 (2단: n̂·û를 근접 값으로 대체 — 접근 우선 조정의 핵심 정보)
        self.mem[rid].update(task.xy, j, "near", now, task.tid)
        if task.gt_class in ("nontask", "hazard"):
            sev = "severe" if task.gt_class == "hazard" else "minor"
            task.t_done = now  # 개입해버림 (오개입)
            self.trace.log(now, "misintervention", rid=rid, tid=task.tid, sev=sev)
            return
        # 진짜 task — 인원 충분?
        task.arrived.add(rid)
        if len(task.arrived) >= task.n:
            self._start_work(task, rid, now)
        elif self.coalition.get(c.cid) and len(self.coalition[c.cid]) >= min(c.n_hat, 2):
            # 배정 조가 오는 중 (자기 추정 기준) — 현장 대기 (동료 도착 시 완료, 시한 초과 시 복귀)
            r.mode = "wait_site"; r.target = task.xy.copy(); r.target_cid = c.cid
            self.wait_since[rid] = now
            self.trace.log(now, "wait_site", rid=rid, tid=task.tid)
        else:
            # 인원부족 복귀: 물리적 실패가 알려주는 것은 "지금 인원으론 부족" — 하한만 상향
            # (종전엔 GT 대수를 그대로 열람하는 이상화 + 떠난 로봇이 도착 명단에 유령으로 잔존 #25)
            c.n_hat = max(c.n_hat, len(task.arrived) + 1)
            task.arrived.discard(rid)
            c.committed = False; c.agenda = True
            self.trace.log(now, "bounce", rid=rid, tid=task.tid, need=task.n)

    def _encounter(self, a: int, b: int, now: float):
        """자연 조우: 수첩 병합 + 즉석 2자 협의 — 집결 이동 0의 조정 (프레임 C 제안 시스템).

        협의 = 병합된 시야에서 근접 확정된 여럿-임무 중 (긴급 우선, 그다음 가까운 것) 하나를
        골라 둘이 함께 출동 약속. n̂=3의 셋째는 후속 조우·현장 대기 시한이 안전망."""
        self.mem[a].merge_from(self.mem[b])
        self.mem[b].merge_from(self.mem[a])
        self.trace.log(now, "encounter", pair=(a, b))
        if self.v2 and 0 in (a, b):
            if self.arm.truck_fetch:
                small = self.robots[b if a == 0 else a]
                if small.cargo >= 2 and small.mode == "patrol":
                    small.mode = "detour"; small.target = self.robots[0].xy.copy()
                    small.target_cid = -3  # 조우를 하역 기회로 (30m → 접촉까지 접근)
                    self.trace.log(now, "fetch_dump", rid=small.rid, cargo=small.cargo)
            return  # 대형은 레일 전용 — 조우는 정보 전파만 (상차·하역은 물류 틱이 담당)
        ra, rb = self.robots[a], self.robots[b]
        # ① 합류: 한쪽이 여럿-임무로 가는 중이거나 현장에서 대기 중이면 파트너가 동행
        #    (대기 로봇 옆을 지나는 순찰 로봇이 통신 반경에 들어온 것도 조우)
        for rid_c, rid_p in ((a, b), (b, a)):
            rc, rp = self.robots[rid_c], self.robots[rid_p]
            if rc.mode not in ("detour", "wait_site") or rp.mode != "patrol" or rc.target_cid < 0:
                continue
            cc = next((x for x in self.mem[rid_c].items if x.cid == rc.target_cid), None)
            if cc is None or cc.n_hat < 2 or not cc.near_confirmed:
                continue
            task = self.tasks_by_id.get(cc.gt_tid)
            if task is None or not task.active:
                continue
            cp = next((x for x in self.mem[rid_p].items
                       if np.linalg.norm(x.xy - cc.xy) < MATCH_RADIUS), None)
            self.coalition[cc.cid] = {a, b}
            if cp is not None:
                cp.committed = True
                self.coalition[cp.cid] = {a, b}
            rp.mode = "detour"; rp.target = cc.xy.copy()
            rp.target_cid = cp.cid if cp is not None else cc.cid
            self.trace.log(now, "pair_join", pair=(rid_c, rid_p), tid=cc.gt_tid)
            return
        if ra.mode != "patrol" or rb.mode != "patrol":
            return  # (합류 불가 시) 둘 다 손이 비어 있을 때만 즉석 약속
        mid = (ra.xy + rb.xy) / 2
        best, best_key = None, None
        for c in self.mem[a].items:
            if c.committed or not c.near_confirmed or c.n_hat < 2:
                continue
            if gate(c, self.p) != "act":
                continue
            task = self.tasks_by_id.get(c.gt_tid)
            if task is None or not task.active:
                continue
            key = (-c.u_hat, float(np.linalg.norm(c.xy - mid)))
            if best is None or key < best_key:
                best, best_key = c, key
        if best is None:
            return
        cb = next((x for x in self.mem[b].items
                   if np.linalg.norm(x.xy - best.xy) < MATCH_RADIUS), None)
        best.committed = True
        self.coalition[best.cid] = {a, b}
        if cb is not None:
            cb.committed = True
            self.coalition[cb.cid] = {a, b}
        for rid, c_own in ((a, best), (b, cb or best)):
            r = self.robots[rid]
            r.mode = "detour"; r.target = best.xy.copy(); r.target_cid = c_own.cid
        self.trace.log(now, "pair_assign", pair=(a, b), tid=best.gt_tid, n_hat=best.n_hat)

    def _v2_logistics(self, now: float):
        """v2 물류 틱: 판단6(소형 비우기) + 접촉 하역 + 대형 정책(상차 우선) + 상차 완료."""
        truck = self.robots[0]
        smalls = (0, 1, 2) if self.arm.fixed_bin else (1, 2)
        depot = (lambda: self.bin_xy) if self.arm.fixed_bin else (lambda: truck.xy)
        # ── 접촉 하역: 소형이 하역처 3m 이내 + 짐 있음 → 30s (수거함은 무인, 대형은 정차)
        for rid in smalls:
            r = self.robots[rid]
            truck_ok = self.arm.fixed_bin or truck.mode in ("patrol", "dumping")
            if (r.cargo > 0 and r.mode in ("patrol", "detour") and truck_ok
                    and float(np.linalg.norm(r.xy - depot())) < 3.0):
                r.mode = "dumping"; r.work_tid = -1
                if not self.arm.fixed_bin:
                    truck.mode = "dumping"
                self._dump_until[rid] = now + self.dump_s
                self.trace.log(now, "dump_start", rid=rid, cargo=r.cargo)
        for rid in smalls:
            r = self.robots[rid]
            if r.mode == "dumping" and now >= self._dump_until.get(rid, 0):
                self.trace.log(now, "dump_done", rid=rid, cargo=r.cargo)
                self._last_dump[rid] = now
                r.cargo = 0
                r.mode = "patrol"
                if not self.arm.fixed_bin and all(self.robots[o].mode != "dumping" for o in (1, 2)):
                    truck.mode = "patrol"
                self._resume_at_idle(rid, now)
        # ── 판단6 (비우기 시점) — 변형: no_preempt=가득 찰 때만 / dump_timer=정기 하역
        thr = self.cap if self.arm.no_preempt else max(2, int(0.8 * self.cap))
        for rid in smalls:
            r = self.robots[rid]
            timer_due = (self.arm.dump_timer and r.cargo > 0
                         and now - self._last_dump.get(rid, 0) > 2400)
            if r.mode == "patrol" and (r.cargo >= thr or timer_due):
                r.mode = "detour"; r.target = np.asarray(depot(), float).copy(); r.target_cid = -3
                self.trace.log(now, "empty_run", rid=rid, cargo=r.cargo)
        for rid in smalls:
            r = self.robots[rid]
            if r.mode == "detour" and r.target_cid == -3:
                r.target = np.asarray(depot(), float).copy()  # 이동 하역처 추적
                if r.move_towards(r.target):
                    pass  # 접촉 하역 블록이 3m에서 잡음
        if self.arm.fixed_bin:
            return  # 대형 정책 없음 (수거함은 무인 고정)
        # ── 대형 정책: 적재 대기(staged) 지점 우선, 없으면 레일 순찰(+자가 수거)
        staged = [t for t in self.tasks_by_id.values()
                  if t.active and t.staged_at is not None and t.carry == "loadable"
                  and self._truck_knows(t)]
        if truck.mode == "patrol":
            if staged:
                tgt = min(staged, key=lambda t: abs(self.env.route.project(t.xy) - truck.hub_s))
                s_t = self.env.route.project(tgt.xy)
                if abs(s_t - truck.hub_s) > self.arm_r:
                    truck.hub_dir = 1 if s_t > truck.hub_s else -1  # 방향만 — 이동은 본 루프
                else:
                    # 상차: 소형 2 대기 중이어야 (사용자 확정 의미론)
                    waiters = [o for o in (1, 2)
                               if self.robots[o].mode == "staged_wait"
                               and self.robots[o].work_tid == tgt.tid]
                    if len(waiters) >= 2:
                        tgt.work_until = now + self.load_s
                        tgt.arrived |= {0, 1, 2}
                        truck.mode = "working"; truck.work_tid = tgt.tid
                        for o in waiters:
                            self.robots[o].mode = "working"
                            self.robots[o].work_tid = tgt.tid
                        self.trace.log(now, "load_start", tid=tgt.tid)
        # ── 대형 자가 수거: 팔 범위 내 portable 임무 (근접 판정 후 작업)
        if truck.mode == "patrol":
            for t in self.stream.spawned(now):
                if (t.active and t.carry == "portable" and t.work_until is None
                        and float(np.linalg.norm(t.xy - truck.xy)) < self.arm_r):
                    j = self.vlm[0].observe(t.kind, "near", oid=t.tid)
                    if j is not None and j["is_task"]:
                        t.work_until = now + t.c
                        t.arrived.add(0)
                        truck.mode = "working"; truck.work_tid = t.tid
                        self.trace.log(now, "truck_pickup", tid=t.tid)
                    break

    def _truck_knows(self, task) -> bool:
        """대형이 적재 대기를 아는가 = 자기 기억에 그 후보가 있고 근접 확인됨 (병합으로 전파)."""
        return any(c.gt_tid == task.tid and c.near_confirmed for c in self.mem[0].items)

    def _assign_virtual(self, now: float):
        """broadcast 상한선: 만남 없이 안건 즉시 배정."""
        agenda = []
        for c in self.mem[0].items:
            if c.committed:
                continue
            task = self.tasks_by_id.get(c.gt_tid)
            if task is None or not task.active:
                continue
            if gate(c, self.p) == "act" and c.n_hat >= 2:
                agenda.append(c)
        if not agenda:
            return
        # 현장 대기 중(핀 고정) 로봇은 새 낙찰에서 제외 — 순환 대기 형성 자체를 차단
        free = {rid: r.xy for rid, r in self.robots.items() if r.mode != "wait_site"}
        if not free:
            return
        assign = auction(agenda, free, self.env.v_sweep, self.p)
        for c in agenda:
            crew = assign.get(c.cid, [])
            c.committed = True
            self.coalition[c.cid] = set(crew)
            for rid in crew:
                self.robots[rid].queue.append((c.cid, c.xy.copy(), c.gt_tid))
            self.trace.log(now, "assign", cid=c.cid, crew=crew, n_hat=c.n_hat)

    # ── 랑데뷰 의사일정 ──────────────────────────────────────────
    def hold_meeting(self, now: float, present: list | None = None):
        present = sorted(present) if present is not None else [0, 1, 2]
        # 병합 (참석자 ↔ 참석자)
        for a in present:
            for b in present:
                if a != b:
                    self.mem[a].merge_from(self.mem[b])
        # 재판정 + 안건 수집
        agenda = []
        for c in self.mem[present[0]].items:  # 병합 후 선임 참석자 메모리 = 팀 뷰
            if c.committed:
                continue
            task = self.tasks_by_id.get(c.gt_tid)
            if task is None or not task.active:
                continue
            if gate(c, self.p) == "act" or (c.agenda and c.s > 0.5):
                agenda.append(c)
        # 경매 배정
        if not self.arm.use_agenda:
            agenda = []
        assign = auction(agenda, {rid: self.robots[rid].xy for rid in present},
                         self.env.v_sweep, self.p)
        for c in agenda:
            crew = assign.get(c.cid, [])
            c.committed = True
            self.coalition[c.cid] = set(crew)
            for rid in crew:
                self.robots[rid].queue.append((c.cid, c.xy.copy(), c.gt_tid))
            self.trace.log(now, "assign", cid=c.cid, crew=crew, n_hat=c.n_hat)
        # 차기 랑데뷰 합의 (판단4)
        interval = next_interval(agenda, self.d0_s) if self.arm.adaptive_rdv else self.d0_s
        self.rdv.schedule(now, self._hub().hub_s, self._hub().hub_dir, interval)
        self.trace.log(now, "rendezvous", n_agenda=len(agenda), next_in=interval)

    # ── 메인 루프 ────────────────────────────────────────────────
    def run(self):
        now = 0.0
        while now < self.horizon:
            now += 1.0
            due = (not self.arm.broadcast and not self.arm.encounter_coord) \
                  and self.rdv.is_due(now)
            meet_xy = self.rdv.meet_point_xy()
            for rid, r in self.robots.items():
                # 재관측 예약 들르기 (경로 자율성 — 판단3의 일부)
                if r.mode == "patrol" and not r.queue and not self.arm.sebs_patrol:
                    for c in self.mem[rid].items:
                        if (c.reobserve and not c.committed
                                and c.reobs_visits < 2
                                and now - c.last_seen > 300
                                and np.linalg.norm(c.xy - r.xy) < 60):
                            c.reobs_visits += 1
                            r.mode = "detour"; r.target = c.xy.copy(); r.target_cid = c.cid
                            self.trace.log(now, "reobserve_visit", rid=rid, cid=c.cid)
                            break
                # 큐 소화 (배정 임무) 우선
                if r.mode == "patrol" and r.queue:
                    cid, xy, gt = r.queue.pop(0)
                    r.mode = "detour"; r.target = xy; r.target_cid = cid
                if r.mode == "working":
                    task = self.tasks_by_id.get(r.work_tid)
                    if task is None or task.t_done is not None or task.work_until is None:
                        r.mode = "patrol"; r.work_tid = -1
                        self._resume_at_idle(rid, now)
                    elif now >= task.work_until:
                        if self._finish_semantics(task, now):
                            task.work_until = None  # 전환은 semantics가 두 소형 모두 처리
                        else:
                            task.t_done = now
                            task.served_by = tuple(task.arrived)
                            self.trace.log(now, "complete", tid=task.tid, rid=rid,
                                           tkind=task.kind, delay=now - task.t_spawn)
                            r.mode = "patrol"; r.work_tid = -1
                            self._resume_at_idle(rid, now)
                    continue
                if r.mode == "staged_wait":
                    task = self.tasks_by_id.get(r.work_tid)
                    if task is None or task.t_done is not None:
                        r.mode = "patrol"; r.work_tid = -1
                        self._resume_at_idle(rid, now)
                    elif now - self.wait_since.get(rid, now) > WAIT_TIMEOUT_S:
                        task.staged_at = None  # 내려놓고 철수 — 재적재 필요
                        task.arrived.discard(rid)
                        r.mode = "patrol"; r.work_tid = -1
                        self.trace.log(now, "stage_timeout", rid=rid, tid=task.tid)
                        self._resume_at_idle(rid, now)
                    continue
                if r.mode == "wait_site":
                    c = next((x for x in self.mem[rid].items if x.cid == r.target_cid), None)
                    task = self.tasks_by_id.get(c.gt_tid) if c else None
                    if task is not None and task.active:
                        task.arrived.add(rid)
                        if len(task.arrived) >= task.n:
                            self._start_work(task, rid, now)
                            if r.mode != "working":  # c=0 즉시 완료였으면 순찰 복귀
                                r.mode = "patrol"; r.target = None; r.target_cid = -1
                        elif now - self.wait_since.get(rid, now) > WAIT_TIMEOUT_S:
                            # 시한 초과 — 배정조가 안 옴 (다른 현장 대기 등 순환 교착 가능성)
                            # → 인원부족 복귀와 동일 처리: 도착 철회, 재안건, 순찰 복귀
                            task.arrived.discard(rid)
                            if c.cid in self.coalition:
                                self.coalition[c.cid].discard(rid)
                            c.n_hat = task.n; c.committed = False; c.agenda = True
                            if self.v2 and task.carry == "loadable":
                                task.staged_at = None  # 들어올린 것 내려놓고 철수 — 재적재 필요
                            r.mode = "patrol"; r.target = None; r.target_cid = -1
                            self.trace.log(now, "wait_timeout", rid=rid, tid=task.tid)
                            self._resume_at_idle(rid, now)
                    else:
                        r.mode = "patrol"; r.target = None; r.target_cid = -1
                elif r.mode == "detour":
                    if r.move_towards(r.target):
                        self.on_arrival(rid, r.target_cid, now)
                elif (r.mode != "at_rdv" and not self.arm.broadcast
                      and not self.arm.encounter_coord
                      and self.rdv.next is not None
                      and np.linalg.norm(r.xy - meet_xy) / r.v > (self.rdv.next.t - now) + 5.0):
                    # 집결 리드타임: 정상 순찰 흐름으로는 약속에 지각할 로봇만 조기 직행
                    # (종전엔 약속 시각에야 출발해 유예를 넘김 → 만남 70% 무산 — 일지 #24.
                    #  대형에 맞춰 도는 로봇은 자연 도착하므로 순찰 유지 — 해석 검증 보존)
                    if r.move_towards(meet_xy):
                        r.mode = "at_rdv"
                elif self.arm.sebs_patrol:
                    # 방치 탐욕: 목표 셀 없거나 도달 시, 담당 구역에서 가장 오래 방치된 셀로
                    if r.target is None or r.move_towards(r.target):
                        cells, last = self.idle_map.cells, self.idle_map.last_seen
                        if r.role == "hub":
                            mask = np.abs(cells[:, 1]) < 25
                        else:
                            sign = 1 if r.role == "left" else -1
                            mask = (cells[:, 1] * sign) > 0
                        sub = cells[mask]
                        r.target = sub[int(np.argmin(last[mask]))].copy()
                        r.target_cid = -1
                else:
                    if r.role == "hub" and not self.three_sweep:
                        advance_hub(r, self.env)
                        r.hub_s_sync = r.hub_s
                    else:
                        advance_flanker(r, self.env, self.wpath[rid], self.wcum[rid])
                        if rid == 0 and self.three_sweep:
                            r.hub_s = self.env.route.project(r.xy)  # 만남 앵커 근사 유지
                self.sense(rid, now)
            self.idle_map.update([r.xy for r in self.robots.values()], now)
            if int(now) % 20 == 0:  # 장면 기록 (20초마다)
                self.snaps.append({
                    "t": now,
                    "robots": {rid: (float(r.xy[0]), float(r.xy[1]), r.mode)
                               for rid, r in self.robots.items()},
                    "tasks": [(float(x.xy[0]), float(x.xy[1]), x.kind, x.gt_class)
                              for x in self.stream.spawned(now)],
                    "meet": (self.rdv.next.t, tuple(map(float, meet_xy))),
                })
            if self.arm.broadcast and int(now) % 5 == 0:
                for a in range(3):
                    for b in range(3):
                        if a != b:
                            self.mem[a].merge_from(self.mem[b])
                self._assign_virtual(now)  # 즉시 배정 (물리 회합·이벤트 오염 없음)
            if self.arm.encounter_coord:
                for a in range(3):
                    for b in range(a + 1, 3):
                        near = float(np.linalg.norm(self.robots[a].xy - self.robots[b].xy)) < COMM_R
                        if near and not self.enc_active.get((a, b)):
                            self._encounter(a, b, now)  # 조우당 1회 협의
                        self.enc_active[(a, b)] = near
            if self.v2:
                self._v2_logistics(now)
            # 전원 집결 → 의사일정; 유예 초과 시 부분 회의 (판단2 복귀 추정이 빗나간 경우의 안전망)
            if due:
                present = [rid for rid, r in self.robots.items()
                           if np.linalg.norm(r.xy - meet_xy) < MEET_R or r.mode == "at_rdv"]
                concluded = False
                if len(present) == 3:
                    self.hold_meeting(now)
                    concluded = True
                elif self.rdv.grace_expired(now):
                    if len(present) >= 2:
                        self.hold_meeting(now, present=present)
                        self.trace.log(now, "rdv_partial", present=tuple(present))
                    else:  # 0~1대: 회의 무산 — 절반 간격으로 재시도 (전 간격 벌점 완화)
                        self.rdv.schedule(now, self._hub().hub_s,
                                          self._hub().hub_dir, 0.5 * self.d0_s)
                        self.trace.log(now, "rdv_skipped", present=tuple(present))
                    concluded = True
                if concluded:
                    for rid in present:
                        r = self.robots[rid]
                        if r.mode == "at_rdv":
                            r.mode = "patrol"
                            if r.role != "hub":
                                r.sweep_d = nearest_sweep_d(self.wpath[r.rid], self.wcum[r.rid], r.xy)
                            else:
                                r.hub_s = self.env.route.project(r.xy)
        return self.trace
