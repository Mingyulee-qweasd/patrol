"""에피소드 러너 — full arm의 상태기계. 판단 1~5와 랑데뷰 프로토콜을 엮는다.

이벤트 로그(trace)가 지표·시각화의 단일 원천 (그림 역산 스키마).
"""
from dataclasses import dataclass, field

import numpy as np

from .env import load_env
from .geometry import path_length_world
from .memory import RobotMemory
from .metrics import IdlenessMap
from .policy import Params, gate, dispatch, early_convoke, next_interval, auction
from .rendezvous import RendezvousManager
from .robot import Robot, advance_hub, advance_flanker, nearest_sweep_d
from .tasks import TaskStream
from .vlm import VLMSampler, synthetic_model
from .arms import Arm, ARMS

ARRIVE_R = 3.0     # 도착 판정 반경
MEET_R = 10.0      # 랑데뷰 집결 판정 반경
WAIT_TIMEOUT_S = 1200.0  # 현장 대기 시한 (노선 끝→끝 이동 ~1000s + 여유) — 순환 대기 교착 방지


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
                 arm: Arm | str = "full", lambda_calib: float | None = None):
        self.arm = ARMS[arm] if isinstance(arm, str) else arm
        self.env = load_env(cfg_path)
        self.p = params or Params()
        self.horizon = horizon_s
        if lambda_calib is not None:  # 부하 스윕용 오버라이드 (yaml 기본값 대체)
            self.env.task_cfg["lambda_calib"] = lambda_calib
        self.stream = TaskStream(self.env, rho, seed, horizon_s, warmup_s)
        self.vlm = {rid: VLMSampler(error_model or synthetic_model(kinds_from_env(self.env)),
                                    seed * 7919 + rid)
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

        d0_s = e.route.length / e.hub_v  # 기본 간격 = 스윕 1패스 시간
        self.rdv = RendezvousManager(e, d0_s, e.grace_frac)
        self.rdv.schedule(0.0, 0.0, +1, d0_s)
        self.d0_s = d0_s
        self.coalition = {}   # cid -> {"xy", "crew": set 배정, "arrived": set}
        self.wait_since = {}  # rid -> wait_site 진입 시각 (교착 방지 시한용)
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
            j = self.vlm[rid].observe(task.kind, band)
            if j is None:
                continue
            c = self.mem[rid].update(task.xy, j, band, now, task.tid)
            if task.t_found is None and c.s >= self.p.theta_act:
                task.t_found = now
                self.trace.log(now, "found", tid=task.tid, rid=rid, tkind=task.kind)
            self.decide(rid, c, now)

    def decide(self, rid: int, c, now: float):
        r = self.robots[rid]
        if c.committed or r.mode == "detour":
            return
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
        else:
            decision = dispatch(c, r.xy, r.v, t_to_rdv, can_return, self.p)
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

    def _resume_at_idle(self, rid: int, now: float = 0.0):
        """복귀 지점 = 현 위치 주변(±250m 스윕창)에서 방치 최대 지점 — 순간이동 없이 걸어서 (판단3)."""
        r = self.robots[rid]
        self._maybe_swap_role(r, now)
        if r.role == "hub" or self.arm.sebs_patrol:
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
        c = next((x for x in mem.items if x.cid == cid), None)
        r.mode = "patrol"; r.target = None; r.target_cid = -1
        self._resume_at_idle(rid, now)
        if c is None:
            return
        task = self.tasks_by_id.get(c.gt_tid)
        if task is None or not task.active:
            return
        # 근접 최종 관측 (오류 모델 near 행)
        j = self.vlm[rid].observe(task.kind, "near")
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
        if task.gt_class in ("nontask", "hazard"):
            sev = "severe" if task.gt_class == "hazard" else "minor"
            task.t_done = now  # 개입해버림 (오개입)
            self.trace.log(now, "misintervention", rid=rid, tid=task.tid, sev=sev)
            return
        # 진짜 task — 인원 충분?
        task.arrived.add(rid)
        if len(task.arrived) >= task.n:
            task.t_done = now
            task.served_by = tuple(task.arrived)
            self.trace.log(now, "complete", tid=task.tid, rid=rid, tkind=task.kind,
                           delay=now - task.t_spawn)
        elif self.coalition.get(c.cid) and len(self.coalition[c.cid]) >= task.n:
            # 배정 조가 오는 중 — 현장 대기 (동료 도착 시 완료, 시한 초과 시 복귀)
            r.mode = "wait_site"; r.target = task.xy.copy(); r.target_cid = c.cid
            self.wait_since[rid] = now
            self.trace.log(now, "wait_site", rid=rid, tid=task.tid)
        else:
            c.n_hat = task.n  # 근접 확정 (2단 갱신)
            c.committed = False; c.agenda = True
            self.trace.log(now, "bounce", rid=rid, tid=task.tid, need=task.n)

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
            due = (not self.arm.broadcast) and self.rdv.is_due(now)
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
                if r.mode == "wait_site":
                    c = next((x for x in self.mem[rid].items if x.cid == r.target_cid), None)
                    task = self.tasks_by_id.get(c.gt_tid) if c else None
                    if task is not None and task.active:
                        task.arrived.add(rid)
                        if len(task.arrived) >= task.n:
                            task.t_done = now
                            task.served_by = tuple(task.arrived)
                            self.trace.log(now, "complete", tid=task.tid, rid=rid,
                                           tkind=task.kind, delay=now - task.t_spawn)
                            r.mode = "patrol"; r.target = None; r.target_cid = -1
                        elif now - self.wait_since.get(rid, now) > WAIT_TIMEOUT_S:
                            # 시한 초과 — 배정조가 안 옴 (다른 현장 대기 등 순환 교착 가능성)
                            # → 인원부족 복귀와 동일 처리: 도착 철회, 재안건, 순찰 복귀
                            task.arrived.discard(rid)
                            if c.cid in self.coalition:
                                self.coalition[c.cid].discard(rid)
                            c.n_hat = task.n; c.committed = False; c.agenda = True
                            r.mode = "patrol"; r.target = None; r.target_cid = -1
                            self.trace.log(now, "wait_timeout", rid=rid, tid=task.tid)
                            self._resume_at_idle(rid, now)
                    else:
                        r.mode = "patrol"; r.target = None; r.target_cid = -1
                elif r.mode == "detour":
                    if r.move_towards(r.target):
                        self.on_arrival(rid, r.target_cid, now)
                elif due and r.mode != "at_rdv":
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
                    if r.role == "hub":
                        advance_hub(r, self.env)
                        r.hub_s_sync = r.hub_s
                    else:
                        advance_flanker(r, self.env, self.wpath[rid], self.wcum[rid])
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
                    else:  # 0~1대: 회의 무산 — 다음 만남만 예약 (기억 병합 없음)
                        self.rdv.schedule(now, self._hub().hub_s,
                                          self._hub().hub_dir, self.d0_s)
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
