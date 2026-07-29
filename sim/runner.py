"""에피소드 러너 — full arm의 상태기계. 판단 1~5와 랑데뷰 프로토콜을 엮는다.

이벤트 로그(trace)가 지표·시각화의 단일 원천 (그림 역산 스키마).
"""
from dataclasses import dataclass, field

import numpy as np

from .env import load_env
from .geometry import path_length_world
from .memory import RobotMemory
from .policy import Params, gate, dispatch, early_convoke, next_interval, auction
from .rendezvous import RendezvousManager
from .robot import Robot, advance_hub, advance_flanker, nearest_sweep_d
from .tasks import TaskStream
from .vlm import VLMSampler, synthetic_model

ARRIVE_R = 3.0     # 도착 판정 반경
MEET_R = 10.0      # 랑데뷰 집결 판정 반경


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
                 params: Params | None = None, error_model: dict | None = None):
        self.env = load_env(cfg_path)
        self.p = params or Params()
        self.horizon = horizon_s
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
        g = gate(c, self.p)
        if g == "reobserve":
            c.reobserve = True
            return
        if g == "drop":
            return
        # 판단2 — 즉시/안건
        t_to_rdv = max(self.rdv.next.t - now, 0.0)
        can_return = True  # v1: 유예·레일 회복이 안전망이라 완화
        if dispatch(c, r.xy, r.v, t_to_rdv, can_return, self.p) == "now":
            c.committed = True
            r.mode = "detour"; r.target = c.xy.copy(); r.target_cid = c.cid
            self.trace.log(now, "dispatch_now", rid=rid, cid=c.cid)
        else:
            if not c.agenda:
                c.agenda = True
                self.trace.log(now, "agenda", rid=rid, cid=c.cid, n_hat=c.n_hat, u_hat=c.u_hat)
            if not c.convoked and early_convoke(c, t_to_rdv, self.p):
                c.convoked = True
                new_t = now + 0.5 * self.d0_s
                if new_t < self.rdv.next.t:  # 앞당기기만 허용 (뒤로 밀기 금지)
                    self.rdv.schedule(now, self.robots[0].hub_s, self.robots[0].hub_dir,
                                      0.5 * self.d0_s)
                    self.trace.log(now, "convoke", rid=rid, cid=c.cid)

    # ── 도착 처리 (개입·회송·오개입) ─────────────────────────────
    def on_arrival(self, rid: int, cid: int, now: float):
        mem = self.mem[rid]
        c = next((x for x in mem.items if x.cid == cid), None)
        r = self.robots[rid]
        r.mode = "patrol"; r.target = None; r.target_cid = -1
        if c is None:
            return
        task = self.tasks_by_id.get(c.gt_tid)
        if task is None or not task.active:
            return
        # 근접 최종 관측 (오류 모델 near 행)
        j = self.vlm[rid].observe(task.kind, "near")
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
            # 배정 조가 오는 중 — 현장 대기 (동료 도착 시 완료)
            r.mode = "wait_site"; r.target = task.xy.copy(); r.target_cid = c.cid
            self.trace.log(now, "wait_site", rid=rid, tid=task.tid)
        else:
            c.n_hat = task.n  # 근접 확정 (2단 갱신)
            c.committed = False; c.agenda = True
            self.trace.log(now, "bounce", rid=rid, tid=task.tid, need=task.n)

    # ── 랑데뷰 의사일정 ──────────────────────────────────────────
    def hold_meeting(self, now: float):
        # 병합 (전원 ↔ 전원)
        for a in range(3):
            for b in range(3):
                if a != b:
                    self.mem[a].merge_from(self.mem[b])
        # 재판정 + 안건 수집
        agenda = []
        for c in self.mem[0].items:  # 병합 후 허브 메모리 = 팀 뷰
            if c.committed:
                continue
            task = self.tasks_by_id.get(c.gt_tid)
            if task is None or not task.active:
                continue
            if gate(c, self.p) == "act" or (c.agenda and c.s > 0.5):
                agenda.append(c)
        # 경매 배정
        assign = auction(agenda, {rid: r.xy for rid, r in self.robots.items()},
                         self.env.v_sweep, self.p)
        for c in agenda:
            crew = assign.get(c.cid, [])
            c.committed = True
            self.coalition[c.cid] = set(crew)
            for rid in crew:
                self.robots[rid].queue.append((c.cid, c.xy.copy(), c.gt_tid))
            self.trace.log(now, "assign", cid=c.cid, crew=crew, n_hat=c.n_hat)
        # 차기 랑데뷰 합의 (판단4)
        interval = next_interval(agenda, self.d0_s)
        self.rdv.schedule(now, self.robots[0].hub_s, self.robots[0].hub_dir, interval)
        self.trace.log(now, "rendezvous", n_agenda=len(agenda), next_in=interval)

    # ── 메인 루프 ────────────────────────────────────────────────
    def run(self):
        now = 0.0
        while now < self.horizon:
            now += 1.0
            due = self.rdv.is_due(now)
            meet_xy = self.rdv.meet_point_xy()
            for rid, r in self.robots.items():
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
                    else:
                        r.mode = "patrol"; r.target = None; r.target_cid = -1
                elif r.mode == "detour":
                    if r.move_towards(r.target):
                        self.on_arrival(rid, r.target_cid, now)
                elif due and r.mode != "at_rdv":
                    if r.move_towards(meet_xy):
                        r.mode = "at_rdv"
                else:
                    if r.role == "hub":
                        advance_hub(r, self.env)
                        r.hub_s_sync = r.hub_s
                    else:
                        advance_flanker(r, self.env, self.wpath[rid], self.wcum[rid])
                self.sense(rid, now)
            # 전원 집결 → 의사일정
            if due and all(np.linalg.norm(r.xy - meet_xy) < MEET_R or r.mode == "at_rdv"
                           for r in self.robots.values()):
                self.hold_meeting(now)
                for r in self.robots.values():
                    if r.mode == "at_rdv":
                        r.mode = "patrol"
                        if r.role != "hub":
                            r.sweep_d = nearest_sweep_d(self.wpath[r.rid], self.wcum[r.rid], r.xy)
                        else:
                            r.hub_s = self.env.route.project(r.xy)
        return self.trace
