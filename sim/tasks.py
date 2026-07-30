"""임무 스트림 생성기 — 부하율 ρ에서 λ를 유도, 회랑 내 무작위 배치.

세계의 정답지(GT): 타입·필요 n·긴급도 u는 환경 YAML의 온톨로지에서.
비개입 항목(정상은 배경이므로 미생성; 애매·위험물만 개체로 생성).
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Task:
    tid: int
    kind: str            # 타입명 (trash/sharps/... 또는 'ambiguous'/'hazard')
    gt_class: str        # 'task' | 'nontask' | 'hazard'
    xy: np.ndarray
    n: int               # GT 필요 대수 (비task는 0)
    u: int               # GT 긴급도
    t_spawn: float
    c: float = 0.0       # GT 처리 시간 [s] — n대가 동시에 이 시간 동안 붙어야 완료
    work_until: float | None = None   # 작업 시작 후 완료 예정 시각
    t_found: float | None = None      # 어느 로봇이든 첫 확신 문턱 통과 시각
    t_done: float | None = None
    served_by: tuple = ()
    # 실행 상태
    arrived: set = field(default_factory=set)   # 현장 도착한 로봇 id들

    @property
    def active(self):
        return self.t_done is None


def derive_lambda(env, mean_travel_s: float, rho: float) -> float:
    """부하율 ρ = (건당 평균 이동 로봇초 × λ) / (3대 가용) → λ [건/초].

    건당 이동 로봇초 ≈ 평균 필요 대수 × 평균 왕복 이동시간 (수치 인자).
    """
    types = env.task_cfg["types"]
    mean_n = sum(t["n"] * t["share"] for t in types.values())
    robot_seconds_per_task = mean_n * mean_travel_s
    return rho * 3.0 / robot_seconds_per_task


class TaskStream:
    def __init__(self, env, rho: float, seed: int, horizon_s: float,
                 warmup_s: float, urgent_bias: float = 0.0):
        """urgent_bias: 0=기본 비중, >0이면 고긴급 타입 비중을 위로 치움 (스윕용)."""
        self.env = env
        rng = np.random.default_rng(seed)
        types = env.task_cfg["types"]
        names = list(types)
        shares = np.array([types[k]["share"] for k in names], float)
        if urgent_bias > 0:
            us = np.array([types[k]["u"] for k in names], float)
            shares = shares * np.exp(urgent_bias * us)
        shares /= shares.sum()

        # 왕복 이동시간 수치 추정: 회랑 폭 절반 평균 × 2 / v (보수적 근사)
        mean_half_w = np.mean([w for w in env.corridor_width.values()]) / 2
        mean_travel = 2 * mean_half_w / env.v_sweep + 30.0  # 노선방향 접근 여유
        lam = derive_lambda(env, mean_travel, rho)
        lam *= env.task_cfg.get("lambda_calib", 1.0)  # 경험 보정 (발견 병목 반영)

        self.tasks: list[Task] = []
        tid = 0
        t = warmup_s if lam > 0 else horizon_s  # λ=0 → 임무 없는 세계 (0 나눗셈 가드)
        while t < horizon_s:
            t += rng.exponential(1.0 / lam)
            if t >= horizon_s:
                break
            kind = str(rng.choice(names, p=shares))
            spec = types[kind]
            self.tasks.append(Task(tid, kind, "task", self._rand_xy(rng),
                                   spec["n"], spec["u"], t, c=spec.get("c", 0.0)))
            tid += 1
        n_task = len(self.tasks)

        # 비개입 개체: 애매(=task 총량 × ratio), 위험물(hazard_share)
        n_amb = int(n_task * env.task_cfg.get("nontask_ratio", 0.5))
        haz_share = env.task_cfg.get("hazard_share", 0.05)
        n_haz = max(1, int(n_task * haz_share)) if haz_share > 0 and n_task > 0 else 0
        for _ in range(n_amb):
            ts = warmup_s + rng.uniform(0, horizon_s - warmup_s)
            self.tasks.append(Task(tid, "ambiguous", "nontask",
                                   self._rand_xy(rng), 0, 0, ts)); tid += 1
        for _ in range(n_haz):
            ts = warmup_s + rng.uniform(0, horizon_s - warmup_s)
            self.tasks.append(Task(tid, "hazard", "hazard",
                                   self._rand_xy(rng), 0, 0, ts)); tid += 1
        self.tasks.sort(key=lambda x: x.t_spawn)
        self.lam = lam

    def _rand_xy(self, rng) -> np.ndarray:
        s = rng.uniform(0, self.env.route.length)
        side = rng.choice(["left", "right"])
        w = self.env.corridor_width[side]
        t_off = rng.uniform(2.0, w) * (1 if side == "left" else -1)
        return self.env.route.frame_to_world(s, t_off)

    def spawned(self, now: float) -> list[Task]:
        return [x for x in self.tasks if x.t_spawn <= now and x.active]
