"""지표 계산 — 논문 그림의 원료. trace(사건 로그) + 에피소드 상태에서 산출.

인용 계보 (문헌정독 §6b):
  idleness 정의·정상상태 집계 = Portugal & Rocha 식1~5 관행
  오개입 채점 TPR/FPR 프레임 = AESOP (개입해야 할 것에 개입=TP, 함정 기각=TN)
  지연-보상 곡선 = Agmon (긴급도 가중)
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class EpisodeMetrics:
    # 순찰 커버리지
    idleness_avg: float = 0.0
    idleness_max: float = 0.0
    # 임무 처리
    n_tasks: int = 0
    n_completed: int = 0
    completion_rate: float = 0.0
    delay_mean: float = 0.0          # 발생→완료 (완료분)
    delay_p95: float = 0.0
    weighted_delay: float = 0.0      # 긴급도 가중 (미완료는 관측 종료까지로 벌점)
    # 개입 판단 (AESOP 프레임)
    n_misintervention_minor: int = 0   # 애매 사례에 개입
    n_misintervention_severe: int = 0  # 위험물에 개입
    n_abort: int = 0                   # 가보고 임무 아님 판단 (헛걸음 — 올바른 기각)
    # 기제 동작
    n_bounce: int = 0                  # 인원 오판 회송
    n_convoke: int = 0                 # 긴급 소집 발화
    n_rendezvous: int = 0
    n_dispatch_now: int = 0
    n_assign: int = 0
    backlog_end: int = 0               # 종료 시점 미처리 안건 수
    # 실행 품질
    robot_travel_s: float = 0.0        # 임무 이탈 총 시간 (커버리지 비용의 원료)

    def as_dict(self):
        return {k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


def compute(ep, steady_frac: float = 0.15) -> EpisodeMetrics:
    """에피소드 종료 후 지표 산출. steady_frac: 앞쪽 과도기 버림 (정상상태 집계)."""
    m = EpisodeMetrics()
    ev = ep.trace.events
    horizon = ep.horizon
    t_min = horizon * steady_frac  # 과도기 컷

    # ── 사건 카운트 ──
    for e in ev:
        if e["t"] < t_min:
            continue
        k = e["e"]
        if k == "misintervention":
            if e.get("sev") == "severe":
                m.n_misintervention_severe += 1
            else:
                m.n_misintervention_minor += 1
        elif k == "abort_nontask":
            m.n_abort += 1
        elif k == "bounce":
            m.n_bounce += 1
        elif k == "convoke":
            m.n_convoke += 1
        elif k == "rendezvous":
            m.n_rendezvous += 1
        elif k == "dispatch_now":
            m.n_dispatch_now += 1
        elif k == "assign":
            m.n_assign += 1

    # ── 임무 지연 (정상상태 발생분만) ──
    tasks = [x for x in ep.stream.tasks
             if x.gt_class == "task" and t_min <= x.t_spawn <= horizon * 0.85]
    # 뒤쪽 15%도 컷: 관측 종료 직전 발생분은 완료 기회가 없어 편향
    m.n_tasks = len(tasks)
    delays, wdelays = [], []
    for x in tasks:
        if x.t_done is not None:
            d = x.t_done - x.t_spawn
            m.n_completed += 1
            delays.append(d)
        else:
            d = horizon - x.t_spawn  # 미완료 벌점: 관측 종료까지 대기한 것으로
        wdelays.append(x.u * d)
    m.completion_rate = m.n_completed / max(m.n_tasks, 1)
    if delays:
        m.delay_mean = float(np.mean(delays))
        m.delay_p95 = float(np.percentile(delays, 95))
    m.weighted_delay = float(np.mean(wdelays)) if wdelays else 0.0
    m.backlog_end = sum(1 for x in ep.stream.tasks
                        if x.gt_class == "task" and x.t_done is None
                        and x.t_spawn <= horizon)
    return m


class IdlenessMap:
    """회랑 셀별 방치 시간 — runner가 매초 갱신 호출.

    리셋 = 관측 품질 충족 시만 (reliable_r 이내). 정상상태 집계용 시계열 축적.
    """
    def __init__(self, env, sample_every: float = 60.0):
        pts = []
        for side, w in env.corridor_width.items():
            sign = 1 if side == "left" else -1
            for s in np.arange(0, env.route.length, env.cell_m):
                for t_off in np.arange(2.0, w, env.cell_m):
                    pts.append(env.route.frame_to_world(s, sign * t_off))
        self.cells = np.array(pts)
        self.last_seen = np.zeros(len(pts))
        self.reliable_r = env.reliable_r
        self.sample_every = sample_every
        self.samples_avg = []
        self.samples_max = []
        self._next_sample = 0.0

    def update(self, robot_xys: list, now: float):
        for xy in robot_xys:
            d = np.linalg.norm(self.cells - np.asarray(xy), axis=1)
            self.last_seen[d <= self.reliable_r] = now
        if now >= self._next_sample:
            idle = now - self.last_seen
            self.samples_avg.append(float(idle.mean()))
            self.samples_max.append(float(idle.max()))
            self._next_sample = now + self.sample_every

    def steady_stats(self, steady_frac: float = 0.15) -> tuple[float, float]:
        k = int(len(self.samples_avg) * steady_frac)
        avg = self.samples_avg[k:] or [0.0]
        mx = self.samples_max[k:] or [0.0]
        return float(np.mean(avg)), float(np.max(mx))
