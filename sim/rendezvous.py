"""랑데뷰 프로토콜 — 설계 §3 상세표의 직역.

예약(전방 지점)·유예·레일 회복·의사일정(병합→재판정→경매→차기 합의)·조기 소집 요격.
"""
from dataclasses import dataclass, field

import numpy as np

from .geometry import hub_pos_at, intercept_hub


@dataclass
class Meeting:
    t: float                # 약속 시각
    s: float                # 약속 지점 (노선 호길이)
    attendees: set = field(default_factory=set)
    opened_at: float | None = None


class RendezvousManager:
    def __init__(self, env, d0_s: float, grace_frac: float):
        """d0_s: 기본 간격(시간). 장소는 허브 궤적상 그 시각 위치 (전방 예약)."""
        self.env = env
        self.d0 = d0_s
        self.grace = grace_frac * d0_s
        self.next: Meeting | None = None

    def schedule(self, now: float, hub_s: float, hub_dir: int, interval_s: float):
        t_meet = now + interval_s
        s_meet = hub_pos_at(hub_s, self.env.hub_v, hub_dir, interval_s,
                            self.env.route, self.env.lap_len)
        self.next = Meeting(t=t_meet, s=s_meet)
        return self.next

    def meet_point_xy(self) -> np.ndarray:
        return self.env.route.point_at(self.next.s)

    def is_due(self, now: float) -> bool:
        return self.next is not None and now >= self.next.t

    def grace_expired(self, now: float) -> bool:
        return self.next is not None and now >= self.next.t + self.grace

    def rail_recovery_target(self, now: float, hub_s: float, hub_dir: int,
                             from_xy: np.ndarray, v: float):
        """약속 놓친 로봇: 예측 가능한 허브 궤적을 따라 요격점 계산."""
        return intercept_hub(self.env.route, hub_s, self.env.hub_v, hub_dir,
                             from_xy, v, self.env.lap_len)


def convoke_intercept(env, hub_s: float, hub_dir: int, finder_xy: np.ndarray,
                      v: float):
    """조기 소집: 발견 로봇이 허브를 길목에서 만나는 (τ, s*) — 실패 시 None."""
    return intercept_hub(env.route, hub_s, env.hub_v, hub_dir, finder_xy, v,
                         env.lap_len)
