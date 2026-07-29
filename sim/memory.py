"""task 후보 메모리 — 확신 로그오즈 누적, n̂ 2단 갱신, 랑데뷰 병합.

설계 §2: 항목 = (위치, 타입, 확신 s, n̂, û, 목격 시각들). sgc 무관 — 교과서 베이즈 한 줄.
"""
from dataclasses import dataclass, field

import numpy as np

MATCH_RADIUS = 8.0  # 같은 대상 판정 반경 [m]


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return float(np.log(p / (1 - p)))


@dataclass
class Candidate:
    cid: int
    xy: np.ndarray
    s_logodds: float          # task 여부 확신 (로그오즈 누적)
    n_hat: int
    u_hat: int
    first_seen: float
    last_seen: float
    near_confirmed: bool = False   # 근거리 재판정 거침
    committed: bool = False        # 즉시 실행/배정으로 소비됨
    agenda: bool = False           # 안건 큐 등재
    reobserve: bool = False        # 재관측 예약
    convoked: bool = False         # 조기 소집 1회 발화 플래그
    reobs_visits: int = 0          # 재관측 전용 방문 횟수 (상한 2)
    gt_tid: int = -1               # 채점용 (정책은 안 봄)

    @property
    def s(self) -> float:
        return 1.0 / (1.0 + np.exp(-self.s_logodds))


class RobotMemory:
    def __init__(self, robot_id: int):
        self.rid = robot_id
        self.items: list[Candidate] = []
        self._next = 0

    def update(self, xy: np.ndarray, judgment: dict, band: str, now: float,
               gt_tid: int) -> Candidate:
        """관측 판정 1건 반영. 같은 자리(반경 내) 항목이 있으면 누적, 없으면 신설."""
        # calibrated conf → 로그오즈 증분. task 판정은 +, 비task 판정은 −.
        conf01 = judgment["conf"] / 100.0
        delta = _logit(conf01) * (1.0 if judgment["is_task"] else -1.0)
        if band == "far":
            delta *= 0.5  # 원거리 가중 하향 (관측 품질)

        for c in self.items:
            if np.linalg.norm(c.xy - xy) < MATCH_RADIUS:
                c.s_logodds = float(np.clip(c.s_logodds + delta, -30, 30))
                c.last_seen = now
                if band == "near":
                    c.near_confirmed = True
                    # n̂ 2단: 근거리 재판정이 원거리 추정을 대체
                    if judgment["is_task"]:
                        c.n_hat, c.u_hat = judgment["n_hat"], judgment["u_hat"]
                return c
        c = Candidate(self._next, xy.copy(), delta,
                      judgment["n_hat"], judgment["u_hat"], now, now,
                      near_confirmed=(band == "near"), gt_tid=gt_tid)
        self._next += 1
        self.items.append(c)
        return c

    def merge_from(self, other: "RobotMemory"):
        """랑데뷰 병합 — 독립 목격의 확신 합산 (교차 확인 효과)."""
        for oc in other.items:
            matched = False
            for c in self.items:
                if np.linalg.norm(c.xy - oc.xy) < MATCH_RADIUS:
                    if oc.cid != c.cid or other.rid != self.rid:
                        c.s_logodds += oc.s_logodds * 0.5  # 이중 계상 완화 절충
                        c.near_confirmed |= oc.near_confirmed
                        c.committed |= oc.committed
                        c.last_seen = max(c.last_seen, oc.last_seen)
                    matched = True
                    break
            if not matched:
                self.items.append(Candidate(self._next, oc.xy.copy(), oc.s_logodds,
                                            oc.n_hat, oc.u_hat, oc.first_seen,
                                            oc.last_seen, oc.near_confirmed,
                                            oc.committed, oc.agenda, oc.reobserve,
                                            oc.gt_tid))
                self._next += 1
