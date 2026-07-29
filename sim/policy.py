"""판단 1~5 — 설계 문서 §3 확정 수식의 직역. 재해석 금지.

전부 순수 함수(상태 입력 → 결정 출력) — runner가 호출. VLM은 여기 없음 (측정은 이미 끝난 값).
"""
from dataclasses import dataclass

import numpy as np

from .memory import Candidate


@dataclass
class Params:
    theta_act: float = 0.80       # 판단1 확신 문턱 (보정 확률 기준)
    w_idle: float = 1.0           # J와 공유하는 커버리지 가중
    u_high: int = 3               # 조기 소집 긴급 문턱
    meet_eps: float = 30.0        # 협의 소요
    assign_travel_est: float = 120.0  # Δ의 "배정 로봇 이동" 추정 상수 (v1)
    convoke_cost_s: float = 180.0     # 조기 소집 비용 (요격 우회 상수 추정)


# ── 판단1: 확신 문턱 ──────────────────────────────────────────────
def gate(c: Candidate, p: Params) -> str:
    """'act' = 개입 가능 / 'reobserve' = 재관측 예약 / 'drop' = 비task 확정."""
    if c.s >= p.theta_act and c.near_confirmed:
        return "act"
    if c.s <= 0.10:
        return "drop"
    return "reobserve"


# ── 판단2: 임무 배정 분기 (c=0 확정식) ────────────────────────────
def dispatch(c: Candidate, my_xy: np.ndarray, v: float,
             t_to_rdv: float, can_return: bool, p: Params) -> str:
    """'now'(즉시 실행) / 'agenda'(안건화). n̂≥2는 항상 agenda (조기 소집은 별도)."""
    if c.n_hat >= 2:
        return "agenda"
    my_oneway = float(np.linalg.norm(c.xy - my_xy)) / v
    delta = (t_to_rdv + p.meet_eps + p.assign_travel_est) - my_oneway  # 미룸의 추가 지연
    gap_cost = p.w_idle * (2 * my_oneway)                              # 순찰 공백(왕복)
    if c.s * c.u_hat * max(delta, 0.0) > gap_cost and can_return:
        return "now"
    return "agenda"


# ── 조기 소집 (경제성 발화) ───────────────────────────────────────
def early_convoke(c: Candidate, t_to_rdv: float, p: Params) -> bool:
    """u 높은 다수-로봇 안건: u·(절약 시간) > 소집 비용이면 발화."""
    if c.n_hat < 2 or c.u_hat < p.u_high:
        return False
    saved = t_to_rdv  # 정기 랑데뷰 대기를 통째로 절약한다고 근사
    return c.u_hat * saved > p.convoke_cost_s * c.n_hat


# ── 판단4: 랑데뷰 간격 조정 ───────────────────────────────────────
def next_interval(agenda: list[Candidate], d0: float, theta_v: float = 4.0) -> float:
    V = sum(c.u_hat for c in agenda)
    if V > theta_v:
        return 0.5 * d0
    if V == 0:
        return min(2.0 * d0, 1.5 * d0)  # 안건 없으면 연장 (상한 내)
    return d0


# ── 판단5: 순차 경매 배정 ─────────────────────────────────────────
def auction(agenda: list[Candidate], robot_xys: dict, v: float,
            p: Params) -> dict:
    """긴급 안건부터, 안건마다 최저 입찰 n̂개 로봇 낙찰.

    bid = 이동시간 + w_idle·(왕복 이동 = 커버리지 공백 근사).
    반환: {cid: [robot_id, ...]}. 로봇은 여러 안건에 순차 배정될 수 있으며
    이동 누적을 반영해 다음 안건 입찰 위치가 갱신된다 (경로 연쇄).
    """
    pos = {r: np.asarray(xy, float).copy() for r, xy in robot_xys.items()}
    out = {}
    for c in sorted(agenda, key=lambda x: -x.u_hat):
        bids = []
        for r, xy in pos.items():
            travel = float(np.linalg.norm(c.xy - xy)) / v
            bids.append((travel + p.w_idle * 2 * travel, r))
        bids.sort()
        winners = [r for _, r in bids[:max(1, c.n_hat)]]
        out[c.cid] = winners
        for r in winners:
            pos[r] = c.xy.copy()  # 낙찰자는 그 지점에서 다음 입찰
    return out
