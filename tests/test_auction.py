"""판단5 순차 경매(sim.policy.auction) 검증.

bid = 이동시간 + w_idle·(왕복 이동) = travel·(1 + 2·w_idle).
긴급(u_hat) 내림차순으로 안건을 처리하고, 안건마다 최저 bid n̂개(최소 1개)
로봇을 낙찰시키며, 낙찰자는 그 지점으로 위치가 갱신돼 다음 입찰에 반영된다.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.memory import Candidate
from sim.policy import Params, auction


def mk(cid, xy, u_hat, n_hat):
    """테스트용 최소 Candidate."""
    return Candidate(cid=cid, xy=np.asarray(xy, float), s_logodds=5.0,
                     n_hat=n_hat, u_hat=u_hat, first_seen=0.0, last_seen=0.0)


P = Params()  # w_idle=1.0 → bid = 3·(거리/v)


# ── 1) 소규모 손계산 대조 ─────────────────────────────────────────
def test_hand_computed_two_robots_two_items():
    """r0@(0,0), r1@(100,0); 긴급 cB@(90,0)→r1, 이후 cA@(10,0)→r0.

    손계산(v=1): cB 먼저(u=5) — bid r0=3·90=270, r1=3·10=30 → r1 낙찰,
    r1은 (90,0)으로 연쇄. cA(u=2) — bid r0=3·10=30, r1=3·80=240 → r0 낙찰.
    """
    agenda = [mk(1, (10, 0), u_hat=2, n_hat=1), mk(2, (90, 0), u_hat=5, n_hat=1)]
    out = auction(agenda, {0: (0.0, 0.0), 1: (100.0, 0.0)}, v=1.0, p=P)
    assert out == {2: [1], 1: [0]}


def test_winner_invariant_to_speed_scale():
    """v는 모든 bid를 같은 비율로 나누므로 낙찰자는 불변."""
    agenda = [mk(1, (10, 0), u_hat=2, n_hat=1), mk(2, (90, 0), u_hat=5, n_hat=1)]
    out = auction(agenda, {0: (0.0, 0.0), 1: (100.0, 0.0)}, v=2.5, p=P)
    assert out == {2: [1], 1: [0]}


def test_urgent_item_processed_first():
    """긴급 안건이 리스트 순서와 무관하게 먼저 배정된다.

    r0@(0,0), r1@(60,0); cA@(25,0) u=1, cB@(35,0) u=9 — 리스트에는 cA가 먼저.
    긴급 우선이면 cB: r1(bid 75) < r0(105) → r1 낙찰·(35,0) 연쇄,
    cA: r1(bid 30) < r0(75) → 둘 다 r1. (순서대로 처리했다면 둘 다 r0가 됨.)
    """
    agenda = [mk(0, (25, 0), u_hat=1, n_hat=1), mk(1, (35, 0), u_hat=9, n_hat=1)]
    out = auction(agenda, {0: (0.0, 0.0), 1: (60.0, 0.0)}, v=1.0, p=P)
    assert out == {1: [1], 0: [1]}


def test_every_agenda_item_gets_entry():
    agenda = [mk(7, (10, 0), u_hat=2, n_hat=1), mk(8, (90, 0), u_hat=5, n_hat=1)]
    out = auction(agenda, {0: (0.0, 0.0), 1: (100.0, 0.0)}, v=1.0, p=P)
    assert set(out) == {7, 8}


# ── 2) 경로 연쇄 ──────────────────────────────────────────────────
def test_chained_position_used_for_next_bid():
    """낙찰자의 다음 입찰 위치 = 직전 낙찰 지점.

    r0@(0,0), r1@(100,0); c1@(60,0) u=5 → r1 낙찰(40 < 60), (60,0)으로 이동.
    c2@(35,0) u=3 — 연쇄 반영 시 r1 거리 25 < r0 35 → r1 낙찰.
    연쇄가 없다면 r0(35) < r1 원위치(65)로 r0가 낙찰됐을 사례.
    """
    agenda = [mk(10, (60, 0), u_hat=5, n_hat=1), mk(11, (35, 0), u_hat=3, n_hat=1)]
    out = auction(agenda, {0: (0.0, 0.0), 1: (100.0, 0.0)}, v=1.0, p=P)
    assert out == {10: [1], 11: [1]}


def test_loser_position_not_chained():
    """낙찰 못 한 로봇은 원위치에서 다음 입찰.

    r0@(0,0), r1@(100,0); c1@(60,0) u=5 → r1. c2@(5,0) u=3 —
    r0는 원위치 그대로(거리 5)라 r1(연쇄 후 55)을 이긴다.
    """
    agenda = [mk(10, (60, 0), u_hat=5, n_hat=1), mk(11, (5, 0), u_hat=3, n_hat=1)]
    out = auction(agenda, {0: (0.0, 0.0), 1: (100.0, 0.0)}, v=1.0, p=P)
    assert out == {10: [1], 11: [0]}


def test_multi_winner_all_chained():
    """n̂=2 낙찰자 2대 모두 지점으로 연쇄 — 이후 안건에서 둘 다 그 지점 거리로 입찰.

    r0@(0,0), r1@(12,0), r2@(100,0); c1@(5,0) n̂=2 → r0(5), r1(7) 낙찰.
    c2@(5,30) n̂=1 — r0·r1 모두 (5,0)에서 거리 30 동률, r2는 99.6 →
    동률은 낮은 id 우선으로 r0.
    """
    agenda = [mk(0, (5, 0), u_hat=5, n_hat=2), mk(1, (5, 30), u_hat=1, n_hat=1)]
    out = auction(agenda, {0: (0.0, 0.0), 1: (12.0, 0.0), 2: (100.0, 0.0)},
                  v=1.0, p=P)
    assert set(out[0]) == {0, 1}
    assert out[1] == [0]


# ── 3) n̂개 낙찰 / 가용 부족 ──────────────────────────────────────
def test_nhat2_assigns_two_robots():
    agenda = [mk(0, (5, 0), u_hat=3, n_hat=2)]
    out = auction(agenda, {0: (0.0, 0.0), 1: (12.0, 0.0), 2: (100.0, 0.0)},
                  v=1.0, p=P)
    assert len(out[0]) == 2
    assert set(out[0]) == {0, 1}          # 최저 bid 2대 (5, 7); r2(95)는 제외
    assert out[0] == [0, 1]               # bid 오름차순 정렬 순서


def test_insufficient_robots_assigns_all_available():
    """n̂=3인데 로봇 2대 → 정의된 동작: 가용 전원(2대)만 낙찰, 오류 없음."""
    agenda = [mk(0, (50, 50), u_hat=4, n_hat=3)]
    out = auction(agenda, {0: (0.0, 0.0), 1: (100.0, 0.0)}, v=1.0, p=P)
    assert len(out[0]) == 2
    assert set(out[0]) == {0, 1}


def test_no_robots_yields_empty_crew():
    """로봇 0대 — 예외 없이 빈 낙찰 목록 (정의된 동작; findings 참조)."""
    out = auction([mk(0, (1, 1), u_hat=2, n_hat=2)], {}, v=1.0, p=P)
    assert out == {0: []}


# ── 4) 경계 사례 ──────────────────────────────────────────────────
def test_tie_bid_breaks_by_lower_robot_id():
    """동률 bid: (bid, rid) 튜플 정렬로 낮은 id가 결정적으로 낙찰."""
    agenda = [mk(0, (0, 0), u_hat=2, n_hat=1)]
    out = auction(agenda, {0: (0.0, 10.0), 1: (0.0, -10.0)}, v=1.0, p=P)
    assert out == {0: [0]}


def test_tie_bid_independent_of_dict_insertion_order():
    agenda = [mk(0, (0, 0), u_hat=2, n_hat=1)]
    out = auction(agenda, {1: (0.0, -10.0), 0: (0.0, 10.0)}, v=1.0, p=P)
    assert out == {0: [0]}


def test_empty_agenda_returns_empty_dict():
    assert auction([], {0: (0.0, 0.0), 1: (50.0, 50.0)}, v=1.0, p=P) == {}


def test_nhat3_assigns_all_three_robots():
    agenda = [mk(0, (50, 0), u_hat=5, n_hat=3)]
    out = auction(agenda, {0: (0.0, 0.0), 1: (40.0, 0.0), 2: (90.0, 0.0)},
                  v=1.0, p=P)
    assert len(out[0]) == 3
    assert set(out[0]) == {0, 1, 2}
    assert out[0] == [1, 2, 0]            # 거리 10 < 40 < 50 → bid 오름차순


def test_nhat_zero_still_gets_one_winner():
    """max(1, n̂) 하한 — n̂=0이어도 1대는 낙찰."""
    agenda = [mk(0, (10, 0), u_hat=1, n_hat=0)]
    out = auction(agenda, {0: (0.0, 0.0), 1: (100.0, 0.0)}, v=1.0, p=P)
    assert out == {0: [0]}


def test_input_positions_not_mutated():
    """호출자 좌표(robot_xys)는 경매 중 변경되지 않는다 (내부 copy)."""
    xy0 = np.array([0.0, 0.0])
    xy1 = np.array([100.0, 0.0])
    auction([mk(0, (60, 0), u_hat=5, n_hat=2)], {0: xy0, 1: xy1}, v=1.0, p=P)
    assert xy0.tolist() == [0.0, 0.0] and xy1.tolist() == [100.0, 0.0]
