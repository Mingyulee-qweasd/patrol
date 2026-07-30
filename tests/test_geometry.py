"""tests/test_geometry.py — sim/geometry.py 검증.

담당 범위:
1) Polyline 호길이: 알려진 좌표의 length가 수계산과 일치
2) 왕복(반사) 좌표계: s가 길이를 넘어갈 때 반사 위치 수계산 대조 (hub_pos_at)
3) hub_speed_sync: 노선 길이 / 스윕 1패스 시간 수계산 대조
4) intercept_hub: 정지 허브(속도 0 경계)·등속 허브 요격점의 기하 타당성
   (요격 시각에 허브 실위치와 요격점 오차 < 5 m, 로봇이 τ 안에 도달 가능)
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.geometry import (  # noqa: E402
    Polyline,
    hub_pos_at,
    hub_speed_sync,
    intercept_hub,
    travel_time,
)


def straight_route(length: float = 1000.0, loop: bool = False) -> Polyline:
    return Polyline(np.array([[0.0, 0.0], [length, 0.0]]), loop=loop)


# ---------------------------------------------------------------- 1) 호길이

class TestPolylineLength:
    def test_straight_1km(self):
        r = straight_route(1000.0)
        assert r.length == pytest.approx(1000.0)

    def test_two_segments_L_shape(self):
        # (0,0)->(300,0)->(300,400): 300 + 400 = 700 m
        r = Polyline(np.array([[0.0, 0.0], [300.0, 0.0], [300.0, 400.0]]))
        assert r.length == pytest.approx(700.0)

    def test_diagonal_3_4_5(self):
        # 빗변 sqrt(300^2 + 400^2) = 500 m
        r = Polyline(np.array([[0.0, 0.0], [300.0, 400.0]]))
        assert r.length == pytest.approx(500.0)

    def test_point_at_known_s(self):
        r = Polyline(np.array([[0.0, 0.0], [300.0, 0.0], [300.0, 400.0]]))
        np.testing.assert_allclose(r.point_at(150.0), [150.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(r.point_at(300.0), [300.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(r.point_at(500.0), [300.0, 200.0], atol=1e-9)

    def test_point_at_clips_when_not_loop(self):
        # 비루프 노선: s가 범위를 벗어나면 끝점으로 클립 (반사는 hub_pos_at 담당)
        r = straight_route(1000.0)
        np.testing.assert_allclose(r.point_at(1500.0), [1000.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(r.point_at(-10.0), [0.0, 0.0], atol=1e-9)


# ------------------------------------------------- 2) 왕복(반사) 좌표계

class TestReflection:
    def test_forward_past_end(self):
        r = straight_route(1000.0)
        # s0=0, +방향 10 m/s, 120 s → 1200 m 진행: 1000 도달 후 200 반사 → 800
        s = hub_pos_at(0.0, 10.0, +1, 120.0, r, lap=2 * r.length)
        assert s == pytest.approx(800.0)

    def test_backward_past_zero(self):
        r = straight_route(1000.0)
        # s0=100, -방향 10 m/s, 30 s → 300 m 후진: 0에서 반사 → 200
        s = hub_pos_at(100.0, 10.0, -1, 30.0, r, lap=2 * r.length)
        assert s == pytest.approx(200.0)

    def test_multiple_periods(self):
        r = straight_route(100.0)
        # 0→100(10s)→0(20s)→100(30s), t=35 s: 후진 50 m 지점 → 50
        s = hub_pos_at(0.0, 10.0, +1, 35.0, r, lap=2 * r.length)
        assert s == pytest.approx(50.0)

    def test_exact_endpoint(self):
        r = straight_route(1000.0)
        # 정확히 끝점 도달 순간 (경계값): s = L
        s = hub_pos_at(0.0, 10.0, +1, 100.0, r, lap=2 * r.length)
        assert s == pytest.approx(1000.0)
        assert 0.0 <= s <= r.length

    def test_loop_wraps_instead_of_reflecting(self):
        sq = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0],
                       [0.0, 100.0], [0.0, 0.0]])
        r = Polyline(sq, loop=True)
        assert r.length == pytest.approx(400.0)
        # 350 + 100 = 450 → mod 400 = 50 (루프는 반사가 아니라 랩)
        s = hub_pos_at(350.0, 10.0, +1, 10.0, r, lap=r.length)
        assert s == pytest.approx(50.0)


# ------------------------------------------------------ 3) hub_speed_sync

class TestHubSpeedSync:
    def test_hand_calc(self):
        r = straight_route(1000.0)
        # 스윕 1패스 = 5000 m / 1 m/s = 5000 s → 허브 = 1000 / 5000 = 0.2 m/s
        assert hub_speed_sync(r, 5000.0, 1.0) == pytest.approx(0.2)

    def test_hand_calc_other_values(self):
        r = straight_route(500.0)
        # 스윕 1패스 = 1250 m / 2.5 m/s = 500 s → 허브 = 500 / 500 = 1.0 m/s
        assert hub_speed_sync(r, 1250.0, 2.5) == pytest.approx(1.0)


# -------------------------------------------------------- 4) intercept_hub

def _assert_geometrically_valid(route, hub_s, hub_v, hub_dir, from_xy, v, lap, res):
    """요격 시각 τ에 (a) 허브 실위치와 요격점 오차 < 5 m, (b) 로봇 도달 가능."""
    assert res is not None
    tau, s_star = res
    s_hub = hub_pos_at(hub_s, hub_v, hub_dir, tau, route, lap)
    p_hub = route.point_at(s_hub)
    p_star = route.point_at(s_star)
    assert np.linalg.norm(p_hub - p_star) < 5.0
    assert travel_time(from_xy, p_star, v) <= tau + 1e-9
    return tau, s_star


class TestInterceptHub:
    def test_stationary_hub(self):
        r = straight_route(1000.0)
        lap = 2 * r.length
        from_xy = np.array([500.0, 300.0])
        res = intercept_hub(r, hub_s=400.0, hub_v=0.0, hub_dir=+1,
                            from_xy=from_xy, v=2.0, lap=lap)
        tau, s_star = _assert_geometrically_valid(
            r, 400.0, 0.0, +1, from_xy, 2.0, lap, res)
        # 정지 허브 → 요격점은 허브 위치 그대로
        assert s_star == pytest.approx(400.0)
        # 수계산: dist = sqrt(100^2 + 300^2) = 316.23 m, /2 = 158.1 s
        # → 5 s 그리드에서 최소 가능 τ = 160
        assert tau == pytest.approx(160.0)

    def test_stationary_hub_colocated_zero_tau(self):
        # 속도 0 + 거리 0 경계: 즉시 요격 (τ=0)
        r = straight_route(1000.0)
        lap = 2 * r.length
        res = intercept_hub(r, hub_s=400.0, hub_v=0.0, hub_dir=+1,
                            from_xy=r.point_at(400.0), v=1.0, lap=lap)
        tau, s_star = _assert_geometrically_valid(
            r, 400.0, 0.0, +1, r.point_at(400.0), 1.0, lap, res)
        assert tau == pytest.approx(0.0)
        assert s_star == pytest.approx(400.0)

    def test_moving_hub(self):
        r = straight_route(1000.0)
        lap = 2 * r.length
        from_xy = np.array([500.0, 100.0])
        res = intercept_hub(r, hub_s=0.0, hub_v=2.0, hub_dir=+1,
                            from_xy=from_xy, v=3.0, lap=lap)
        tau, s_star = _assert_geometrically_valid(
            r, 0.0, 2.0, +1, from_xy, 3.0, lap, res)
        # 수계산: 허브 s=2τ, dist = sqrt((500-2τ)^2 + 100^2), 조건 dist/3 <= τ
        #   τ=100: sqrt(300^2+100^2)/3 = 105.4 > 100 (불가)
        #   τ=105: sqrt(290^2+100^2)/3 = 102.3 <= 105 (가능) → τ=105, s*=210
        assert tau == pytest.approx(105.0)
        assert s_star == pytest.approx(210.0)
        # 직전 그리드 점은 실제로 도달 불가 (그리드 최소성)
        prev = tau - 5.0
        s_prev = hub_pos_at(0.0, 2.0, +1, prev, r, lap)
        assert travel_time(from_xy, r.point_at(s_prev), 3.0) > prev

    def test_moving_hub_after_reflection(self):
        # 허브가 노선 끝에서 반사한 뒤에 요격되는 케이스
        r = straight_route(1000.0)
        lap = 2 * r.length
        from_xy = np.array([800.0, 50.0])
        res = intercept_hub(r, hub_s=950.0, hub_v=2.0, hub_dir=+1,
                            from_xy=from_xy, v=1.5, lap=lap)
        tau, s_star = _assert_geometrically_valid(
            r, 950.0, 2.0, +1, from_xy, 1.5, lap, res)
        # 수계산: 반사(τ=25 s, s=1000) 후 허브 s = 1050 - 2τ
        #   τ=70: 허브 910, sqrt(110^2+50^2)/1.5 = 80.5 > 70 (불가)
        #   τ=75: 허브 900, sqrt(100^2+50^2)/1.5 = 74.5 <= 75 (가능)
        assert tau == pytest.approx(75.0)
        assert s_star == pytest.approx(900.0)

    def test_unreachable_returns_none(self):
        r = straight_route(1000.0)
        res = intercept_hub(r, hub_s=0.0, hub_v=0.0, hub_dir=+1,
                            from_xy=np.array([100000.0, 0.0]), v=1.0,
                            lap=2 * r.length, max_horizon_s=60.0)
        assert res is None
