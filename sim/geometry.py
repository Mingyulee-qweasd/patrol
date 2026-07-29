"""기하 유틸 — 폴리라인 좌표계 (s=호길이, t=횡방향 오프셋)와 스윕 경로 생성.

환경 무관 원칙: 모든 함수는 환경 데이터를 인자로 받는다. 환경 이름·특성 하드코딩 금지.
"""
import numpy as np


class Polyline:
    def __init__(self, points: np.ndarray, loop: bool = False):
        self.pts = np.asarray(points, dtype=float)
        self.loop = loop
        seg = np.diff(self.pts, axis=0)
        self.seg_len = np.linalg.norm(seg, axis=1)
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.length = float(self.cum[-1])
        self._tangents = seg / self.seg_len[:, None]

    def _locate(self, s: float) -> tuple[int, float]:
        s = s % self.length if self.loop else np.clip(s, 0.0, self.length)
        i = int(np.searchsorted(self.cum, s, side="right") - 1)
        i = min(i, len(self.seg_len) - 1)
        return i, s - self.cum[i]

    def point_at(self, s: float) -> np.ndarray:
        i, ds = self._locate(s)
        return self.pts[i] + self._tangents[i] * ds

    def normal_at(self, s: float) -> np.ndarray:
        i, _ = self._locate(s)
        tx, ty = self._tangents[i]
        return np.array([-ty, tx])  # 좌측 = +t

    def frame_to_world(self, s: float, t: float) -> np.ndarray:
        return self.point_at(s) + self.normal_at(s) * t

    def project(self, xy: np.ndarray) -> float:
        """월드 점 → 최근접 호길이 s (조밀 샘플 근사; v1 충분)."""
        ss = np.linspace(0, self.length, max(200, int(self.length / 2)))
        pts = np.array([self.point_at(s) for s in ss])
        return float(ss[np.argmin(np.linalg.norm(pts - xy, axis=1))])


def sweep_path(route: Polyline, side: str, width: float, spacing: float) -> np.ndarray:
    """회랑 톱니(경운기) 스윕 웨이포인트 [(s,t)...] — 노선 방향 전진 + 횡 왕복.

    v1: 구간 균일 폭. park(다각형 회랑)은 폭 함수를 s별로 평가하는 확장으로 처리.
    """
    sign = +1 if side == "left" else -1
    ss = np.arange(0.0, route.length + 1e-6, spacing)
    wp = []
    for k, s in enumerate(ss):
        t_near, t_far = sign * 2.0, sign * width  # 도로변 2m부터 바깥 경계까지
        pair = [(s, t_near), (s, t_far)] if k % 2 == 0 else [(s, t_far), (s, t_near)]
        wp.extend(pair)
    return np.array(wp)  # (s, t) 좌표


def path_length_world(route: Polyline, st_waypoints: np.ndarray) -> float:
    pts = np.array([route.frame_to_world(s, t) for s, t in st_waypoints])
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def hub_speed_sync(route: Polyline, sweep_len_max: float, v_sweep: float) -> float:
    """허브 속도 동기화: 측면의 '노선 방향 전진 속도'에 맞춤 — 셋이 나란히 전진.

    측면이 스윕 1패스(노선 0→L)를 마치는 시간에 허브도 노선 길이 L을 진행.
    (왕복형이면 끝에서 셋이 함께 반환 — 반환 지점이 자연 랑데뷰가 됨)
    """
    t_sweep_pass = sweep_len_max / v_sweep
    return route.length / t_sweep_pass


def travel_time(a: np.ndarray, b: np.ndarray, v: float) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / v)


def intercept_hub(route: Polyline, hub_s: float, hub_v: float, hub_dir: int,
                  from_xy: np.ndarray, v: float, lap: float,
                  max_horizon_s: float = 3600.0) -> tuple[float, float] | None:
    """허브 요격점: 시각 τ에 허브가 있을 s*(τ)에 나도 τ에 도달 가능한 최소 τ.

    허브 궤적이 결정적(레일)이라 1차원 탐색으로 충분. 반환 (τ, s*) 또는 None.
    """
    for tau in np.arange(0.0, max_horizon_s, 5.0):
        s_h = hub_pos_at(hub_s, hub_v, hub_dir, tau, route, lap)
        p_h = route.point_at(s_h)
        if travel_time(from_xy, p_h, v) <= tau:
            return float(tau), float(s_h)
    return None


def hub_pos_at(s0: float, v: float, direction: int, dt: float,
               route: Polyline, lap: float) -> float:
    """왕복형/루프형 허브의 dt초 후 호길이 위치."""
    if route.loop:
        return (s0 + direction * v * dt) % route.length
    # 왕복: 반사 경계
    x = s0 + direction * v * dt
    period = 2 * route.length
    x = x % period
    return x if x <= route.length else period - x
