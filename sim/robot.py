"""로봇 상태와 운동 — 점 로봇·등속 (1스텝=1초)."""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Robot:
    rid: int
    role: str                 # 'hub' | 'left' | 'right'
    xy: np.ndarray
    v: float
    # 순찰 진행 상태
    hub_s: float = 0.0        # hub 전용: 노선 호길이
    hub_dir: int = 1
    sweep_d: float = 0.0      # flanker 전용: 스윕 경로 누적거리
    sweep_dir: int = 1
    # 임무·랑데뷰 상태
    mode: str = "patrol"      # patrol | detour | wait_site | working | at_rdv
    work_tid: int = -1        # working 중인 임무 id
    target: np.ndarray | None = None
    target_cid: int = -1      # detour 대상 후보 id (-1 = 랑데뷰/기타)
    queue: list = field(default_factory=list)   # 배정받은 (cid, xy) 목록

    def move_towards(self, dest: np.ndarray, dt: float = 1.0) -> bool:
        """dest로 등속 이동. 도착(1스텝 내) 시 True."""
        d = np.asarray(dest, float) - self.xy
        dist = float(np.linalg.norm(d))
        step = self.v * dt
        if dist <= step:
            self.xy = np.asarray(dest, float).copy()
            return True
        self.xy = self.xy + d / dist * step
        return False


def advance_hub(r: Robot, env, dt: float = 1.0):
    from .geometry import hub_pos_at
    r.hub_s = hub_pos_at(r.hub_s, env.hub_v, r.hub_dir, dt, env.route, env.lap_len)
    # 왕복형 방향 추적 (반사 감지)
    if not env.route.loop:
        nxt = r.hub_s + r.hub_dir * env.hub_v * 0.5
        if nxt <= 0 or nxt >= env.route.length:
            r.hub_dir *= -1
    r.xy = env.route.point_at(r.hub_s)


def advance_flanker(r: Robot, env, world_path: np.ndarray, cum: np.ndarray,
                    dt: float = 1.0):
    total = cum[-1]
    r.sweep_d += r.sweep_dir * r.v * dt
    if r.sweep_d >= total:
        r.sweep_d = total; r.sweep_dir = -1
    elif r.sweep_d <= 0:
        r.sweep_d = 0; r.sweep_dir = 1
    i = min(int(np.searchsorted(cum, r.sweep_d, "right") - 1), len(cum) - 2)
    seg = world_path[i + 1] - world_path[i]
    seglen = float(np.linalg.norm(seg))
    frac = (r.sweep_d - cum[i]) / max(seglen, 1e-9)
    r.xy = world_path[i] + seg * frac


def nearest_sweep_d(world_path: np.ndarray, cum: np.ndarray, xy: np.ndarray) -> float:
    """복귀 시: 현 위치에서 가장 가까운 스윕 경로상 진행거리."""
    d2 = np.linalg.norm(world_path - xy, axis=1)
    return float(cum[int(np.argmin(d2))])
