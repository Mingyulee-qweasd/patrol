"""환경 로더 — YAML이 환경의 전부, 코드는 환경 불변."""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .geometry import Polyline, sweep_path, path_length_world, hub_speed_sync


@dataclass
class Env:
    name: str
    route: Polyline
    corridor_width: dict          # {'left': w, 'right': w}
    v_sweep: float
    sense_r: float
    reliable_r: float
    cell_m: float
    sweep_spacing: float
    meet_eps: float
    grace_frac: float
    task_cfg: dict
    sweep_paths: dict = field(default_factory=dict)   # side -> (s,t) waypoints
    hub_v: float = 0.0
    lap_len: float = 0.0
    collect: dict = field(default_factory=dict)       # v2 이종 수거 파라미터 (없으면 v1 동작)

    @property
    def lap_time(self) -> float:
        return self.lap_len / self.hub_v


def load_env(path: str | Path) -> Env:
    cfg = yaml.safe_load(Path(path).read_text())
    route = Polyline(np.array(cfg["route"]["points"], float), cfg["route"].get("loop", False))
    e = Env(
        name=cfg["name"],
        route=route,
        corridor_width={s: c["width"] for s, c in cfg["corridors"].items()},
        v_sweep=cfg["robots"]["speed_mps"],
        sense_r=cfg["robots"]["sense_radius_m"],
        reliable_r=cfg["robots"]["reliable_radius_m"],
        cell_m=cfg["patrol"]["cell_m"],
        sweep_spacing=cfg["patrol"]["sweep_spacing_m"],
        meet_eps=cfg["rendezvous"]["meet_eps_s"],
        grace_frac=cfg["rendezvous"]["grace_frac"],
        task_cfg=cfg["tasks"],
        collect=cfg.get("collect", {}),
    )
    lens = {}
    for side, w in e.corridor_width.items():
        wp = sweep_path(route, side, w, e.sweep_spacing)
        e.sweep_paths[side] = wp
        lens[side] = path_length_world(route, wp)
    e.hub_v = hub_speed_sync(route, max(lens.values()), e.v_sweep)
    e.lap_len = route.length * (1 if route.loop else 2)
    return e
