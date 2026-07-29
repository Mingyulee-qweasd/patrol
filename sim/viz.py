"""애니메이션 시각화 — 정책 버그는 눈으로 잡는다 (D1 조기 제작).

demo 모드: 정책 없이 세 로봇이 기하 경로를 따라 움직이는 것만 확인 (D1 검증용).
이후 runner의 상태 기록(trace)을 받아 그리는 방식으로 확장.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from sim.env import load_env
from sim.geometry import hub_pos_at, path_length_world


def demo(cfg_path: str, out_gif: str, sim_seconds: int = 2400, fps_step: int = 20):
    env = load_env(cfg_path)
    # 측면 로봇: 스윕 웨이포인트를 등속 주파 / 허브: 동기화 속도로 노선 왕복
    world_paths = {}
    for side, wp in env.sweep_paths.items():
        pts = np.array([env.route.frame_to_world(s, t) for s, t in wp])
        world_paths[side] = pts

    def flanker_pos(side, t_now):
        pts = world_paths[side]
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0], np.cumsum(seg)])
        total = cum[-1]
        d = (env.v_sweep * t_now) % (2 * total)      # 스윕 왕복
        d = d if d <= total else 2 * total - d
        i = min(int(np.searchsorted(cum, d, side="right") - 1), len(seg) - 1)
        r = (d - cum[i]) / max(seg[i], 1e-9)
        return pts[i] + (pts[i + 1] - pts[i]) * r

    times = np.arange(0, sim_seconds, fps_step)
    fig, ax = plt.subplots(figsize=(12, 4))
    rpts = np.array([env.route.point_at(s) for s in np.linspace(0, env.route.length, 100)])
    ax.plot(rpts[:, 0], rpts[:, 1], "k-", lw=2, label="route")
    for side, pts in world_paths.items():
        ax.plot(pts[:, 0], pts[:, 1], "--", lw=0.5, alpha=0.5, label=f"sweep {side}")
    dots = {name: ax.plot([], [], marker, ms=10, label=name)[0]
            for name, marker in [("hub", "rs"), ("left", "b^"), ("right", "g^")]}
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    ax.set_title(f"{env.name}: hub_v={env.hub_v:.2f} m/s (sync), lap={env.lap_time:.0f}s")

    def update(t_now):
        s_h = hub_pos_at(0.0, env.hub_v, +1, t_now, env.route, env.lap_len)
        dots["hub"].set_data(*[[v] for v in env.route.point_at(s_h)])
        for side in ("left", "right"):
            dots[side].set_data(*[[v] for v in flanker_pos(side, t_now)])
        return list(dots.values())

    anim = FuncAnimation(fig, update, frames=times, blit=True)
    anim.save(out_gif, writer=PillowWriter(fps=10))
    print(f"저장: {out_gif} ({len(times)} 프레임, 시뮬 {sim_seconds}s)")


if __name__ == "__main__":
    demo("exp/configs/simple.yaml", "out_demo.gif")
