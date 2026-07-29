"""동작 확인 동영상 — 실제 에피소드의 장면 기록을 렌더링.

로봇 3대(빨강 허브·파랑 왼쪽·초록 오른쪽), 임무(주황 ●=처리 대상, 회색 ×=애매,
보라 ◆=위험물), 별=다음 만남 지점. 하단에 최근 사건 자막.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
_f = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"
font_manager.fontManager.addfont(_f)
matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=_f).get_name()
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from sim.runner import Episode

KIND_KO = {"trash": "쓰레기", "sharps": "위험쓰레기", "carcass_s": "소형사체",
           "carcass_l": "대형사체", "bulky": "대형폐기물", "obstacle": "장애물",
           "ambiguous": "애매", "hazard": "위험물"}
EVENT_KO = {"found": "발견", "dispatch_now": "즉시 처리 출발", "agenda": "안건 기억",
            "bounce": "인원 부족—되돌아감", "assign": "만남서 배정", "convoke": "긴급 소집",
            "complete": "완료", "misintervention": "오개입!", "wait_site": "현장서 동료 대기",
            "rendezvous": "랑데뷰 개최", "abort_nontask": "가보니 임무 아님"}


def render(cfg="exp/configs/simple.yaml", seed=3, horizon=7200, out="out_run.gif"):
    ep = Episode(cfg, seed=seed, horizon_s=horizon, warmup_s=600)
    trace = ep.run()
    snaps = ep.snaps
    events = trace.events
    print(f"장면 {len(snaps)}개, 사건 {len(events)}건 — 렌더링 시작", flush=True)

    fig, ax = plt.subplots(figsize=(14, 5))
    rp = np.array([ep.env.route.point_at(s) for s in np.linspace(0, ep.env.route.length, 120)])

    def draw(k):
        ax.clear()
        snap = snaps[k]
        t = snap["t"]
        ax.plot(rp[:, 0], rp[:, 1], "k-", lw=1.5, alpha=0.6)
        # 임무들
        for x, y, kind, cls in snap["tasks"]:
            if cls == "task":
                ax.plot(x, y, "o", color="orange", ms=7)
            elif cls == "hazard":
                ax.plot(x, y, "D", color="purple", ms=7)
            else:
                ax.plot(x, y, "x", color="gray", ms=6)
        # 다음 만남 지점
        mt, mxy = snap["meet"]
        ax.plot(*mxy, "*", color="black", ms=16, mfc="yellow")
        # 로봇
        for rid, (x, y, mode) in snap["robots"].items():
            style = {0: ("s", "red"), 1: ("^", "blue"), 2: ("^", "green")}[rid]
            ax.plot(x, y, style[0], color=style[1], ms=11)
            ax.annotate(mode, (x, y), textcoords="offset points", xytext=(0, 10),
                        fontsize=7, ha="center")
        # 최근 사건 자막 (직전 60초)
        recent = [e for e in events if t - 60 <= e["t"] <= t][-4:]
        lines = [f"t={e['t']:.0f}s  {EVENT_KO.get(e['e'], e['e'])}"
                 + (f" [{KIND_KO.get(e.get('tkind', ''), e.get('tkind', ''))}]" if e.get("tkind") else "")
                 for e in recent]
        ax.set_title(f"t = {t/60:.1f}분   |   " + ("  ·  ".join(lines) if lines else "순찰 중"),
                     fontsize=10)
        ax.set_xlim(-30, ep.env.route.length + 30)
        ax.set_ylim(-55, 55)
        ax.set_aspect("equal")

    anim = FuncAnimation(fig, draw, frames=len(snaps))
    anim.save(out, writer=PillowWriter(fps=8))
    print(f"저장: {out} ({len(snaps)} 프레임 = 시뮬 {horizon/3600:.0f}시간)", flush=True)
    # 키프레임 정지 이미지 (자체 검증용)
    for name, idx in [("key_early", len(snaps)//6), ("key_mid", len(snaps)//2),
                      ("key_late", 5*len(snaps)//6)]:
        draw(idx)
        fig.savefig(f"{Path(out).stem}_{name}.png", dpi=80)
    print("키프레임 저장", flush=True)


if __name__ == "__main__":
    render()
