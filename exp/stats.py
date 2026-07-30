"""통계 파이프라인 — 비교군별 요약(신뢰구간) + full 대비 짝 검정.

문헌 관행 (정독 §6): bootstrap 10k 신뢰구간 + 비모수 검정, 같은 씨앗끼리 짝 비교.
사용: python exp/stats.py exp/out/main.jsonl [--load 0.25] [--metric completion_rate]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as sps

KEY_METRICS = ["completion_rate", "delay_mean", "weighted_delay", "idleness_avg",
               "n_misintervention_minor", "n_misintervention_severe", "n_bounce", "n_abort"]


def boot_ci(x: np.ndarray, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--load", type=float, default=None, help="특정 부하만 (기본: 전부 개별)")
    ap.add_argument("--metric", default="completion_rate")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.results)]
    loads = sorted({r["load"] for r in rows}) if args.load is None else [args.load]

    for load in loads:
        sub = [r for r in rows if r["load"] == load]
        by_arm = defaultdict(dict)   # arm -> seed -> row
        for r in sub:
            by_arm[r["arm"]][r["seed"]] = r
        arms = sorted(by_arm, key=lambda a: (a != "full", a))
        n_seeds = len(by_arm.get("full", {}))
        print(f"\n===== 부하 {load} (씨앗 {n_seeds}개) =====")

        # ① 요약표
        print(f"{'비교군':16s} " + " ".join(f"{m[:14]:>15s}" for m in KEY_METRICS))
        for a in arms:
            vals = {m: np.array([r[m] for r in by_arm[a].values()], float) for m in KEY_METRICS}
            line = f"{a:16s} "
            for m in KEY_METRICS:
                lo, hi = boot_ci(vals[m])
                line += f" {vals[m].mean():6.2f}[{lo:5.2f},{hi:5.2f}]"[:16].rjust(16)
            print(line)

        # ② full 대비 짝 비교 (같은 씨앗 차이)
        mtr = args.metric
        full = by_arm.get("full", {})
        print(f"\n-- full 대비 짝 비교 ({mtr}) --")
        for a in arms:
            if a == "full":
                continue
            common = sorted(set(full) & set(by_arm[a]))
            if len(common) < 3:
                print(f"{a:16s} (짝 부족)")
                continue
            d = np.array([full[s][mtr] - by_arm[a][s][mtr] for s in common], float)
            lo, hi = boot_ci(d)
            if np.allclose(d, 0):
                w_p = 1.0
            else:
                w_p = float(sps.wilcoxon(d).pvalue)
            u_p = float(sps.mannwhitneyu(
                [full[s][mtr] for s in common],
                [by_arm[a][s][mtr] for s in common]).pvalue)
            print(f"{a:16s} full−{a} 평균차 {d.mean():+7.3f} CI[{lo:+.3f},{hi:+.3f}] "
                  f"| Wilcoxon p={w_p:.4f} | MW p={u_p:.4f}")


if __name__ == "__main__":
    main()
