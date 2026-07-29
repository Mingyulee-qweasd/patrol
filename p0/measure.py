"""P0 채점기: main_results.jsonl → error_model.json + go/no-go 리포트.

산출 스키마는 sim/vlm.py의 합성 모델과 동일:
    model[category][band] = {"p_detect": .., "judgments": [{is_task,n,u,conf_mu,conf_sd,p}, ..]}
p_detect(원거리에서 애초에 알아챌 확률)는 P0가 재는 대상이 아니라 감지의 물리 가정
(near .98 / far .85, 합성값 유지 — 논문에 명시). P0가 재는 것 = "봤을 때의 판정 분포".
is_task=false 판정도 judgments에 그대로 들어가 미판정(놓침)과 오판정을 구분한다.
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
P_DETECT = {"near": 0.98, "far": 0.85}
TASK_CATS = ["trash", "sharps", "carcass_s", "carcass_l", "bulky", "obstacle"]
NONTASK_CATS = ["normal", "ambiguous", "hazard"]


def norm_n(v) -> int:
    s = str(v)
    return int(s) if s.isdigit() else 4  # "beyond" → 4


def load_rows(path: Path) -> tuple[list[dict], dict]:
    """유효 판정 행 + (category,band)별 무응답 수.

    무응답(3단 회복 실패)은 '그 관측은 아무것도 내놓지 못함' = 실측된 응답 실패율로
    p_detect에 곱해 반영한다 (제외하고 잊는 게 아니라 감지 모델의 일부로).
    """
    rows = [json.loads(l) for l in open(path)]
    ok = [r for r in rows if isinstance(r.get("judgment"), dict)]
    fails = defaultdict(int)
    for r in rows:
        if not isinstance(r.get("judgment"), dict):
            cat = r["file"].split("/")[-2]
            fails[(cat, r["band"])] += 1
    if fails:
        print(f"[반영] 무응답 {sum(fails.values())}건 → 해당 칸 p_detect에 응답률 곱함")
    return ok, dict(fails)


def build_error_model(rows: list[dict], fails: dict) -> dict:
    """(category, band)별 판정 경험 분포. 문안 3종은 같은 조건의 반복 표본으로 합산."""
    cells = defaultdict(list)
    for r in rows:
        cells[(r["category"], r["band"])].append(r["judgment"])
    model = {}
    for (cat, band), js in sorted(cells.items()):
        resp_rate = len(js) / (len(js) + fails.get((cat, band), 0))
        groups = defaultdict(list)  # (is_task, n, u) → conf 목록
        for j in js:
            key = (bool(j["is_task"]), norm_n(j["n_robots"]), int(j["urgency"]))
            groups[key].append(float(j["confidence"]))
        judgments = []
        for (is_task, n, u), confs in sorted(groups.items()):
            mu = sum(confs) / len(confs)
            sd = math.sqrt(sum((c - mu) ** 2 for c in confs) / len(confs)) if len(confs) > 1 else 5.0
            judgments.append({"is_task": is_task, "n": n, "u": u,
                              "conf_mu": round(mu, 1), "conf_sd": round(max(sd, 2.0), 1),
                              "p": round(len(confs) / len(js), 4)})
        model.setdefault(cat, {})[band] = {"p_detect": round(P_DETECT[band] * resp_rate, 4),
                                           "judgments": judgments, "n_samples": len(js),
                                           "resp_rate": round(resp_rate, 4)}
    return model


def report(rows: list[dict]) -> dict:
    out = {}

    # 1) is_task 정확도: task류 = true가 정답 / 비개입류(normal·ambiguous·hazard) = false가 정답
    acc = defaultdict(lambda: [0, 0])
    for r in rows:
        want = r["category"] in TASK_CATS
        got = bool(r["judgment"]["is_task"])
        k = (r["category"], r["band"])
        acc[k][0] += int(got == want)
        acc[k][1] += 1
    print("\n== ① 개입/비개입 판정 정확도 (category × band) ==")
    for cat in TASK_CATS + NONTASK_CATS:
        line = f"  {cat:10s}"
        for band in ("near", "far"):
            c, t = acc.get((cat, band), (0, 0))
            line += f"  {band} {c}/{t} ({c / t * 100:3.0f}%)" if t else f"  {band}    -    "
        print(line)
    task_rows = [r for r in rows if r["category"] in TASK_CATS]
    non_rows = [r for r in rows if r["category"] in NONTASK_CATS]
    tpr = sum(bool(r["judgment"]["is_task"]) for r in task_rows) / max(len(task_rows), 1)
    fpr = sum(bool(r["judgment"]["is_task"]) for r in non_rows) / max(len(non_rows), 1)
    out["tpr"], out["fpr"] = tpr, fpr
    print(f"  전체: 임무를 임무로 {tpr * 100:.0f}% / 비개입을 임무로(오경보) {fpr * 100:.0f}%")

    # 2) n̂ 사이징 (go/no-go 핵심): 임무로 본 task류에서 GT n과 일치율 vs "전부 1대" 기준선
    print("\n== ② 필요 대수 추정 (임무로 판정한 task류만) ==")
    for band in ("near", "far"):
        sub = [r for r in task_rows if r["band"] == band and r["judgment"]["is_task"]]
        if not sub:
            continue
        hit = sum(norm_n(r["judgment"]["n_robots"]) == r["gt_n"] for r in sub)
        base = sum(r["gt_n"] == 1 for r in sub)  # 전부 1대라고 찍는 기준선
        n2 = [r for r in sub if r["gt_n"] >= 2]
        hit2 = sum(norm_n(r["judgment"]["n_robots"]) >= 2 for r in n2)
        print(f"  {band}: 일치 {hit}/{len(sub)} ({hit / len(sub) * 100:.0f}%)"
              f"  vs 전부1대 기준선 {base / len(sub) * 100:.0f}%"
              f"  | 다수필요(n≥2)를 다수로 {hit2}/{len(n2)}"
              f" ({hit2 / max(len(n2), 1) * 100:.0f}%)")
        out[f"n_acc_{band}"] = hit / len(sub)
        out[f"n_base_{band}"] = base / len(sub)
        out[f"n2_recall_{band}"] = hit2 / max(len(n2), 1)

    # 3) 확신 보정 (ECE, 10칸): confidence vs ①의 정오
    bins = [[0, 0, 0.0] for _ in range(10)]
    for r in rows:
        want = r["category"] in TASK_CATS
        got = bool(r["judgment"]["is_task"])
        conf = float(r["judgment"]["confidence"]) / 100
        b = min(int(conf * 10), 9)
        bins[b][0] += int(got == want)
        bins[b][1] += 1
        bins[b][2] += conf
    ece = sum(abs(c / t - s / t) * t for c, t, s in bins if t) / len(rows)
    out["ece"] = ece
    print(f"\n== ③ 확신 보정: ECE {ece:.3f} ==")
    for i, (c, t, s) in enumerate(bins):
        if t:
            print(f"  확신 {i * 10}-{i * 10 + 10}: 말한 확신 {s / t * 100:3.0f} vs 실제 정답률 {c / t * 100:3.0f}  (n={t})")

    # 4) 문안 안정성: 같은 (이미지, band)에서 3문안 is_task 전원일치 비율
    votes = defaultdict(list)
    for r in rows:
        votes[(r["file"], r["band"])].append(bool(r["judgment"]["is_task"]))
    full = [v for v in votes.values() if len(v) == 3]
    stable = sum(len(set(v)) == 1 for v in full) / max(len(full), 1)
    out["prompt_stability"] = stable
    print(f"\n== ④ 문안 안정성: 3문안 전원일치 {stable * 100:.0f}% ({len(full)}쌍 기준) ==")
    return out


def main():
    src = HERE / "out/main_results.jsonl"
    rows, fails = load_rows(src)
    total = 1002
    partial = len(rows) + sum(fails.values()) < total
    print(f"판정 {len(rows)}/{total}건 채점 (무응답 {sum(fails.values())}건 반영)"
          + (" [부분 — 완주 후 재실행]" if partial else " [완결]"))
    stats = report(rows)
    model = build_error_model(rows, fails)
    dst = HERE / ("out/error_model_partial.json" if partial else "out/error_model.json")
    json.dump({"stats": stats, "model": model, "n_rows": len(rows)},
              open(dst, "w"), indent=1, ensure_ascii=False)
    print(f"\n저장: {dst.name}")


if __name__ == "__main__":
    main()
