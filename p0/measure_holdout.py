"""홀드아웃 일반화 측정 — 온톨로지 밖 12타입 × 2장 × 2밴드 × 동결 문안 = 48판정.

본 측정과 분리된 부록 실험 (동결 문안·설정 그대로 사용 — 동결 위반 아님).
질문: 채점표에 없던 타입도 ①임무면 상식대로 발견·사이징하는가 ②함정이면 손대지 않는가.
"""
import csv
import json
import time
from pathlib import Path

from measure_run import PROMPTS, GEN, far_variant, client  # 동결 설정 재사용

ROOT = Path(__file__).parent
OUT = ROOT / "out/holdout_results.jsonl"


def main():
    rows = list(csv.DictReader(open(ROOT / "labels_holdout.csv")))
    ptxt = PROMPTS["frozen"].read_text()
    done = set()
    if OUT.exists():
        done = {(json.loads(l)["file"], json.loads(l)["band"]) for l in open(OUT)}
    print(f"대상 {len(rows)}장 × 2밴드 = {len(rows) * 2}판정 (기완료 {len(done)})", flush=True)
    out_f = open(OUT, "a")
    for row in rows:
        src = ROOT / row["file"]
        for band, path in [("near", src), ("far", far_variant(src))]:
            if (row["file"], band) in done:
                continue
            try:
                resp = client.call(ptxt, image_path=path, gen_config=GEN,
                                   tag=f"holdout:{band}")
                j = client.extract_json(resp)
            except Exception as e:
                j = None
                print(f"[실패] {row['file']} {band}: {e}", flush=True)
            out_f.write(json.dumps({"file": row["file"], "category": row["category"],
                                    "gt_class": row["gt_class"], "gt_n": int(row["n"]),
                                    "gt_u": int(row["u"]), "band": band,
                                    "judgment": j}, ensure_ascii=False) + "\n")
            out_f.flush()
    print("완료", flush=True)


if __name__ == "__main__":
    main()
