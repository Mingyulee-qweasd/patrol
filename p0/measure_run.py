"""P0 본 측정 — labels.csv 전 이미지 × (원본 near + 합성 far) × 동결 3문안.

산출: out/main_results.jsonl (한 줄 = 한 판정). 캐시 덕에 중단 후 재실행 = 이어하기.
합성 far: 장변 200px 다운스케일 (원거리 저해상 관측의 대리 — 방식 논문 명시).
"""
import csv
import json
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import client_ollama as client
from pilot import RESPONSE_SCHEMA

ROOT = Path(__file__).parent
FAR_DIR = ROOT / "images" / "far_synth"
OUT = ROOT / "out" / "main_results.jsonl"
PROMPTS = {
    "frozen": ROOT / "prompts/frozen_v1.txt",
    "para1": ROOT / "prompts/frozen_v1_para1.txt",
    "para2": ROOT / "prompts/frozen_v1_para2.txt",
}
GEN = {"temperature": 0, "responseMimeType": "application/json",
       "responseSchema": RESPONSE_SCHEMA}


def far_variant(src: Path) -> Path:
    dst = FAR_DIR / src.parent.name / src.name
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    im = Image.open(src).convert("RGB")
    im.thumbnail((200, 200))
    im.save(dst, quality=85)
    return dst


def main():
    rows = list(csv.DictReader(open(ROOT / "labels.csv")))
    prompts = {k: p.read_text() for k, p in PROMPTS.items()}
    done = set()
    if OUT.exists():
        for line in open(OUT):
            try:
                r = json.loads(line)
                done.add((r["file"], r["band"], r["prompt"]))
            except Exception:
                pass
    total = len(rows) * 2 * len(prompts)
    print(f"대상 {len(rows)}장 × 2밴드 × {len(prompts)}문안 = {total}판정 (기완료 {len(done)})", flush=True)

    out_f = open(OUT, "a")
    n_ok = n_fail = 0
    t0 = time.time()
    for i, row in enumerate(rows):
        src = ROOT / row["file"]
        for band, path in [("near", src), ("far", far_variant(src))]:
            for pk, ptxt in prompts.items():
                key = (row["file"], band, pk)
                if key in done:
                    continue
                try:
                    resp = client.call(ptxt, image_path=path, gen_config=GEN,
                                       tag=f"p0main:{band}:{pk}")
                    j = client.extract_json(resp)
                    rec = {"file": row["file"], "category": row["category"],
                           "gt_class": row["gt_class"], "gt_n": int(row["n"]),
                           "gt_u": int(row["u"]), "tag": row.get("tag", ""),
                           "band": band, "prompt": pk, "judgment": j,
                           "latency_s": resp.get("_latency_s")}
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_f.flush()
                    n_ok += 1
                except Exception as e:
                    n_fail += 1
                    out_f.write(json.dumps({"file": row["file"], "band": band,
                                            "prompt": pk, "error": str(e)[:150]}) + "\n")
                    out_f.flush()
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            done_now = n_ok + n_fail
            rate = done_now / max(el, 1)
            remain = (total - len(done) - done_now) / max(rate, 1e-9)
            print(f"[{i+1}/{len(rows)}장] 판정 {done_now} (실패 {n_fail}) — "
                  f"경과 {el/60:.0f}분, 남은 예상 {remain/60:.0f}분", flush=True)
    print(f"P0MAIN_DONE ok={n_ok} fail={n_fail}", flush=True)


if __name__ == "__main__":
    main()
