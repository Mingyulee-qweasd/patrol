"""본 실험 실행기 — 씨앗 × 비교군 × 부하 전 조합, 병렬 + 이어하기.

짝 설계: 같은 (씨앗, 부하)면 전 비교군이 같은 임무 스트림을 받음 (paired — 검정력↑).
사용:  python exp/run_all.py --pilot          # 씨앗 3 × 비교군 8 × 부하 1종 (리허설)
       python exp/run_all.py                  # 본 실험 (25 × 8 × 3 = 600회)
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

ARMS8 = ["full", "no-sizing", "no-gate", "fixed-rdv", "solo-only",
         "broadcast", "greedy-reactive", "role-adaptive"]
LOADS = [0.15, 0.25, 0.4]   # 저·중·고 (D5 부하 보정에서 채택된 축)


def run_one(job: dict) -> dict:
    from sim.runner import Episode
    from sim.vlm import load_model
    from sim.metrics import compute
    em = load_model(HERE.parent / "p0/out/error_model.json")
    ep = Episode(str(HERE / "configs" / job["cfg"]), seed=job["seed"],
                 horizon_s=job["horizon"], warmup_s=1200, rho=0.5,
                 error_model=em, arm=job["arm"], lambda_calib=job["load"])
    ep.run()
    m = compute(ep)
    return {**job, **m.as_dict()}


def key(j: dict) -> str:
    return f'{j["cfg"]}|{j["seed"]}|{j["arm"]}|{j["load"]}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="씨앗 3·부하 0.25만")
    ap.add_argument("--seeds", type=int, default=25)
    ap.add_argument("--horizon", type=float, default=43200)
    ap.add_argument("--cfg", default="simple.yaml")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seeds = range(1, (4 if args.pilot else args.seeds + 1))
    loads = [0.25] if args.pilot else LOADS
    out = Path(args.out or HERE / ("out/pilot.jsonl" if args.pilot else "out/main.jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out.exists():  # 이어하기
        done = {key(json.loads(l)) for l in open(out)}
    jobs = [{"cfg": args.cfg, "seed": s, "arm": a, "load": ld, "horizon": args.horizon}
            for s in seeds for ld in loads for a in ARMS8]
    jobs = [j for j in jobs if key(j) not in done]
    print(f"실행 {len(jobs)}회 (기완료 {len(done)} 스킵), 병렬 {args.procs}")

    with ProcessPoolExecutor(max_workers=args.procs) as ex, open(out, "a") as f:
        futs = {ex.submit(run_one, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            j = futs[fut]
            try:
                row = fut.result()
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
            except Exception as e:
                print(f"[실패] {key(j)}: {e}")
            if i % 20 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
