"""실내 연구(v3) — 근접 함정(ambiguous) 이미지 수집.

임무처럼 보이지만 임무가 아닌 실내 장면 20장 (임무 8종의 시각적 이웃, 각 2~3장).
collect_indoor.py의 검색·시트 함수를 재사용 (기존 파일 무수정).

산출:
  p0/images_indoor/candidates/indoor_ambiguous/<서브타입>/amb_<서브타입>_c###.jpg
    + candidates/indoor_ambiguous/meta_candidates_ambiguous.jsonl
  p0/images_indoor/raw/indoor_ambiguous_###.jpg + images_indoor/meta_ambiguous.jsonl
  p0/images_indoor/review/indoor_ambiguous_sheet.jpg

사용:
  python collect_ambiguous.py collect            # 후보 풀 수집 (서브타입별 ~9장)
  python collect_ambiguous.py sheet              # 후보 검수 시트 생성
  python collect_ambiguous.py adopt picks.json   # {"서브타입": [후보번호,...]} → raw 채택
  python collect_ambiguous.py final-sheet        # 채택본 검수 시트 (한글 캡션)
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from PIL import Image

from collect_indoor import (ROOT, BASE, RAW, REVIEW,
                            commons_search, openverse_search, fetch_image, make_sheet)

CAND_A = BASE / "candidates" / "indoor_ambiguous"
CAND_META_A = CAND_A / "meta_candidates_ambiguous.jsonl"
META_A = BASE / "meta_ambiguous.jsonl"
TYPE = "indoor_ambiguous"
TARGET_CAND = 9  # 서브타입별 2~3장 채택 × 3배수

# 서브타입 → (검색어, 한글 캡션 "무엇 — 어느 임무의 함정")
SUBTYPES = {
    "arranged_chairs": ([
        "conference room chairs arranged", "lecture hall rows seats", "meeting room chairs table",
        "classroom desks rows", "auditorium seating rows",
    ], "정렬된 의자 (함정대상: blocking_chair)"),
    "shiny_floor": ([
        "polished floor hallway reflection", "shiny marble floor lobby", "glossy floor corridor",
        "reflective tile floor interior", "waxed floor hallway",
    ], "광택 마른 바닥 (함정대상: spill)"),
    "delivered_package": ([
        "package at door delivered", "parcel on doormat", "amazon package front door",
        "delivery box porch door", "package delivery doorstep",
    ], "정상 배송 택배 (함정대상: package_box)"),
    "posted_flyers": ([
        "bulletin board flyers", "notice board posters office", "community bulletin board indoor",
        "flyers pinned board", "poster display wall corridor",
    ], "게시판 인쇄물 (함정대상: fallen_papers)"),
    "standing_sign": ([
        "standing banner indoor", "roll up banner exhibition", "wet floor sign standing",
        "sign stand lobby", "information sign board indoor event",
    ], "서있는 입간판 (함정대상: fallen_sign)"),
    "upright_bin": ([
        "trash can hallway", "recycling bins indoor", "waste bin office corner",
        "garbage bin building interior", "trash bins lobby",
    ], "제자리 쓰레기통 (함정대상: tipped_bin)"),
    "janitor_cart": ([
        "janitor cart hallway", "cleaning cart hotel corridor", "housekeeping trolley hallway",
        "mop bucket janitor", "cleaning supplies cart indoor",
    ], "청소도구 비치 (함정대상: belongings)"),
    "mounted_wall": ([
        "fire extinguisher mounted wall", "fire extinguisher on wall indoor", "picture frames on wall",
        "framed paintings wall gallery", "wall mounted fire extinguisher corridor",
    ], "벽부착 정상 (함정대상: fallen_object)"),
}


# 2차 재검색(1차 수율 낮은 서브타입 보강) — 서브타입 → (검색어, 추가 장수)
ROUND2 = {
    "arranged_chairs": ([
        "empty classroom rows of desks", "empty lecture hall seats", "empty conference room table",
        "meeting room empty chairs", "waiting room chairs row", "empty auditorium seats",
    ], 9),
    "delivered_package": ([
        "parcel on doorstep", "package delivered front door", "package on doormat",
        "cardboard box at apartment door", "postal parcel delivery door", "package left at door",
        # 3차: 실내 복도·현관 지향
        "package apartment hallway", "parcel outside apartment door", "box on welcome mat",
        "package delivery porch box", "parcels lobby mailroom", "package by the door",
    ], 9),
    "upright_bin": ([
        "wastebasket office", "recycling bin lobby", "trash bin airport terminal",
        "garbage bin shopping mall interior", "waste paper basket room", "dustbin corridor building",
        "trash can classroom", "bin station office recycling",
    ], 9),
    "mounted_wall": ([
        "framed paintings hanging wall museum", "pictures hanging hallway wall",
        "framed photographs on wall interior",
    ], 6),
}


def cmd_collect(round2: bool = False):
    CAND_A.mkdir(parents=True, exist_ok=True)
    seen_urls, seen_hash = set(), set()
    if CAND_META_A.exists():
        for line in CAND_META_A.read_text().splitlines():
            seen_urls.add(json.loads(line)["url"])
    meta_f = open(CAND_META_A, "a")
    plan = (ROUND2 if round2 else
            {s: (qs, TARGET_CAND) for s, (qs, _) in SUBTYPES.items()})
    for sub, (queries, tgt) in plan.items():
        d = CAND_A / sub
        d.mkdir(exist_ok=True)
        n = len(list(d.glob("*.jpg")))
        target = n + tgt if round2 else tgt
        if n >= target:
            print(f"[{sub}] 이미 {n}장 — 건너뜀")
            continue
        for q in queries:
            if n >= target:
                break
            items = openverse_search(q) + commons_search(q)
            time.sleep(2.0)
            for item in items:
                if n >= target:
                    break
                if item["url"] in seen_urls:
                    continue
                seen_urls.add(item["url"])
                img = fetch_image(item["url"])
                if img is None:
                    continue
                h = hashlib.sha1(img.convert("RGB").resize((64, 64)).tobytes()).hexdigest()
                if h in seen_hash:
                    continue
                seen_hash.add(h)
                dest = d / f"amb_{sub}_c{n:03d}.jpg"
                img.convert("RGB").save(dest, "JPEG", quality=90)
                meta_f.write(json.dumps({
                    "file": str(dest.relative_to(ROOT)), "type": TYPE,
                    "subtype": sub, "query": q, **item,
                }, ensure_ascii=False) + "\n")
                meta_f.flush()
                n += 1
                time.sleep(0.4)
        print(f"[{sub}] 후보 {n}장 (목표 {target})")
    meta_f.close()


def cmd_sheet():
    REVIEW.mkdir(parents=True, exist_ok=True)
    for sub in SUBTYPES:
        files = sorted((CAND_A / sub).glob("*.jpg"))
        if not files:
            continue
        labels = [fp.stem.split("_c")[-1] for fp in files]
        dest = REVIEW / f"cand_ambiguous_{sub}.jpg"
        make_sheet(files, labels, dest, cols=3, cell=380,
                   title=f"ambiguous/{sub} candidates ({len(files)})")
        print(dest)


def cmd_adopt(picks_path: str):
    picks = json.loads(Path(picks_path).read_text())
    RAW.mkdir(parents=True, exist_ok=True)
    cand_meta = {}
    for line in CAND_META_A.read_text().splitlines():
        d = json.loads(line)
        cand_meta[d["file"]] = d
    meta_f = open(META_A, "a")
    k = 0
    for sub, idxs in picks.items():
        for idx in idxs:
            src = CAND_A / sub / f"amb_{sub}_c{idx:03d}.jpg"
            dest = RAW / f"{TYPE}_{k:03d}.jpg"
            Image.open(src).convert("RGB").save(dest, "JPEG", quality=92)
            m = dict(cand_meta.get(str(src.relative_to(ROOT)), {}))
            m.pop("file", None)
            meta_f.write(json.dumps({
                "file": str(dest.relative_to(ROOT)), **m,
            }, ensure_ascii=False) + "\n")
            k += 1
        print(f"[{sub}] {len(idxs)}장 채택")
    meta_f.close()
    print(f"총 {k}장 → {RAW}/{TYPE}_***.jpg")


def cmd_final_sheet():
    REVIEW.mkdir(parents=True, exist_ok=True)
    files = sorted(RAW.glob(f"{TYPE}_*.jpg"))
    sub_of = {}
    for line in META_A.read_text().splitlines():
        d = json.loads(line)
        sub_of[d["file"]] = d.get("subtype", "?")
    labels = []
    for fp in files:
        sub = sub_of.get(str(fp.relative_to(ROOT)), "?")
        kor = SUBTYPES.get(sub, ([], "?"))[1]
        labels.append(f"{fp.stem.split('_')[-1]} {kor}")
    dest = REVIEW / f"{TYPE}_sheet.jpg"
    make_sheet(files, labels, dest, cols=3, cell=520,
               title=f"{TYPE} — 근접 함정(임무 아님) {len(files)}장")
    print(dest)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "collect2", "sheet", "adopt", "final-sheet"])
    ap.add_argument("arg", nargs="?")
    a = ap.parse_args()
    if a.cmd == "collect":
        cmd_collect()
    elif a.cmd == "collect2":
        cmd_collect(round2=True)
    elif a.cmd == "sheet":
        cmd_sheet()
    elif a.cmd == "adopt":
        cmd_adopt(a.arg or sys.exit("picks.json 경로 필요"))
    else:
        cmd_final_sheet()
