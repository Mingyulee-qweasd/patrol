"""실내 연구(v3) 이미지 수집 — Wikimedia Commons + Openverse (CC 라이선스).

임무 후보 8종 × 12장 목표. 후보는 3배수(36장) 모아 격자 시트로 사람 검수 후 채택.
산출:
  p0/images_indoor/candidates/<타입>/<타입>_c###.jpg  + candidates/meta_candidates.jsonl
  p0/images_indoor/raw/<타입>_##.jpg + p0/images_indoor/meta.jsonl  (채택본)
  p0/images_indoor/review/<타입>_sheet.jpg  (검수 시트)

사용:
  python collect_indoor.py collect              # 후보 풀 수집
  python collect_indoor.py sheet                # 후보 검수 시트 생성 (번호 격자)
  python collect_indoor.py adopt picks.json     # {"타입": [후보번호,...]} → raw/ 채택 + meta.jsonl
  python collect_indoor.py final-sheet          # 채택본 검수 시트 생성 (한글 캡션)
"""
import argparse
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent
BASE = ROOT / "images_indoor"
CAND = BASE / "candidates"
RAW = BASE / "raw"
REVIEW = BASE / "review"
META = BASE / "meta.jsonl"
CAND_META = CAND / "meta_candidates.jsonl"
UA = {"User-Agent": "patrol-research-v3/0.1 (academic; contact: alsrb000929@gmail.com)"}
FONT = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"

TARGET_CAND = 36  # 12장 × 3배수

# 타입 → (검색어 목록, 한글 캡션)
TYPES = {
    "indoor_trash": ([
        "litter on floor hallway", "trash on office floor", "litter corridor indoor",
        "trash on floor building", "rubbish on floor indoors", "messy floor lobby",
    ], "바닥 쓰레기(컵·포장지·휴지)"),
    "spill": ([
        "spilled coffee floor", "spilled drink floor", "coffee spill office",
        "spilled milk floor", "water puddle floor indoor", "liquid spill floor",
    ], "흘린 음료·물웅덩이"),
    "fallen_papers": ([
        "papers scattered floor office", "papers on floor", "documents scattered floor",
        "books on floor library", "dropped papers floor", "scattered paperwork floor",
    ], "떨어진 유인물·책·서류"),
    "fallen_sign": ([
        "fallen sign indoor", "banner stand fallen", "sign fallen floor",
        "toppled sign indoor", "fallen wet floor sign", "roll up banner fallen",
    ], "쓰러진 입간판·배너 스탠드"),
    "blocking_chair": ([
        "chair in hallway", "chairs in corridor", "office chair hallway",
        "chair blocking corridor", "desk in hallway", "furniture in corridor",
    ], "통로를 막은 의자·책상"),
    "package_box": ([
        "cardboard boxes hallway", "packages at door", "parcel at front door",
        "cardboard boxes corridor", "delivery boxes doorstep", "cardboard boxes office floor",
    ], "방치된 택배·골판지 박스"),
    "tipped_bin": ([
        "knocked over trash can", "overturned trash can", "tipped over garbage bin",
        "overturned bin litter", "fallen trash can", "overturned wastebasket",
    ], "넘어진 쓰레기통"),
    "fallen_object": ([
        "fire extinguisher on floor", "fallen picture frame floor", "fallen fire extinguisher",
        "picture frame on floor", "fallen shelf office", "bulletin board fallen",
    ], "떨어진 벽부착물(소화기·액자 등)"),
}


def commons_search(query: str, limit: int = 20) -> list[dict]:
    api = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query", "format": "json", "generator": "search",
        "gsrsearch": f"filetype:bitmap {query}", "gsrnamespace": 6, "gsrlimit": limit,
        "prop": "imageinfo", "iiprop": "url|extmetadata", "iiurlwidth": 1024,
    }
    try:
        r = requests.get(api, params=params, headers=UA, timeout=30)
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
    except Exception as e:
        print(f"  commons 오류({query}): {e}")
        return []
    out = []
    for p in pages.values():
        ii = (p.get("imageinfo") or [{}])[0]
        url = ii.get("thumburl") or ii.get("url")
        if not url:
            continue
        lic = (ii.get("extmetadata", {}).get("LicenseShortName", {}) or {}).get("value", "?")
        out.append({"url": url, "title": p.get("title", ""), "license": lic,
                    "source": "commons"})
    return out


def openverse_search(query: str, limit: int = 20) -> list[dict]:
    api = "https://api.openverse.org/v1/images/"
    params = {"q": query, "page_size": limit, "extension": "jpg,jpeg"}
    for attempt in range(3):
        try:
            r = requests.get(api, params=params, headers=UA, timeout=30)
            if r.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"  openverse 429 — {wait}s 대기")
                time.sleep(wait)
                continue
            r.raise_for_status()
            results = r.json().get("results", [])
            break
        except Exception as e:
            print(f"  openverse 오류({query}): {e}")
            return []
    else:
        return []
    out = []
    for it in results:
        url = it.get("url")
        if not url:
            continue
        lic = f"CC {it.get('license', '?').upper()} {it.get('license_version', '')}".strip()
        if it.get("license") in ("cc0", "pdm"):
            lic = {"cc0": "CC0", "pdm": "Public Domain"}[it["license"]]
        out.append({"url": url, "title": it.get("title") or "",
                    "license": lic, "source": "openverse",
                    "landing": it.get("foreign_landing_url", ""),
                    "creator": it.get("creator", "")})
    return out


def fetch_image(url: str) -> Image.Image | None:
    try:
        r = requests.get(url, headers=UA, timeout=60)
        r.raise_for_status()
        if len(r.content) < 15_000:
            return None
        img = Image.open(io.BytesIO(r.content))
        img.load()
        if min(img.size) < 250:
            return None
        return img
    except Exception:
        return None


def cmd_collect():
    CAND.mkdir(parents=True, exist_ok=True)
    seen_urls, seen_hash = set(), set()
    if CAND_META.exists():
        for line in CAND_META.read_text().splitlines():
            d = json.loads(line)
            seen_urls.add(d["url"])
    meta_f = open(CAND_META, "a")
    for t, (queries, _) in TYPES.items():
        d = CAND / t
        d.mkdir(exist_ok=True)
        n = len(list(d.glob("*.jpg")))
        if n >= TARGET_CAND:
            print(f"[{t}] 이미 {n}장 — 건너뜀")
            continue
        for q in queries:
            if n >= TARGET_CAND:
                break
            items = openverse_search(q) + commons_search(q)
            time.sleep(2.0)
            for item in items:
                if n >= TARGET_CAND:
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
                dest = d / f"{t}_c{n:03d}.jpg"
                img.convert("RGB").save(dest, "JPEG", quality=90)
                meta_f.write(json.dumps({
                    "file": str(dest.relative_to(ROOT)), "type": t, "query": q, **item,
                }, ensure_ascii=False) + "\n")
                meta_f.flush()
                n += 1
                time.sleep(0.4)
        print(f"[{t}] 후보 {n}장 (목표 {TARGET_CAND})")
    meta_f.close()


def _font(size: int):
    try:
        return ImageFont.truetype(FONT, size)
    except Exception:
        return ImageFont.load_default()


def make_sheet(files: list[Path], labels: list[str], dest: Path,
               cols: int = 4, cell: int = 380, title: str = ""):
    rows = (len(files) + cols - 1) // cols
    cap_h = 34
    top = 48 if title else 8
    W = cols * cell + 8
    H = top + rows * (cell + cap_h) + 8
    sheet = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(sheet)
    f_cap = _font(24)
    if title:
        dr.text((10, 8), title, fill="black", font=_font(30))
    for i, (fp, lab) in enumerate(zip(files, labels)):
        r, c = divmod(i, cols)
        x = 4 + c * cell
        y = top + r * (cell + cap_h)
        try:
            im = Image.open(fp).convert("RGB")
            im.thumbnail((cell - 8, cell - 8))
            sheet.paste(im, (x + (cell - im.width) // 2, y + (cell - im.height) // 2))
        except Exception:
            dr.text((x + 10, y + 10), "ERR", fill="red", font=f_cap)
        dr.text((x + 6, y + cell + 2), lab, fill="black", font=f_cap)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dest, "JPEG", quality=88)


def cmd_sheet():
    REVIEW.mkdir(parents=True, exist_ok=True)
    per_sheet = 12
    for t in TYPES:
        files = sorted((CAND / t).glob("*.jpg"))
        for k in range(0, len(files), per_sheet):
            chunk = files[k:k + per_sheet]
            labels = [fp.stem.split("_c")[-1] for fp in chunk]
            dest = REVIEW / f"cand_{t}_{k // per_sheet}.jpg"
            make_sheet(chunk, labels, dest, cols=4, cell=380, title=f"{t} candidates {k}-{k+len(chunk)-1}")
            print(dest)


def cmd_adopt(picks_path: str):
    picks = json.loads(Path(picks_path).read_text())
    RAW.mkdir(parents=True, exist_ok=True)
    cand_meta = {}
    for line in CAND_META.read_text().splitlines():
        d = json.loads(line)
        cand_meta[d["file"]] = d
    meta_f = open(META, "a")
    for t, idxs in picks.items():
        for j, idx in enumerate(idxs, 1):
            src = CAND / t / f"{t}_c{idx:03d}.jpg"
            dest = RAW / f"{t}_{j:02d}.jpg"
            Image.open(src).convert("RGB").save(dest, "JPEG", quality=92)
            m = dict(cand_meta.get(str(src.relative_to(ROOT)), {}))
            m.pop("file", None)
            meta_f.write(json.dumps({
                "file": str(dest.relative_to(ROOT)), **m,
            }, ensure_ascii=False) + "\n")
        print(f"[{t}] {len(idxs)}장 채택")
    meta_f.close()


def cmd_final_sheet():
    REVIEW.mkdir(parents=True, exist_ok=True)
    for t, (_, kor) in TYPES.items():
        files = sorted(RAW.glob(f"{t}_*.jpg"))
        if not files:
            continue
        labels = [fp.stem.split("_")[-1] for fp in files]
        dest = REVIEW / f"{t}_sheet.jpg"
        make_sheet(files, labels, dest, cols=4, cell=380, title=f"{t} — {kor} ({len(files)}장)")
        print(dest)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "sheet", "adopt", "final-sheet"])
    ap.add_argument("arg", nargs="?")
    a = ap.parse_args()
    if a.cmd == "collect":
        cmd_collect()
    elif a.cmd == "sheet":
        cmd_sheet()
    elif a.cmd == "adopt":
        cmd_adopt(a.arg or sys.exit("picks.json 경로 필요"))
    else:
        cmd_final_sheet()
