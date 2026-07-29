"""P0 이미지 수집 — TACO(쓰레기) + Wikimedia Commons(나머지 카테고리, CC 라이선스).

산출: p0/images/raw/{카테고리}/*.jpg + p0/images/raw/meta.jsonl (출처·라이선스)
이후 사람 검수(큐레이션)를 거쳐 labels.csv 확정 — 자동 수집은 후보 풀일 뿐.
"""
import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).parent
RAW = ROOT / "images" / "raw"
META = RAW / "meta.jsonl"
UA = {"User-Agent": "patrol-research-p0/0.1 (academic; contact: alsrb000929@gmail.com)"}

# 카테고리 → (Commons 검색어 목록, 목표 장수)  — 온톨로지 3안 기준
CATEGORIES = {
    "trash":      (["litter on ground", "trash on street", "plastic bottle litter"], 30),
    "bulky":      (["illegal dumping furniture", "abandoned sofa street", "discarded mattress street"], 30),
    "carcass":    (["dead pigeon", "roadkill animal road", "dead rat street"], 30),
    "obstacle":   (["fallen tree across road", "fallen tree blocking path", "storm fallen tree street"], 30),
    "normal":     (["park footpath", "campus walkway", "clean sidewalk park"], 40),
    "ambiguous":  (["pile of autumn leaves path", "bag on park bench", "person lying grass park",
                    "outdoor sculpture park", "puddle on path", "construction sign sidewalk"], 40),
    "hazard":     (["discarded syringe ground", "broken glass on street", "fallen power line street"], 25),
}


def commons_search(query: str, limit: int = 12) -> list[dict]:
    """Wikimedia Commons에서 CC 이미지 검색 → [{url, title, license}]."""
    api = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query", "format": "json", "generator": "search",
        "gsrsearch": f"filetype:bitmap {query}", "gsrnamespace": 6, "gsrlimit": limit,
        "prop": "imageinfo", "iiprop": "url|extmetadata", "iiurlwidth": 1024,
    }
    r = requests.get(api, params=params, headers=UA, timeout=30)
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", {})
    out = []
    for p in pages.values():
        ii = (p.get("imageinfo") or [{}])[0]
        url = ii.get("thumburl") or ii.get("url")
        if not url:
            continue
        lic = (ii.get("extmetadata", {}).get("LicenseShortName", {}) or {}).get("value", "?")
        out.append({"url": url, "title": p.get("title", ""), "license": lic})
    return out


def download(url: str, dest: Path) -> bool:
    try:
        r = requests.get(url, headers=UA, timeout=60)
        r.raise_for_status()
        if len(r.content) < 10_000:  # 아이콘·깨진 파일 배제
            return False
        dest.write_bytes(r.content)
        return True
    except Exception:
        return False


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    meta_f = open(META, "a")
    for cat, (queries, target) in CATEGORIES.items():
        d = RAW / cat
        d.mkdir(exist_ok=True)
        have = len(list(d.glob("*.jpg")))
        if have >= target:
            print(f"[{cat}] 이미 {have}장 — 건너뜀")
            continue
        n = have
        for q in queries:
            if n >= target:
                break
            for item in commons_search(q):
                if n >= target:
                    break
                dest = d / f"{cat}_{n:03d}.jpg"
                if download(item["url"], dest):
                    meta_f.write(json.dumps({"file": str(dest.relative_to(ROOT)),
                                             "category": cat, **item}, ensure_ascii=False) + "\n")
                    n += 1
                time.sleep(0.5)  # 예의상 지연
        print(f"[{cat}] {n}장 확보 (목표 {target})")
    meta_f.close()


if __name__ == "__main__":
    main()
