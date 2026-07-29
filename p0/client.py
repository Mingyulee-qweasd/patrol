"""P0 Gemini 호출 클라이언트 — 동결 프로토콜의 실행부.

규율 (설계 문서 §4):
  - 키 값은 어떤 로그·예외·출력에도 싣지 않는다 (존재 여부만).
  - 캐시 히트 시 API 호출 0회 — 같은 (이미지, 프롬프트, 모델, 생성설정)은 항상 같은 응답.
  - 모든 호출은 calls.jsonl에 기록 (재현성·한도 추적).
"""
import base64
import hashlib
import json
import time
from pathlib import Path

import requests

MODEL = "gemini-3.5-flash"  # 고정 (버전 3.5-flash-05-2026) — 변경 = P0 재실행
KEY_PATH = Path.home() / ".config/gemini/api_key"
ROOT = Path(__file__).parent
CACHE_DIR = ROOT / "out" / "cache"
CALLS_LOG = ROOT / "out" / "calls.jsonl"
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

MAX_RETRIES = 6
BACKOFF_BASE_S = 5.0


def _load_key() -> str:
    if not KEY_PATH.exists():
        raise FileNotFoundError(f"API 키 파일 없음: {KEY_PATH}")
    key = KEY_PATH.read_text().strip()
    if len(key) < 20:
        raise ValueError("API 키 형식 이상 (길이 부족)")  # 값은 출력하지 않는다
    return key


def _cache_key(image_bytes: bytes | None, prompt: str, gen_config: dict) -> str:
    h = hashlib.sha256()
    h.update(image_bytes or b"")
    h.update(prompt.encode())
    h.update(MODEL.encode())
    h.update(json.dumps(gen_config, sort_keys=True).encode())
    return h.hexdigest()


def _log_call(record: dict) -> None:
    CALLS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(CALLS_LOG, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def call(prompt: str, image_path: str | Path | None = None,
         gen_config: dict | None = None, tag: str = "") -> dict:
    """단일 호출. 반환: API 응답 전체(dict). 캐시 히트 시 네트워크 접근 없음."""
    gen_config = gen_config or {}
    image_bytes = Path(image_path).read_bytes() if image_path else None
    ck = _cache_key(image_bytes, prompt, gen_config)
    cache_file = CACHE_DIR / f"{ck}.json"

    if cache_file.exists():
        return json.loads(cache_file.read_text())

    parts = []
    if image_bytes is not None:
        mime = "image/png" if str(image_path).lower().endswith(".png") else "image/jpeg"
        parts.append({"inline_data": {"mime_type": mime,
                                      "data": base64.b64encode(image_bytes).decode()}})
    parts.append({"text": prompt})
    body = {"contents": [{"parts": parts}]}
    if gen_config:
        body["generationConfig"] = gen_config

    key = _load_key()
    url = ENDPOINT.format(model=MODEL) + f"?key={key}"

    last_status = None
    for attempt in range(MAX_RETRIES):
        r = requests.post(url, json=body, timeout=120)
        last_status = r.status_code
        if r.status_code == 200:
            resp = r.json()
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(resp, ensure_ascii=False))
            _log_call({"ts": time.time(), "cache_key": ck, "tag": tag,
                       "image": str(image_path) if image_path else None,
                       "status": 200, "attempt": attempt,
                       "usage": resp.get("usageMetadata", {})})
            return resp
        if r.status_code in (429, 500, 502, 503):
            wait = BACKOFF_BASE_S * (2 ** attempt)
            _log_call({"ts": time.time(), "cache_key": ck, "tag": tag,
                       "status": r.status_code, "attempt": attempt, "retry_in_s": wait})
            time.sleep(wait)
            continue
        # 그 외 오류: 본문에 키가 없음이 확인된 표준 오류 메시지만 기록
        try:
            err = r.json().get("error", {}).get("message", "")[:200]
        except Exception:
            err = f"(본문 파싱 실패, HTTP {r.status_code})"
        _log_call({"ts": time.time(), "cache_key": ck, "tag": tag,
                   "status": r.status_code, "error": err})
        raise RuntimeError(f"Gemini 호출 실패 HTTP {r.status_code}: {err}")
    raise RuntimeError(f"재시도 {MAX_RETRIES}회 소진 (마지막 HTTP {last_status}) — 한도 문제면 유료 전환 검토")


def extract_text(resp: dict) -> str:
    return resp["candidates"][0]["content"]["parts"][0]["text"]


def extract_json(resp: dict) -> dict:
    """모델 출력에서 JSON 오브젝트 파싱 (```json 펜스·전후 잡음 허용)."""
    txt = extract_text(resp).strip()
    if txt.startswith("```"):
        txt = txt.split("```")[1]
        txt = txt[4:] if txt.startswith("json") else txt
    start, end = txt.find("{"), txt.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"JSON 미발견: {txt[:120]}")
    return json.loads(txt[start:end + 1])


if __name__ == "__main__":
    # 스모크: 텍스트 1회 (JSON 강제 모드 확인)
    resp = call('Return JSON exactly: {"ok": true}',
                gen_config={"temperature": 0, "responseMimeType": "application/json"},
                tag="smoke")
    print("스모크:", extract_json(resp), "| usage:", resp.get("usageMetadata", {}).get("totalTokenCount"))
