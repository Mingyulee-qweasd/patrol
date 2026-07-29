"""로컬 Ollama(Qwen3-VL) 백엔드 — client.py와 동일 인터페이스·캐시 규율.

Gemini 백엔드와의 A/B용. 채택 시 이 모델이 '동결 대상'이 된다 (가중치 해시 고정).
"""
import base64
import hashlib
import json
import time
from pathlib import Path

import requests

MODEL = "qwen3-vl:8b"
HOST = "http://127.0.0.1:11434"
ROOT = Path(__file__).parent
CACHE_DIR = ROOT / "out" / "cache"
CALLS_LOG = ROOT / "out" / "calls.jsonl"


def _cache_key(image_bytes: bytes | None, prompt: str, gen_config: dict) -> str:
    h = hashlib.sha256()
    h.update(image_bytes or b"")
    h.update(prompt.encode())
    h.update(MODEL.encode())
    h.update(json.dumps(gen_config, sort_keys=True).encode())
    return h.hexdigest()


def _log_call(rec: dict) -> None:
    CALLS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(CALLS_LOG, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def call(prompt: str, image_path: str | Path | None = None,
         gen_config: dict | None = None, tag: str = "") -> dict:
    gen_config = gen_config or {}
    image_bytes = Path(image_path).read_bytes() if image_path else None
    ck = _cache_key(image_bytes, prompt, gen_config)
    cache_file = CACHE_DIR / f"{ck}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    msg = {"role": "user", "content": prompt}
    if image_bytes is not None:
        msg["images"] = [base64.b64encode(image_bytes).decode()]
    body = {
        "model": MODEL,
        "messages": [msg],
        "stream": False,
        "think": False,  # thinking이 답변 토큰을 소진해 빈 본문을 만드는 버그 회피 (동결 설정의 일부)
        "options": {"temperature": gen_config.get("temperature", 0.0)},
    }
    if "responseSchema" in gen_config:
        body["format"] = _to_ollama_schema(gen_config["responseSchema"])

    t0 = time.time()
    resp = None
    for attempt in range(3):
        b = dict(body)
        if attempt == 1:
            b = {k: v for k, v in body.items() if k != "format"}  # 스키마 없이 재시도
        elif attempt == 2:
            b["options"] = dict(body.get("options", {}), temperature=0.3)
        r = requests.post(f"{HOST}/api/chat", json=b, timeout=600)
        if r.status_code != 200:
            _log_call({"ts": time.time(), "backend": "ollama", "tag": tag,
                       "status": r.status_code, "error": r.text[:200]})
            raise RuntimeError(f"Ollama 호출 실패 HTTP {r.status_code}: {r.text[:200]}")
        resp = r.json()
        if resp.get("message", {}).get("content", "").strip():
            if attempt:
                _log_call({"ts": time.time(), "backend": "ollama", "tag": tag,
                           "status": 200, "note": f"빈 응답 재시도 {attempt}회로 회복"})
            break
    if not resp.get("message", {}).get("content", "").strip():
        raise RuntimeError("Ollama 빈 응답 3회 — 모델/서버 점검 필요")
    resp["_latency_s"] = round(time.time() - t0, 2)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(resp, ensure_ascii=False))
    _log_call({"ts": time.time(), "backend": "ollama", "tag": tag, "status": 200,
               "cache_key": ck, "image": str(image_path) if image_path else None,
               "latency_s": resp["_latency_s"],
               "usage": {"prompt": resp.get("prompt_eval_count"),
                          "out": resp.get("eval_count")}})
    return resp


def _to_ollama_schema(gemini_schema: dict) -> dict:
    """Gemini responseSchema(대문자 타입) → 표준 JSON Schema(소문자)."""
    def conv(node):
        if isinstance(node, dict):
            out = {}
            for k, v in node.items():
                if k == "type" and isinstance(v, str):
                    out[k] = v.lower()
                else:
                    out[k] = conv(v)
            return out
        if isinstance(node, list):
            return [conv(x) for x in node]
        return node
    return conv(gemini_schema)


def extract_text(resp: dict) -> str:
    return resp["message"]["content"]


def extract_json(resp: dict) -> dict:
    txt = extract_text(resp).strip()
    start = txt.find("{")
    if start == -1:
        raise ValueError(f"JSON 미발견: {txt[:120]}")
    obj, _ = json.JSONDecoder().raw_decode(txt[start:])
    return obj
