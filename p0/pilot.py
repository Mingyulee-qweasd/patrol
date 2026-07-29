"""파일럿 — 프롬프트 문안 반복용. 소수 이미지 × 프롬프트 변형 × 생성설정을 돌려 판정 표 출력.

사용: python p0/pilot.py <이미지들...> [--prompt prompts/role_v0.txt] [--thinking 0]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import os
if os.environ.get("PATROL_BACKEND", "gemini") == "ollama":
    import client_ollama as client
else:
    import client


RESPONSE_SCHEMA = {  # 구조 강제 — 본 측정에서 파싱 실패 원천 차단 (동결 설정의 일부)
    "type": "OBJECT",
    "properties": {
        "is_task": {"type": "BOOLEAN"},
        "type": {"type": "STRING"},
        "n_robots": {"type": "STRING", "enum": ["0", "1", "2", "3", "beyond"]},
        "urgency": {"type": "INTEGER"},
        "confidence": {"type": "INTEGER"},
        "reason": {"type": "STRING"},
    },
    "required": ["is_task", "type", "n_robots", "urgency", "confidence", "reason"],
}


def judge(image_path: str, prompt_file: str, thinking_budget: int | None,
          temperature: float = 0.0) -> dict:
    prompt = Path(prompt_file).read_text()
    gc = {"temperature": temperature, "responseMimeType": "application/json",
          "responseSchema": RESPONSE_SCHEMA}
    if thinking_budget is not None:
        gc["thinkingConfig"] = {"thinkingBudget": thinking_budget}
    resp = client.call(prompt, image_path=image_path, gen_config=gc,
                       tag=f"pilot:{Path(prompt_file).stem}")
    out = client.extract_json(resp)
    out["_tokens"] = resp.get("usageMetadata", {}).get("totalTokenCount")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--prompt", default=str(Path(__file__).parent / "prompts/role_v0.txt"))
    ap.add_argument("--thinking", type=int, default=None,
                    help="thinkingBudget (0=끔). 미지정 = 모델 기본")
    args = ap.parse_args()

    print(f"{'이미지':40} {'task':5} {'n':>6} {'u':>2} {'conf':>4} {'tok':>5}  type/reason")
    for img in args.images:
        try:
            j = judge(img, args.prompt, args.thinking)
            print(f"{Path(img).name:40} {str(j.get('is_task')):5} "
                  f"{str(j.get('n_robots')):>6} {str(j.get('urgency')):>2} "
                  f"{str(j.get('confidence')):>4} {str(j.get('_tokens')):>5}  "
                  f"{j.get('type','')[:30]} | {j.get('reason','')[:50]}")
        except Exception as e:
            print(f"{Path(img).name:40} 오류: {e}")


if __name__ == "__main__":
    main()
