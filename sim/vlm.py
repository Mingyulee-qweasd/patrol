"""VLM 판정 샘플러 — P0 오류 모델(error_model.json)에서 추첨.

P0 완료 전에는 합성 모델(synthetic_model)로 전 파이프라인을 개발·테스트하고,
error_model.json이 생기면 load_model()로 교체 (스키마 동일 — 코드 무변경).

스키마: model[gt_kind][band] = {
    "p_detect": 관측 이벤트에서 뭔가 알아챌 확률 (미탐 = 판정 자체 없음),
    "judgments": [ {"is_task": bool, "n": int, "u": int, "conf_mu": .., "conf_sd": .., "p": ..}, ... ]
}
band ∈ {"far", "near"} — reliable_r 이내면 near.
"""
import json
from pathlib import Path

import numpy as np


def synthetic_model(kinds: dict) -> dict:
    """온톨로지에서 그럴듯한 임시 오류 모델 생성 (파일럿 관찰 반영한 수준).

    kinds: {kind: {"n":, "u":, "gt_class":}} — task 타입 + ambiguous/hazard 포함.
    """
    m = {}
    for kind, spec in kinds.items():
        gt_task = spec["gt_class"] == "task"
        n, u = spec.get("n", 0), spec.get("u", 0)
        far_correct = 0.75 if gt_task else 0.80   # 원거리 정판정률
        near_correct = 0.92 if gt_task else 0.93  # 근거리
        m[kind] = {}
        for band, pc in [("far", far_correct), ("near", near_correct)]:
            J = []
            if gt_task:
                J.append({"is_task": True, "n": n, "u": u,
                          "conf_mu": 88 if band == "near" else 74,
                          "conf_sd": 8, "p": pc})
                # 사이징 오차 (파일럿: 대형을 과소, 경계를 혼동)
                n_err = max(1, n - 1) if n > 1 else min(3, n + 1)
                J.append({"is_task": True, "n": n_err, "u": u,
                          "conf_mu": 78, "conf_sd": 10, "p": (1 - pc) * 0.6})
                J.append({"is_task": False, "n": 0, "u": 0,
                          "conf_mu": 70, "conf_sd": 12, "p": (1 - pc) * 0.4})
            else:
                J.append({"is_task": False, "n": 0, "u": 0,
                          "conf_mu": 90 if band == "near" else 78,
                          "conf_sd": 8, "p": pc})
                # 오검(비task를 task로) — hazard의 경우 이게 '접근 유발'이라 중대
                J.append({"is_task": True, "n": 1, "u": max(1, u),
                          "conf_mu": 72, "conf_sd": 12, "p": 1 - pc})
            m[kind][band] = {"p_detect": 0.85 if band == "far" else 0.98,
                             "judgments": J}
    return m


def load_model(path: str | Path) -> dict:
    d = json.loads(Path(path).read_text())
    # P0 산출물(error_model.json)은 {"stats", "model", "n_rows"} 래퍼 — 샘플러 스키마만 꺼냄
    return d["model"] if "model" in d else d


class VLMSampler:
    def __init__(self, model: dict, seed: int):
        self.model = model
        self.rng = np.random.default_rng(seed)

    def observe(self, gt_kind: str, band: str) -> dict | None:
        """관측 이벤트 → 판정 or None(미탐). 반환: {is_task, n̂, û, conf(0~100)}."""
        cell = self.model[gt_kind][band]
        if self.rng.random() > cell["p_detect"]:
            return None
        ps = np.array([j["p"] for j in cell["judgments"]], float)
        ps /= ps.sum()
        j = cell["judgments"][int(self.rng.choice(len(ps), p=ps))]
        conf = float(np.clip(self.rng.normal(j["conf_mu"], j["conf_sd"]), 5, 100))
        # 실측 판정의 "beyond"(=4)는 전체 대수(3)로 상한 — 과대 추정은 도착/랑데뷰에서 자가 교정
        return {"is_task": j["is_task"], "n_hat": min(j["n"], 3), "u_hat": j["u"],
                "conf": conf}
