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
    """오류 = 개체 고정 성분(맹점) + 관측 흔들림 성분으로 분해 재생.

    맹점: 실측 문안 만장일치-오답률(p_blind)만큼의 개체는 몇 번을 봐도 같은 오판을 반환
    (개체 성질이므로 world_seed+개체 id에서 유도 — 전 로봇이 같은 맹점을 공유).
    흔들림: 나머지 오답 질량. 총 오답률은 실측과 동일하게 유지 (구조만 재배치).
    oid=None이면 종전과 동일한 순수 흔들림 동작 (개발·테스트 호환).
    """

    def __init__(self, model: dict, seed: int, world_seed: int | None = None):
        self.model = model
        self.rng = np.random.default_rng(seed)
        self.world_seed = world_seed
        self._blind: dict = {}   # (oid, band) → None(보통 개체) | 고정 오판 dict

    def _blind_state(self, band: str, oid, cell: dict):
        key = (oid, band)
        if key in self._blind:
            return self._blind[key]
        state = None
        p_blind = cell.get("p_blind", 0.0)
        wrong = [j for j in cell["judgments"] if j.get("wrong")]
        if p_blind > 0 and wrong:
            # 주의: 파이썬 hash()는 문자열에 프로세스별 무작위 시드 → 재현성·짝 비교 파괴 (#27)
            band_code = 0 if band == "near" else 1
            rng = np.random.default_rng(
                (int(self.world_seed) * 1_000_003 + int(oid) * 2 + band_code) & 0x7FFFFFFF)
            if rng.random() < p_blind:
                ps = np.array([j["p"] for j in wrong], float)
                ps /= ps.sum()
                state = wrong[int(rng.choice(len(wrong), p=ps))]
        self._blind[key] = state
        return state

    def observe(self, gt_kind: str, band: str, oid=None) -> dict | None:
        """관측 이벤트 → 판정 or None(미탐). 반환: {is_task, n̂, û, conf(0~100)}."""
        cell = self.model[gt_kind][band]
        if self.rng.random() > cell["p_detect"]:
            return None
        blind = (self._blind_state(band, oid, cell)
                 if oid is not None and self.world_seed is not None else None)
        if blind is not None:
            j = blind
        else:
            js = cell["judgments"]
            ps = np.array([x["p"] for x in js], float)
            p_blind = cell.get("p_blind", 0.0) if oid is not None and self.world_seed is not None else 0.0
            if p_blind > 0:
                # 보통 개체의 오답 질량 w_n: p_blind + (1-p_blind)·w_n = 실측 총오답률 이 되도록
                wmask = np.array([bool(x.get("wrong")) for x in js])
                tw = float(ps[wmask].sum())
                if tw > 0 and wmask.any() and (~wmask).any():
                    w_n = max(tw - p_blind, 0.0) / (1.0 - p_blind)
                    ps[wmask] *= w_n / tw
                    rsum = float(ps[~wmask].sum())
                    if rsum > 0:
                        ps[~wmask] *= (1.0 - w_n) / rsum
            ps /= ps.sum()
            j = js[int(self.rng.choice(len(ps), p=ps))]
        conf = float(np.clip(self.rng.normal(j["conf_mu"], j["conf_sd"]), 5, 100))
        # 실측 판정의 "beyond"(=4)는 전체 대수(3)로 상한 — 과대 추정은 도착/랑데뷰에서 자가 교정
        return {"is_task": j["is_task"], "n_hat": min(j["n"], 3), "u_hat": j["u"],
                "conf": conf}
