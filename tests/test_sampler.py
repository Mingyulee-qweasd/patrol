"""tests/test_sampler.py — sim/vlm.py VLMSampler 검증.

검증 항목:
1) p0/out/error_model.json 로드 후 각 (카테고리, 밴드) 칸에서 20,000회 observe →
   미탐률 ≈ 1 - p_detect, 판정 조합 경험 빈도 ≈ judgments p
   (카이제곱 p > 0.01 또는 최대 이탈 < 0.02 이면 합격)
2) 같은 seed → 같은 추첨 열(재현성), 다른 seed → 다른 열
3) n_hat 상한 3 — 실측 "beyond"(n=4) 판정이 있는 칸에서 3으로 클립되는지
4) conf 범위 [5, 100] 클립
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sim.vlm import VLMSampler, load_model  # noqa: E402

MODEL_PATH = REPO / "p0" / "out" / "error_model.json"
MODEL = load_model(MODEL_PATH)
CELLS = [(k, b) for k in sorted(MODEL) for b in sorted(MODEL[k])]
N_DRAWS = 20_000

_draw_cache: dict[tuple[str, str], list] = {}


def draw_cell(kind: str, band: str) -> list:
    """한 칸에서 N_DRAWS회 observe (칸별 고정 seed, 결과 캐시)."""
    key = (kind, band)
    if key not in _draw_cache:
        seed = 20260730 + CELLS.index(key)
        s = VLMSampler(MODEL, seed)
        _draw_cache[key] = [s.observe(kind, band) for _ in range(N_DRAWS)]
    return _draw_cache[key]


def freq_ok(obs_counts, exp_probs, n_total):
    """합격 기준: 카이제곱 적합도 p > 0.01 또는 최대 빈도 이탈 < 0.02."""
    obs = np.asarray(obs_counts, float)
    p = np.asarray(exp_probs, float)
    exp = p * n_total
    max_dev = float(np.max(np.abs(obs / n_total - p)))
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    pval = float(stats.chi2.sf(chi2, df=len(obs) - 1))
    return (pval > 0.01 or max_dev < 0.02), pval, max_dev


def clipped_key(j: dict) -> tuple:
    """judgment → observe 출력에서 관측되는 키 (n은 3으로 클립됨)."""
    return (bool(j["is_task"]), min(j["n"], 3), j["u"])


# ---------------------------------------------------------------- 1) 통계 일치

@pytest.mark.parametrize("kind,band", CELLS, ids=[f"{k}-{b}" for k, b in CELLS])
def test_cell_statistics(kind, band):
    cell = MODEL[kind][band]
    samples = draw_cell(kind, band)

    # 1a. 미탐률 ↔ 1 - p_detect
    n_miss = sum(1 for s in samples if s is None)
    p_miss = 1.0 - cell["p_detect"]
    ok, pval, dev = freq_ok(
        [n_miss, N_DRAWS - n_miss], [p_miss, 1.0 - p_miss], N_DRAWS)
    assert ok, (f"{kind}/{band} 미탐률 {n_miss / N_DRAWS:.4f} vs 기대 "
                f"{p_miss:.4f} (chi2 p={pval:.3g}, 이탈 {dev:.4f})")

    # 1b. 판정 조합 빈도 ↔ judgments p (n=4는 3으로 클립돼 키가 합쳐질 수 있음)
    ps = np.array([j["p"] for j in cell["judgments"]], float)
    ps = ps / ps.sum()  # 샘플러와 동일하게 정규화 (파일의 p 합은 0.9999 등)
    exp: dict[tuple, float] = {}
    for j, p in zip(cell["judgments"], ps):
        exp[clipped_key(j)] = exp.get(clipped_key(j), 0.0) + float(p)

    detected = [s for s in samples if s is not None]
    obs = Counter((s["is_task"], s["n_hat"], s["u_hat"]) for s in detected)
    assert set(obs) <= set(exp), \
        f"{kind}/{band} 기대 밖 판정 키 관측: {sorted(set(obs) - set(exp))}"

    keys = sorted(exp)
    ok, pval, dev = freq_ok([obs.get(k, 0) for k in keys],
                            [exp[k] for k in keys], len(detected))
    assert ok, (f"{kind}/{band} 판정 빈도 불일치 (chi2 p={pval:.3g}, "
                f"최대 이탈 {dev:.4f})\n키별 관측/기대: "
                + str({k: (obs.get(k, 0) / len(detected), round(exp[k], 4))
                       for k in keys}))


# ---------------------------------------------------------------- 2) 재현성

def _sequence(seed: int, n: int = 360) -> list:
    """모든 칸을 돌아가며 n회 observe한 결과 열."""
    s = VLMSampler(MODEL, seed)
    return [s.observe(*CELLS[i % len(CELLS)]) for i in range(n)]


def test_same_seed_same_sequence():
    a, b = _sequence(7), _sequence(7)
    assert a == b, "같은 seed인데 추첨 열이 다름 (conf 부동소수까지 일치해야 함)"


def test_different_seed_different_sequence():
    a, b = _sequence(7), _sequence(8)
    assert a != b, "다른 seed인데 추첨 열이 완전히 동일함"


# ---------------------------------------------------------------- 3) n_hat 상한 3

def test_error_model_has_beyond_cells():
    """전제: 실측 error_model에 n=4('beyond') 판정 칸이 실제로 존재."""
    beyond = [(k, b) for k, b in CELLS
              if any(j["n"] >= 4 for j in MODEL[k][b]["judgments"])]
    assert beyond, "error_model.json에 n>=4 판정이 있는 칸이 없음 — 전제 불성립"


def test_nhat_clipped_to_3():
    # (a) 모든 칸의 모든 관측에서 n_hat <= 3
    for kind, band in CELLS:
        for s in draw_cell(kind, band):
            if s is not None:
                assert s["n_hat"] <= 3, \
                    f"{kind}/{band}에서 n_hat={s['n_hat']} > 3 관측"

    # (b) n=4 판정이 실제로 추첨됐음을 증명: obstacle/near의 (True,n=4,u=3)은
    #     클립 후 (True,3,3)이 되고, 그 키로 가는 다른 judgment가 없어 유일 식별 가능.
    cell = MODEL["obstacle"]["near"]
    sources = [j for j in cell["judgments"] if clipped_key(j) == (True, 3, 3)]
    assert sources == [j for j in cell["judgments"] if j["n"] == 4 and j["u"] == 3], \
        "obstacle/near의 (True,3,3) 키가 n=4 판정만의 것이 아님 — 식별 전제 깨짐"
    seen = Counter((s["is_task"], s["n_hat"], s["u_hat"])
                   for s in draw_cell("obstacle", "near") if s is not None)
    assert seen[(True, 3, 3)] > 0, \
        "20,000회에서 n=4 판정(기대 ~0.9%)이 한 번도 안 나옴 — 클립 실증 실패"

    # (c) 실측 n=4 judgment만 남긴 부분 모델로 직접 확인 (추첨 보장)
    j4 = next(j for k, b in CELLS for j in MODEL[k][b]["judgments"] if j["n"] >= 4)
    sub = {"x": {"near": {"p_detect": 1.0, "judgments": [{**j4, "p": 1.0}]}}}
    s = VLMSampler(sub, 0)
    for _ in range(50):
        r = s.observe("x", "near")
        assert r is not None and r["n_hat"] == 3, \
            f"n={j4['n']} 판정이 n_hat={r and r['n_hat']}로 반환됨 (3이어야 함)"


# ---------------------------------------------------------------- 4) conf 클립

def test_conf_within_5_100_on_real_model():
    for kind, band in CELLS:
        for s in draw_cell(kind, band):
            if s is not None:
                assert 5.0 <= s["conf"] <= 100.0, \
                    f"{kind}/{band} conf={s['conf']} 범위 [5,100] 밖"


def test_conf_clips_at_both_bounds():
    def extreme(mu):
        m = {"x": {"far": {"p_detect": 1.0, "judgments": [
            {"is_task": True, "n": 1, "u": 1,
             "conf_mu": mu, "conf_sd": 1.0, "p": 1.0}]}}}
        s = VLMSampler(m, 0)
        return [s.observe("x", "far")["conf"] for _ in range(20)]

    assert all(c == 5.0 for c in extreme(-500.0)), "하한 5 클립 실패"
    assert all(c == 100.0 for c in extreme(500.0)), "상한 100 클립 실패"
