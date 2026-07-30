"""해석 검증 (analytic validation) — 시뮬을 닫힌형 이론값과 대조. 허용 오차 ±5% (문헌 관행).

설정: exp/configs/simple.yaml 사본(원본 무수정)을 lambda_calib=0으로 만들어 tmp에 두고
rho=0으로 Episode를 돌려 임무 발생을 0으로 — 순수 순찰만 남긴다.
(주의: sim/tasks.py의 n_haz = max(1, ...) 때문에 위험물 1개는 설정만으로 못 끔 →
 fixture에서 stream.tasks를 직접 비운다. findings 참조.)

──────────────────────────────────────────────────────────────────────────────
[이론 1 — 셀 방치시간 (idleness), Portugal & Rocha 계열 관행]

한 셀이 간격 τ_1, ..., τ_k 로 반복 방문되면 방치시간은 각 간격에서 0→τ_i 로
자라는 톱니파다. 한 사이클 적분 = Σ τ_i²/2 이므로

    시간평균 방치  Ī = (Σ_i τ_i²) / (2 Σ_i τ_i).                       (1)

균일 간격 τ면 Ī = τ/2 — "재방문 주기의 절반" 근사의 일반형이 (1)이다.

simple.yaml 노선은 왕복(비루프)이라 방문 간격이 균일하지 않다:
속도 v_h로 [0, L]을 왕복하는 허브(왕복주기 T = 2L/v_h)는 내부 지점 x를
주기마다 2회(양방향 1회씩), 서로 다른 간격으로 지난다. IdlenessMap 리셋은
로봇이 reliable_r 이내일 때이므로, 횡오프셋 t_off인 셀은 허브가 노선상
x ± w (w = √(reliable_r² − t_off²)) 구간에 있는 동안 방치 0으로 유지된다.
따라서 실제 "방치가 자라는" 두 간격은 (반경 이탈→재진입, 각각 2w/v_h 단축):

    τ_1(x) = 2(x − w)/v_h        (가까운 끝 0 갔다 오는 동안)
    τ_2(x) = 2(L − x − w)/v_h    (먼 끝 L 갔다 오는 동안)
    τ_1 + τ_2 + 4w/v_h = T

(1)에서 사이클 길이는 반경 내 체류(방치 0) 포함 T이므로 셀별 이론값은

    Ī(x) = (τ_1(x)² + τ_2(x)²) / (2T).                                 (2)

중앙 x = L/2 에서 Ī ≈ T/4 = (일방 통과시간 L/v_h)의 절반 — 지시서의
"주기의 절반" 근사. 끝 셀로 갈수록 T/2까지 커지는 불균일이 있으므로
(경계 셀 효과) 내부 셀만 집계하고, 근사 대신 정확식 (2)로 셀별 대조한다.
simple.yaml 수치: v_h = 1000/2558 ≈ 0.391 m/s, T = 5116 s, w = √60 ≈ 7.75 m.

[이론 2 — 세 로봇 나란히 전진 (hub_speed_sync)]

geometry.hub_speed_sync: v_h = L / (스윕 1패스 시간) = L·v_sweep / len_sweep.
→ 측면(flanker)이 톱니 스윕 1패스로 노선 0→L을 지나는 시간에 허브도 정확히
L을 진행 = 노선방향 평균 전진 속도 동일. 순간 편차는 측면이 횡단 톱니
(길이 W−2, 소요 (W−2)/v_sweep)에 머무는 동안 허브가 전진하는

    A = (W − 2)·v_h / v_sweep   (≈ 14.9 m, simple.yaml)

진폭의 지그재그로 유계. 좌/우 측면은 대칭이라 s가 항상 같으므로 세 s의
모표준편차는 std({s_h, s_f, s_f}) = |s_h − s_f|·√2/3 ≤ 0.47·A (정상 순찰),
랑데뷰 직후 재동기 과도까지 감안해도 ~A 수준 ≪ L. 검증 문턱: max ≤ 1.5A.

[이론 3 — 임무 0이면 이벤트는 랑데뷰뿐]

임무·비임무 개체가 0이면 감지→판단 경로가 전혀 발화하지 않으므로 trace에는
주기적 rendezvous 이벤트만 남아야 한다. 안건 0이면 policy.next_interval이
V=0 분기로 간격을 1.5·d0로 연장하므로 랑데뷰 사이 간격은 1.5·d0(+집결 지연).
──────────────────────────────────────────────────────────────────────────────
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sim.env import load_env                    # noqa: E402
from sim.geometry import path_length_world      # noqa: E402
from sim.metrics import compute                 # noqa: E402
from sim.runner import Episode                  # noqa: E402

SEED = 1
REL_TOL = 0.05          # 지시서 허용 오차 ±5%
INTERIOR_MARGIN = 50.0  # 내부 셀 집계: 양끝 50 m 제외 (경계 셀 효과 회피)


# ── 공통 준비 ────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def cfg_path(tmp_path_factory):
    """simple.yaml 사본 (원본 무수정) — lambda_calib=0으로 임무 발생 차단."""
    cfg = yaml.safe_load((REPO / "exp/configs/simple.yaml").read_text())
    cfg["tasks"]["lambda_calib"] = 0.0
    p = tmp_path_factory.mktemp("cfg") / "simple_notask.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return str(p)


def _zero_task_episode(cfg_path: str, horizon_s: float) -> Episode:
    # λ=0이면 tasks.py의 1.0/lam이 0나눗셈 경고를 내므로 억제 (findings 참조)
    with np.errstate(divide="ignore"):
        ep = Episode(cfg_path, seed=SEED, horizon_s=horizon_s, rho=0.0)
    # n_haz = max(1, ...) 가 위험물 1개를 강제 생성 — 순수 순찰을 위해 직접 제거
    ep.stream.tasks.clear()
    ep.tasks_by_id.clear()
    return ep


@pytest.fixture(scope="module")
def hub_run(cfg_path):
    """허브 단독 에피소드: flanker를 robots에서 제거해 '왕복 단일 로봇'만 남기고,
    IdlenessMap.update를 감싸(스파이) 내부 허브행 셀의 시간평균 방치를 측정.

    측정 창: 워밍업 1왕복주기(모든 셀 최소 1회 방문 후) + 정확히 2왕복주기.
    """
    env = load_env(cfg_path)
    T = 2 * env.route.length / env.hub_v
    t0 = int(np.ceil(T))
    n_meas = 2 * int(np.ceil(T))
    ep = _zero_task_episode(cfg_path, horizon_s=t0 + n_meas + 60)
    del ep.robots[1]
    del ep.robots[2]  # 허브만 남김 (sim 코드 무수정 — 테스트 목적의 상태 축소)

    imap = ep.idle_map
    cells = imap.cells
    # 허브행 셀 = 노선변 t=±2 (reliable_r=8 이내라 허브가 리셋 가능한 유일한 행)
    sel = (np.isclose(np.abs(cells[:, 1]), 2.0)
           & (cells[:, 0] >= INTERIOR_MARGIN)
           & (cells[:, 0] <= env.route.length - INTERIOR_MARGIN))
    acc = np.zeros(int(sel.sum()))
    cnt = [0]
    orig_update = imap.update

    def spy(robot_xys, now):
        orig_update(robot_xys, now)
        if t0 <= now < t0 + n_meas:
            acc[:] += now - imap.last_seen[sel]
            cnt[0] += 1

    imap.update = spy
    ep.run()
    assert cnt[0] == n_meas, "측정 창이 지평선 안에 다 들어가야 함"
    return ep, cells[sel], acc / cnt[0]


@pytest.fixture(scope="module")
def three_run(cfg_path):
    """세 로봇 전체, 임무 0 에피소드 (기본 지평선 3h)."""
    ep = _zero_task_episode(cfg_path, horizon_s=10800)
    ep.run()
    return ep


def _theory_idleness(env, cells_xy: np.ndarray) -> np.ndarray:
    """식 (2): 왕복 허브의 셀별 시간평균 방치 이론값."""
    v_h, L = env.hub_v, env.route.length
    T = 2 * L / v_h
    w = np.sqrt(env.reliable_r ** 2 - cells_xy[:, 1] ** 2)
    x = cells_xy[:, 0]
    tau1 = 2 * (x - w) / v_h
    tau2 = 2 * (L - x - w) / v_h
    return (tau1 ** 2 + tau2 ** 2) / (2 * T)


# ── 1) 허브 방치시간 = 이론값 ────────────────────────────────────────────────
def test_hub_speed_sync_identity(cfg_path):
    """전제 확인: v_h · (스윕 1패스 시간) = L (hub_speed_sync 정의 그대로)."""
    env = load_env(cfg_path)
    lens = {s: path_length_world(env.route, wp) for s, wp in env.sweep_paths.items()}
    t_pass = max(lens.values()) / env.v_sweep
    assert env.hub_v * t_pass == pytest.approx(env.route.length, rel=1e-9)
    # simple.yaml은 좌우 대칭 — 두 스윕 길이 동일 (이론 2의 s_f 좌우 일치 전제)
    assert lens["left"] == pytest.approx(lens["right"], rel=1e-9)


def test_hub_idleness_mean_matches_theory(hub_run):
    """내부 허브행 셀 평균 방치: 시뮬 vs 식 (2) 평균, ±5%."""
    ep, cells, sim_avg = hub_run
    th = _theory_idleness(ep.env, cells)
    assert sim_avg.mean() == pytest.approx(th.mean(), rel=REL_TOL), (
        f"sim {sim_avg.mean():.1f}s vs theory {th.mean():.1f}s")


def test_hub_idleness_per_cell_matches_theory(hub_run):
    """셀별로도 식 (2)와 ±5% — 왕복 불균일(끝 근처↑, 중앙↓)까지 재현되는지."""
    ep, cells, sim_avg = hub_run
    th = _theory_idleness(ep.env, cells)
    rel = np.abs(sim_avg - th) / th
    assert float(rel.max()) < REL_TOL, (
        f"worst cell rel err {rel.max():.3f} at x={cells[int(rel.argmax()), 0]:.0f}")
    # 불균일 방향 확인: 중앙(≈T/4)이 가장자리 집계분보다 확실히 낮음
    mid = np.abs(cells[:, 0] - ep.env.route.length / 2) < 100
    assert sim_avg[mid].mean() < sim_avg[~mid].mean()


# ── 2) 세 로봇 나란히 전진 ───────────────────────────────────────────────────
def test_three_robots_advance_abreast(three_run):
    """정상상태에서 세 로봇 s(직선 노선이라 s=x)의 표준편차가 지그재그 진폭
    A = (W−2)·v_h/v_sweep 수준으로 유계 (이론 2)."""
    ep = three_run
    env = ep.env
    A = (max(env.corridor_width.values()) - 2.0) * env.hub_v / env.v_sweep
    stds = np.array([
        np.std([xyz[0] for xyz in snap["robots"].values()])
        for snap in ep.snaps
        if all(xyz[2] == "patrol" for xyz in snap["robots"].values())
    ])
    assert len(stds) > 100, "순찰 상태 표본이 충분해야 함"
    assert float(stds.max()) <= 1.5 * A, (
        f"max std {stds.max():.1f} m > 1.5A={1.5 * A:.1f} m")
    assert float(np.median(stds)) <= 0.5 * A       # 평시엔 0.47A 이하 (이론 2)
    assert float(stds.max()) <= 0.025 * env.route.length  # 노선 대비로도 '작음'(≤2.5%)


# ── 3) 임무 0 → 랑데뷰 외 이벤트 없음 ────────────────────────────────────────
def test_zero_mission_only_rendezvous_events(three_run):
    """dispatch/오개입/발견/안건/소집 등 임무 계열 이벤트가 하나도 없어야 함."""
    ep = three_run
    kinds = {e["e"] for e in ep.trace.events}
    assert kinds <= {"rendezvous"}, f"임무 0인데 부적절 이벤트: {kinds - {'rendezvous'}}"
    m = compute(ep)
    assert m.n_tasks == 0 and m.n_completed == 0
    assert m.n_dispatch_now == 0 and m.n_assign == 0
    assert m.n_misintervention_minor == 0 and m.n_misintervention_severe == 0
    assert m.n_bounce == 0 and m.n_convoke == 0 and m.n_abort == 0


def test_rendezvous_periodic(three_run):
    """랑데뷰는 주기적으로 발생 (정상). 안건 0 → 간격은 1.5·d0로 연장(이론 3),
    실측 간격 = 1.5·d0 + 집결 지연(수십 초)."""
    ep = three_run
    rdv = [e for e in ep.trace.events if e["e"] == "rendezvous"]
    assert len(rdv) >= 2, "지평선 3h 안에 랑데뷰가 반복돼야 함"
    expect = 1.5 * ep.d0_s
    for e in rdv:
        assert e["n_agenda"] == 0
        assert e["next_in"] == pytest.approx(expect)
    gaps = np.diff([e["t"] for e in rdv])
    assert np.all(gaps >= expect - 1.0)
    assert np.all(gaps <= expect + 150.0)  # 집결(이동) 지연 여유
