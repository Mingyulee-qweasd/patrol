# patrol 구현 노트

> 설계 문서(`~/Documents/순찰연구_설계.md`)와 분업: 설계 = 무엇을·왜 / **이 문서 = 어떻게·세부**.
> 구현 변경·버그·기술 결정이 생길 때마다 갱신 (설계 결정은 설계 문서 §9로).
> 시작 2026-07-29. 마지막 갱신: 2026-07-29 (프롬프트 동결 p0-freeze-v1, 본 측정 가동).

## 0. 환경·실행

- conda env **patrol** (python 3.11): numpy, matplotlib, pyyaml, shapely, scipy, pytest, pillow, requests
- **모든 실행 전**: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate patrol && unset PYTHONPATH`
  (`.bashrc`의 ROS Jazzy가 py3.12 site-packages를 주입하는 함정 — 기존 프로젝트에서 실측된 표준 처방)
- 로컬 VLM: **Ollama v0.32.5** (`~/.local/ollama/bin/ollama`, sudo 없이 tar.zst 설치)
  - 서버: `nohup ~/.local/ollama/bin/ollama serve > ~/.local/ollama/serve.log 2>&1 &` (127.0.0.1:11434)
  - 모델: **qwen3-vl:8b** (digest 901cae732162, 6.1GB, VRAM ~8.1GB 상주)
- git: 로컬 repo만 (원격 push는 사용자 결정 대기). 동결 태그 **p0-freeze-v1**

## 1. P0 (VLM 판정 측정) — `p0/`

### client.py — Gemini 백엔드 (부록 대조용으로만 잔류)
- 모델 `gemini-3.5-flash` 고정. 키 `~/.config/gemini/api_key`(600) — **값은 어떤 로그·예외에도 비출력**
- 캐시: sha256(이미지바이트+프롬프트+모델+gen_config) → `out/cache/{hash}.json` — 히트 시 API 호출 0
- 재시도: 429/5xx 지수 백오프(5s×2^n, 6회). **실측: 무료 등급 일일 25회 제한** → 본 측정 불가로 로컬 전환
- JSON 파싱: `raw_decode`로 첫 완결 오브젝트만 (뒤 잡음 무시)

### client_ollama.py — 주 백엔드 (동결 대상)
- 동일 캐시·로그 규율. 캐시 키에 **CONFIG_TAG**(`qwen3vl8b-901cae-thinkoff-frozen_v1`) 포함
  — 설정 변경 시 구 캐시 자연 무효화 (본 측정 균일 조건 보장)
- **think=false 필수**: qwen3-vl thinking이 내부 사고에 1만+자 소비 → 답변 토큰 소진 → content 빈 채 반환
  (hazard_ov040에서 실측: think=true → content 0자/thinking 10,749자; false → 정상 249자)
- 빈 응답 3단 회복 사다리: ①원설정 재시도 → ②format(스키마) 제거 재시도 → ③temperature 0.3 재시도.
  실패분은 **캐시에 저장 안 함** (초기에 빈 응답이 캐시를 오염시킨 사고 → 검증 후 저장으로 수정, 오염분 2건 수동 삭제)
- responseSchema: Gemini 대문자 타입 스키마 → 소문자 JSON Schema 변환(`_to_ollama_schema`) 후 `format` 필드로
- 사용 기록: `out/calls.jsonl` (지연·토큰·재시도 회복 이력 — 본 측정 후 재시도율 보고 가능)

### 프롬프트 — `prompts/` (동결: frozen_v1)
- 구조: 역할+헌법("확신 없으면 기록만") → **능력 문단** → 판단 5개 → 비개입 목록 → JSON 지시
- 능력 문단 (정제판): "무릎 높이 바퀴형·집게 팔·혼자 15kg 들기/끌기·둘이면 더 무거운 것·셋이면
  아주 무거운 것 천천히·등반/도구 불가" — **물체 예시 금지** (1차 시도에서 '작은 물건을 바구니에' 예시가
  소형 쓰레기로 닻내림 → 소파를 litter n=1로 오판. 예시 제거 후 n=2 복구)
- 확신 문구 (cal): "비슷한 장면 100개 중 몇 개 맞을지로, 90-100 몰림 금지, 전 범위 사용"
  — A/B 12장: σ 4.4→6.2, 천장 100 소멸, 전선 95→75, 판정 결론 12/12 불변 → 채택
- 패러프레이즈 2종(frozen_v1_para1/2): 의미 동일·문장 재구성 — 안정성 측정용
- **동결 이후 문안·설정 변경 = 본 측정 재실행** (오류 모델이 [모델×프롬프트×설정] 조합의 측정치라서)

### collect.py — 이미지 수집
- 소스: Wikimedia Commons API + **Openverse API**(CC 통합 — 위험물류는 Commons에 없어 Openverse가 살림)
- 전 이미지 출처 URL·라이선스 `images/raw/meta.jsonl` 기록 (논문 출처 명시용)
- 검수 체계: 카테고리별 번호 격자 시트(`images/review/*_sheet.jpg`) → 3중 검수
  (본인 육안 + 검수 요원 5 병렬 + 사용자 최종). 탈락은 `images/raw/_rejected/` 보존
- 최종 167장 (labels.csv): trash 13 / sharps 12 / carcass_s 25 / carcass_l 3 / bulky 16 /
  obstacle 19 / hazard 12 / normal 35 / ambiguous 32 (+mixed_tree_wire 태그 5)

### measure_run.py — 본 측정 (완료 2026-07-30)
- labels.csv × 2밴드 × 3문안 = **1,002판정**. 이어하기 = 캐시 + main_results.jsonl 기완료 스킵
- 합성 far: 장변 200px 다운스케일(`images/far_synth/`) — 원거리 저해상 관측의 대리, 논문 방식 명시
- 산출: `out/main_results.jsonl` (한 줄 = 한 판정: GT·밴드·문안·판정·지연)
- **완주**: 1,002판정 (재시도 후 무응답 20건 잔존 — 응답 실패율로 p_detect에 곱해 반영)
- measure.py 채점 완료 → out/error_model.json. **4방향 독립 검증 통과** (재채점 일치·모델 무결성·
  판정 반박 실패=조건부 GO·GT 정렬 감사). 주의: p_detect의 기본값(near .98/far .85)은 물리 가정
  (실측은 판정 분포와 응답률만) — 논문 방법절 명시 + p_detect 민감도 스윕 후보

## 2. 시뮬레이터 — `sim/`

### geometry.py
- 폴리라인 (s=호길이, t=횡오프셋) 좌표계. loop/왕복(반사) 지원
- `sweep_path`: 톱니 스윕 웨이포인트 (도로변 t=±2m ~ 바깥 폭 w)
- `hub_speed_sync`: **허브 속도 = 노선 길이 / 측면 스윕 1패스 시간** — 셋이 나란히 전진
  (버그 수정 이력: 왕복 2배를 곱해 허브가 2배 속도로 이탈 → 스냅샷 검증으로 발견)
- `intercept_hub`: 허브 궤적 예측 기반 요격점 (τ 5s 격자 탐색) — 레일 회복·조기 소집용

### env.py + exp/configs/simple.yaml
- 환경 = 데이터 파일 (규칙 코드 환경 불변). simple = 1km 직선 왕복, 회랑 폭 40m
- 파라미터: v_sweep 1.0 / sense_r 15 / reliable_r 8 / cell 10m / spacing 25m / ε 30s / grace 0.1
- 온톨로지 ④안 share: trash .35 / sharps .15 / carcass_s .20 / carcass_l .05 / bulky .20 / obstacle .05
- (예정) park.yaml = OSM 실지형 추출 (순환 loop + 다각형 회랑)

### tasks.py
- ρ→λ 유도: λ = ρ·3 / (평균 n × 평균 왕복 이동초). **알려진 조정 항목**: 이동 추정이 낙관적
  → 스모크에서 λ 57건/h로 과부하 (완료율 29%) → metrics 실측 후 재보정 예정
- 비개입 개체: 애매 = task×0.5, hazard = task×0.05. 안정화(warmup) 후에만 주입 (York 관행)

### vlm.py
- 오류 모델 샘플러: model[GT타입][밴드] = {p_detect, judgments 분포} → 로봇별 독립 추첨 (seed 분리)
- P0 완료 전 개발용 `synthetic_model` — **스키마 동일**이라 error_model.json 교체 시 코드 무변경

### memory.py
- 후보 = (xy, s 로그오즈, n̂, û, 시각들, 플래그들). 매칭 반경 8m
- s 갱신: ±logit(보정확신), far은 ×0.5 가중. 근접 관측이 n̂·û 대체 (2단)
- 랑데뷰 병합: 타 로봇 항목 ×0.5 가중 합산 (이중 계상 완화 절충) — 교차 확인 효과

### policy.py — 판단 1~5 (설계 §3 직역)
- 파라미터: θ_act 0.80 / w_idle 1.0 / u_high 3 / ε 30s / 배정이동추정 120s / 소집비용 180s·대
- 판단5 경매: 낙찰자는 그 지점에서 다음 입찰 (경로 연쇄)

### runner.py — full arm 상태기계
- 모드: patrol / detour / wait_site / at_rdv. 1스텝=1초
- 도착 처리: 근접 최종 관측 → 비task면 중단(헛걸음) / GT 비task·hazard에 개입하면 오개입(minor/severe)
  / 인원 미달 + 배정조 오는 중이면 **현장 대기** / 배정조 없으면 **인원부족 복귀**(n̂←GT 확정, 재안건)
- 버그 수정 이력: ①조기 소집 반복 발화 310회 → 후보당 1회 + **앞당기기만 허용**
  ②배정조 선착이 대기 없이 인원부족 복귀 → coalition 집합 기록 + wait_site 모드 신설
  ③Trace.log 인자명 충돌(kind) → tkind로 회피
- 스모크 (2h, seed 1): found 77 / dispatch_now 39 / agenda 16 / bounce 9 / assign 5 /
  convoke 1 / wait_site 1 / rendezvous 2 / complete 29 — 전 메커니즘 발화 확인

### viz.py
- matplotlib 애니메이션 (Agg→gif). 스냅샷 검증이 허브 동기화 버그를 잡음 — **정책 변경 시 스냅샷 확인 습관**

## 3. 남은 구현 (D5~)

- ~~경로 자율성 2건~~ ✅ 구현됨: ①복귀 시 주변 ±250m 창에서 방치 최대 지점으로 걸어가 재개
  ②재관측 예약 지점 들르기 (60m 이내·2회 상한·300s 쿨다운)
- ~~metrics.py~~ ✅ / ~~λ 재보정~~ ✅ (0.25×) / arms 1~7 ✅ (role-adaptive만 잔여)
- metrics.py: idleness 식1~5(정상상태 집계)·GT-u 가중 지연·오개입 TPR/FPR·인원부족 복귀률·소집 발화율·백로그
- arms.py: full / no-sizing / fixed-rdv(TTC·ETC) / no-gate / solo-only / broadcast / greedy-reactive / role-adaptive
- ~~tests/~~ ✅ **63개 통과** (기하 17·경매 16·추첨 분포 24·해석 검증 6 — 해석 검증: 방치 이론식 대비 0.07%)
- 정리: next_interval 죽은 상한식 제거 / tasks λ=0·hazard 0 가드 / hub_pos_at lap 인자 실사용 /
  rail_recovery_target 삭제 (방치 우선 복귀가 대체 — 설계 §9 기록)
- 알려진 단순화: 경매 bid의 공백 항이 이동시간 비례 근사라 낙찰 순위에 w_idle 무영향 (판단2에서만 유효)
- ~~role-adaptive~~ ✅ 구현: 복귀 로봇이 빈 자리(역할+경로+진행도) 중 최근접 자리를 인수 (20m 이상 이득 시만).
  4h 스모크에서 역할 교환 11회 발화. **D5 완료 — 잔여는 D6-7 몫**: exp/run_all.py + stats.py, λ 재점검
- ~~measure.py~~ ✅ (검증 통과, error_model.json 시뮬 주입 스모크 3 arm 통과)
- GT 정비 6건 사용자 검수 대기 (images/review/gt_audit_sheet.jpg) → 재채점 (판정 재사용, 수 초)
- W2 스윕 축 추가: obstacle far 탐지율 30~90% 민감도 (검증단 조건)

## 4. 버그·함정 일지 (시간순)

| # | 증상 | 원인 | 처방 |
|---|---|---|---|
| 1 | ollama.tgz가 9바이트 | 다운로드 URL이 tar.zst로 개편, 구 URL 404 "Not Found" 텍스트 저장 | GitHub 릴리스 API로 실제 에셋 확인 → tar.zst |
| 2 | API 키 형식 경고 | 신형 53자 키 (구형 AIza 39자 아님) | 실호출 검증으로 대체 — 정상 |
| 3 | Gemini 파일럿 중단 | 무료 등급 **일일 25회** (신형 모델, 프로급 취급) | 로컬 Qwen 전환 (사용자 결정) |
| 4 | JSON 파싱 실패 (Extra data) | 모델이 JSON 뒤 잡음 첨부 | raw_decode 첫 오브젝트만 |
| 5 | JSON 파싱 실패 (문자열 내 따옴표) | 스키마 미강제 출력 | responseSchema 구조 강제 |
| 6 | Ollama 간헐 빈 응답 | thinking이 토큰 소진 (10,749자 사고 → 본문 0) | **think=false** + 3단 회복 사다리 |
| 7 | 빈 응답이 계속 재현 | 빈 응답이 캐시에 저장됨 | 검증 후 저장 + 오염 캐시 청소 |
| 8 | 허브가 측면의 2배 속도 | lap에 왕복 ×2 오적용 | 스냅샷 검증으로 발견·수정 |
| 9 | convoke 310회 폭주 | 후보당 발화 제한 없음 + 재예약이 만남을 미래로 밀음 | 1회 플래그 + 앞당기기만 |
| 10 | 코얼리션 선착 인원부족 복귀 | 배정 정보 미참조 | coalition 기록 + wait_site |
| 11 | 재관측 방문 3,124회 폭주 | 방문 상한·쿨다운 없음 | 후보당 2회 + 300s 쿨다운 |
| 12 | 로봇 좌표가 배열로 오염 | 복귀 탐색 빈손 시 pts[None] 인덱싱 | best_i None 가드 |
| 13 | broadcast 랑데뷰 919회 집계 | 가상 회의가 물리 회의 흐름 재사용 | _assign_virtual 분리 + 물리 회합 차단 |
| 14 | 확신 로그오즈 exp 넘침 | 무한 누적 | ±30 클립 |
| 15 | load_model이 래퍼째 반환 → 샘플러 KeyError | error_model.json이 {stats,model} 래퍼 | 'model' 키 자동 언랩 (검증단 적발) |
| 16 | 실측 n̂=4("beyond")가 3대 편대에 미정의 | 스키마엔 있으나 정책 미처리 | 샘플러에서 min(n,3) 상한 (검증단 적발) |
| 17 | broadcast 완료율 14% 붕괴 + 로그오즈 오버플로 | **병합 메아리**: 매초 병합이 상대 총점 절반을 무조건 덧셈 → 같은 증거 초당 재계상 폭주 (랑데뷰 병합도 동일 결함) | 자기 관측분만 교환 + 같은 상대 기여는 교체(재병합 무효과) — full도 33→52% 회복 |
| 18 | 병합 신설 후보의 채점 연결 끊김+소집 차단 | Candidate 자리 인자 실수 — gt_tid가 convoked 칸에 | 키워드 인자 (17 수리 중 발견) |
| 22 | broadcast 12h에서 완료 21% 붕괴, 전 로봇 정지 | **순환 대기 교착**: 경매가 현장 대기 중 로봇을 새 코얼리션에 중복 낙찰 (A·B는 임무1에서 C를, C는 임무2에서 A를 기다림) | ①대기 중 로봇 낙찰 제외 ②대기 시한 20분 → 인원부족 복귀 처리 (파일럿 1차가 적발) |
| 23 | 오개입이 no-gate보다 full에서 더 많은 역전 | **재관측 방문이 개입 경로로 유입** — 보러 간 방문이 일반 출동과 같은 도착 처리 | 커밋 안 된 도착 = 관측만 (기억 갱신+판단 재실행, 개입 불가) — 파일럿 2차가 적발 |
| 20 | 방치 지표가 항상 0 | steady_stats가 계산만 되고 compute()에서 미호출 — 종합 점수 첫 항 누락 위험 | compute 말미 연결 (role-adaptive 스모크가 적발) |
| 21 | 허브 임무 복귀가 순간이동성 | 이탈 시점의 옛 레일 진행도 잔존 | 복귀 시 현 위치를 레일에 투영 |
| 19 | 유예·부분 회의 안전망 미배선 | grace_expired 선언만 되고 호출 없음 — 1대가 늦으면 전원 무한 대기 | due+유예 초과 시 2대 이상이면 부분 회의, 미만이면 회의 무산+재예약 (테스트 요원 적발) |

## 5. 진행 확인 방법

```bash
# 본 측정 진행률
tail -5 /tmp/claude-1000/-home-aprl/*/tasks/bv1qjkgfp.output   # 또는
wc -l ~/patrol/p0/out/main_results.jsonl
# Ollama 상태
curl -s http://127.0.0.1:11434/api/version && nvidia-smi --query-gpu=memory.used --format=csv,noheader
# 시뮬 스모크
cd ~/patrol && python -c "import sys; sys.path.insert(0,'.'); from sim.runner import Episode; ..."
```
