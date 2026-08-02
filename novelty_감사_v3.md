# v3 novelty 감사 (실내 이종 순찰) — 2026-08-03

> 스위프 4파 합산: ①초기 6요원(08-01) ②보강 5각도+심판(어휘·비arXiv·최신·특허·일본유럽)
> ③최대강도 10각도+심판(T-RO/IJRR·SciRob/RSS·ICRA/IROS·CoRL/워크숍·arXiv 2기간·역인용 2·서베이·대회)
> ④CCTV→로봇 파견 각도. 총 요원 28, 위협 후보 60+ 중 실물 초록 검증 48건.

## 최종 판정: 치명 0 — 조건부 생존

**어떤 논문도 ①무지시 VLM 상식 발견 + ②장면→능력 조합(유형×대수) 산정 + ③규칙 파견을
결합하지 않음.** 특히 ②(장면에서 "집게1+운반1"을 산정)는 검증된 전체에서 부재 — 최강 방어축.
가장 근접해도: 단일 헬퍼 입찰 선택(2505.13376) / 텍스트→단일 스킬 태그(2605.21242) /
고정 특화유닛 분해 파견(ACCESS 2026) / 능력-과업 매칭 최적화(T-RO 2023 — 요구는 외부 입력).

## 소멸한 단독 축들 (기여 문장 규정)

| 축 | 선점자 | 함의 |
|---|---|---|
| ① 무지시 발견 단독 | **TASMap** (ICRA 2025, 한양대 — 실내·무명령 과제 제안, 단일 로봇) + **AutoRT** (DeepMind — VLM/LLM 과제 자기제안을 20+ 플릿에서, 데이터 수집 목적) + Patrol Agent (ICRCV 2024, 도시 UAV) | "발견"을 단독으로 팔면 즉사 |
| "발견+할당" 프레임명 | Map-TIDAL (ICRA 2026?, 수중 기하 기반 — 원문 미확인) | 용어 구분 필수 (기하 vs 의미·상식) |
| 장면 인식 역할 할당 | OC-HMAS (JIOT 2025) | "역할 할당 vs 임무·편성 자체의 발견·산정" 구분 |
| 알람→로봇 자동 파견 | Batalin&Sukhatme 2003 (센서망) + **Cobalt Robotics 상용** (Genetec 연동 제품) | 폐쇄 알람 목록·단일 기종 vs 개방 어휘·편성 산정 |
| ③ 경량 파견 효율 논거 | OSDAG (2026 — LLM 1회+온라인 스케줄러, 이종) | ③의 세일즈는 "발견·산정과의 결합"으로 |

**금지 표현**: "발견→파견 루프 최초" / "VLM 순찰 최초" / "이종 LLM 할당 최초" /
"고정센서→로봇 파견 최초" / "capability-aware 최초" (CASH가 용어 선점).
**허용 기여 문장**: "지시 없는 VLM 상식 임무 발견 + 장면 기반 능력 조합(유형×대수) 산정 +
규칙 파견의 **결합**" + "②는 능력-과업 '매칭'이 아니라 **요구량 자체를 장면에서 산정**"
+ "연합 형성(coalition formation)의 소요 산정을 심볼릭 스펙이 아닌 원시 장면+VLM으로 대체".

## 차별화 우선순위 (관련 연구 절 설계도)

- **P1 (개별 문단)**: TASMap · AutoRT — ① 선행 인정 후 단일로봇/데이터수집 한계와 ②③ 부재로 구분
- **P2 (개별 문단)**: DEXTER-LLM (IROS25 — 임무 명세 종속 동적 태스크) · RoboOS-NeXT (시스템 프레임 최근접)
  · Seeing-Saying-Solving (반응적 자기충돌→헬퍼 1대) · OC-HMAS · Patrol Agent · Agentic Skill Discovery
- **P3 (문단 내 병기)**: ras-llm-cp (RAS 동일지 최근접) · OSDAG · CASH (용어) · Map-TIDAL (원문 확인 후)
  · 에너지시설 LLM (ACCESS 26)
- **P4 (묶음+한 줄)**: T-RO 2023 능력 불확실 스케줄링 · CaTL+ (IJRR 24) · LaMMA-P · HVBTA · LLM-GROP
  · 이상감지 묶음 (Sinha RSS24 · Language-as-Cost IROS25) · Patel&Chernova (proactive 계보)
  · 수중 검사 · 해양 UAV-USV
- **P5 (묶음 문단)**: 지시 구동 LLM 분해·할당 전체 (SMART-LLM·COHERENT·Scale-Plan·AutoHMA·DynaHMRC
  ·Melding·STRAP-LLM·ICIIBMS·CoordField·MultiUAV-Plat·CoMuRoS·MHRC·LLM-HBT·EMTP·ad hoc RSS24)
  + 고전 (Gerkey·Vig&Adams·Farinelli·DINTA·WatchBot·Li&Parker·Chakravarty) + Cobalt 산업 한 문장

## 잔여 추적 (집필 전 필수)

1. **Map-TIDAL 원문** (모든 경로 실패 — ①인접 주장 미검증. 확인 전 관련연구 편입 금지)
2. AutoExpand (IROS25, 실내 SAR 서브태스크 생성?) · EMTP (OpenReview 봇차단) · hierprompt/crossreg/funcexpr
3. 2505.13376 게재처 (IROS류 채택 시 위협 상승) · STRAP-LLM 원문
4. AuRo 멀티로봇 LLM 서베이의 GitHub 갱신 목록 1회 크롤
5. 보안·시설관리 순찰 로봇 상용 (Knightscope·SMP 등 — Cobalt 외 미커버)
6. CoRL 2026·IROS 2026 하반기 신간 재확인 (투고 직전), 중국어권 저널

## 커버리지 정직 고백

WebSearch 예산 소진으로 3파 검증은 API 직접 조회(arXiv·OpenAlex·RSS 원문) 위주 —
"순찰+VLM 이상감지" 키워드의 미발견 신간 가능성 잔존. 미검증 8건은 위 잔여 목록에.
**속도 경고 상향**: 이번 발견의 절반이 최근 8개월 논문 (TASMap ICRA25, DEXTER IROS25,
RoboOS-NeXT 25-10, OSDAG 26-06...). 이 지대는 분기 단위로 채워지고 있음 — 조기 투고가 최선의 방어.
