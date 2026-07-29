"""비교군(arm) 정의 — 각 arm은 결정 규칙의 켜고 끄기 조합.

절제 사다리 (문헌정독 §6a): 상한(broadcast) / 절제 4종 / 하한(solo) / 외부(sebs).
전 arm이 동일한 환경·임무 스트림·오류 모델을 공유하고 결정 로직만 다름 (모듈 치환식 공정화).
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Arm:
    name: str
    use_sizing: bool = True       # False: n̂ 무시 — 전부 혼자 즉시 출동 (반응식, DEXTER 원리)
    use_gate: bool = True         # False: 첫 task 판정에 바로 행동 (확신 문턱 없음)
    adaptive_rdv: bool = True     # False: 고정 주기 랑데뷰 (TTC) — 간격 조정·조기 소집 없음
    use_agenda: bool = True       # False: 안건화·경매 자체가 없음 (solo-only)
    broadcast: bool = False       # True: 통신 무제약 — 매초 전 로봇 기억 공유+즉시 배정 (상한선)
    sebs_patrol: bool = False     # True: 고정 톱니 대신 방치 시간 탐욕 순찰 (외부 기준 SEBS 계열)


ARMS = {
    "full":       Arm("full"),
    "no-sizing":  Arm("no-sizing", use_sizing=False),
    "no-gate":    Arm("no-gate", use_gate=False),
    "fixed-rdv":  Arm("fixed-rdv", adaptive_rdv=False),
    "solo-only":  Arm("solo-only", use_agenda=False),
    "broadcast":  Arm("broadcast", broadcast=True),
    "sebs-reactive": Arm("sebs-reactive", sebs_patrol=True, use_sizing=False,
                         use_agenda=False, adaptive_rdv=False),
    # role-adaptive는 P0 주입 후 별도 구현 (역할 재배정 로직)
}
