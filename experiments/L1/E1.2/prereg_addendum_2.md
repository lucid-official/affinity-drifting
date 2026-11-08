# 사전 등록 부록 2 (run 전에 커밋)

- 작성 2026-11-08 · 실험 엔지니어 Claude · 승인 Chan Lee (D24)
- `prereg.md`와 첫 부록(`prereg_addendum.md`)은 그대로 둔다. 판정 규칙(`experiment.yaml`의 `judge:`)은 바뀌지 않는다.
  이 부록은 아래 설정으로 돌린 어떤 run보다 먼저 커밋됐다.

## 바뀐 것

| 항목 | 이전 | 실제 | 근거 |
| --- | --- | --- | --- |
| drift normalization | 항별(V⁺, V⁻ 각각 E‖·‖²/C = 1) | 합친 필드 V 전체를 스케일마다 정규화(release의 temperature별 정규화와 같은 방식) | D24 |
| 학습률 | 1e-3 (G4b 선택 규칙, G4b FAIL, D23) | **1e-3** (Gate G4c **PASS**: 주변 W2 0.136 [0.129, 0.144] ≤ 0.175) | G4c |

## 이전 run

- 항별 정규화로 끝난 run(E1.2 `single` seed 0–3, E1.4 `d0_h0` seed 0–3)과 smoke 기록은
  `runs_superseded/D24/`로 옮겼다. 집계에 쓰지 않는다.
- E1.1은 drift normalization을 쓰지 않는 기울기 검사라 영향이 없다(결과 유지).
