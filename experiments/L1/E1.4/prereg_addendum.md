# 사전 등록 부록 (run 전에 커밋)

- 작성 2026-11-08 · 실험 엔지니어 Claude · 승인 Chan Lee (D20, D21, D22, D23)
- `prereg.md`는 커밋 후 수정 금지라 그대로 두고, 그 뒤에 승인된 설정 변경을 여기에 적는다. **판정 규칙(`experiment.yaml`의
  `judge:`)은 바뀌지 않는다.** 이 부록은 이 실험의 어떤 run보다 먼저 커밋됐다.

## prereg.md의 "설정" 중 바뀐 것

| 항목 | prereg.md | 실제 (이 부록) | 근거 |
| --- | --- | --- | --- |
| kernel 폭 σ | bank의 16번째 최근접 이웃 거리 median 하나(D15) | 다중 스케일 σ = (0.071, 0.384, 2.07): kNN 폭 → median heuristic 기하 수열, 스케일마다 항별 drift normalization 후 합 | D20 (G4 1차 FAIL 진단) |
| batch | 조건 64개 × 생성 8개 | 조건 8개 × 생성 64개 (총 512개 동일) | D22 |
| 학습률 | Gate G4에서 정한 값 | **1e-3** (Gate G4b의 선택 규칙. G4b는 통과선을 약 5% 넘겨 FAIL, 사용자 결정으로 진행) | D17, D21, D23 |

## 영향

- G4b가 FAIL이었으므로(D23), E1.2의 G3(oracle K4의 held-out W2 ≤ 2 × data–data)가 실패하면 L1을 즉시 멈춘다.

- 모든 AffD 계열 arm(AffD, single, K1–K5, factorial의 모든 셀)에 같은 변경이 적용된다. baseline `drifting`은 release 손실
  그대로이고 batch(조건 8 × 64)와 lr만 같다.
- E1.5의 이웃 묶음 batch는 조건 8개 = 앵커 1개 + 캡션 가중치로 뽑은 이웃 7개가 된다.
- E1.5의 NW mixture TV는 가장 작은 스케일(σ = 0.071)로 평활한 밀도끼리 비교한다.
