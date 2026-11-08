# G4c — 사전 등록: Gate G4 3차 (조건 없는 drifting, 학습률 선택)

- 작성 2026-11-08 · 실험 엔지니어 Claude · 승인 Chan Lee (D24)
- 이 파일과 `experiment.yaml`은 첫 run 전에 커밋하고 이후 수정하지 않는다.
- 1차 G4(FAIL, 설계 오류), 2차 G4b(FAIL, 통과선 약 5% 초과)는 기록으로 남는다. G4b의 lr 선택은 항별 drift normalization을
  전제로 한 것이라, 정규화를 원 논문 방식으로 바꾼(D24) 지금 lr을 다시 고른다.

## plan 원문 (docs/plan.md §3, G4 행)

| Gate | 확인하는 것 | 통과 기준 | 실패 시 조치 | 적용 |
| --- | --- | --- | --- | --- |
| G4 Unconditional | 조건 없는 drifting이 주변 분포를 맞추는가 | 주변 분포 지표가 원 논문 수준의 품질을 보임 | 학습 하이퍼파라미터를 먼저 맞춘다 | 모든 Level |

## G4b와 달라진 것

- **D24**: drift normalization을 합친 필드 V 전체에 스케일마다 적용(release의 temperature별 정규화와 같은 방식).
- 그 밖(전체 batch repulsion D21, 다중 스케일 kernel D20, 조건 8 × 생성 64 D22, lr 격자 {1e-4, 3e-4, 1e-3}, 20,000 스텝,
  seed 5개, 지표·통과선)은 G4b와 같다.

## 판정과 그 뒤 (G4b와 같은 규칙)

- **선택**: seed 평균 held-out 주변 W2가 가장 작은 lr. **통과**: 그 값 ≤ 2 × data–data W2(0.0876 → 0.1753).
- 통과하면 선택한 lr을 `levels.L1.train.lr`에 적는다. **실패하면** G4b와 같은 사용자 결정(D23: 통과선의 근거, E1.2의 G3가
  조건부 파이프라인을 따로 점검)을 다시 받기 전에는 L1 본 실험을 돌리지 않는다.

## baseline `drifting` (D11) — 판정 대상 아님

원 논문 Drifting(Lucid release, 자기 positive 하나)을 lr 3e-4로 같은 seed로 돌린다.
