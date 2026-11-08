# G4b — 사전 등록: Gate G4 2차 (조건 없는 drifting, 학습률 선택)

- 작성 2026-11-08 · 실험 엔지니어 Claude · 승인 Chan Lee (D20, D21, D22)
- 이 파일과 `experiment.yaml`은 첫 run 전에 커밋하고 이후 수정하지 않는다.
- 1차 G4는 FAIL이었다(`experiments/L1/G4/report.md`, 진단은 `diagnosis.md`). 그 기록은 그대로 둔다.

## plan 원문 (docs/plan.md §3, G4 행)

| Gate | 확인하는 것 | 통과 기준 | 실패 시 조치 | 적용 |
| --- | --- | --- | --- | --- |
| G4 Unconditional | 조건 없는 drifting이 주변 분포를 맞추는가 | 주변 분포 지표가 원 논문 수준의 품질을 보임 | 학습 하이퍼파라미터를 먼저 맞춘다 | 모든 Level |

## 1차와 달라진 것

- **D21**: "조건 없는 drifting" = h_p = ∞(모든 조건이 bank 전체를 같은 가중치로) **그리고** h_q = ∞(batch의 생성 샘플 512개
  전체끼리 repulsion, 자기 제외). release의 조건 없는 drifting과 같은 구성이다.
- **D20**: 다중 스케일 kernel, σ = (0.071, 0.384, 2.07)(실험별 `sigma.json`에 고정).
- **D22**: 조건 8개 × 생성 64개(batch 512개).
- lr 격자 {1e-4, 3e-4, 1e-3}(1차의 3e-3은 제외하고 1e-4를 넣는다: 1차 탐색에서 큰 lr이 불리했다).

## 운영 정의 (그 밖은 1차와 같다)

- 학습: Adam β = (0.9, 0.999), EMA 0.999, 20,000 스텝, MLP 256 × 3, ε 4차원. seed 5개.
- 지표: held-out 조건 256개 × 생성 512개를 모은 주변 분포와 참 주변 분포의 W2(2,048개 부분표본, 정확 OT).
  기준선: 같은 크기의 참 표본 두 벌 사이 W2.
- **선택**: seed 평균 주변 W2가 가장 작은 lr. **통과**: 그 값 ≤ 2 × data–data W2. ‖V‖²는 쓰지 않는다.
- 통과하면 선택한 lr을 `configs/base_B0.yaml`의 `levels.L1.train.lr`에 적고 모든 L1 arm이 같은 값을 쓴다.
  실패하면 L1을 멈추고 보고한다.
- 1차 탐색(seed 0)에서 lr 3e-4의 이 구성이 0.175로 통과선(0.1753)에 걸쳐 있었다. 5 seed 결과로만 판정한다.

## baseline `drifting` (D11) — 판정 대상 아님

원 논문 Drifting(Lucid release, 자기 positive 하나)을 lr 3e-4로 같은 seed로 돌린다. 주변 W2는 참고용이다.
