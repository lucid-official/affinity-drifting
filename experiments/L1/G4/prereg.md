# G4 — 사전 등록: Gate G4 (unconditional drifting, 학습률 선택)

- 작성 2026-11-08 · 실험 엔지니어 Claude · 결정 위임: Chan Lee (D17)
- 이 파일과 `experiment.yaml`은 첫 run 전에 커밋하고 이후 수정하지 않는다.

## plan 원문 (docs/plan.md §3, G4 행)

| Gate | 확인하는 것 | 통과 기준 | 실패 시 조치 | 적용 |
| --- | --- | --- | --- | --- |
| G4 Unconditional | 조건 없는 drifting이 주변 분포를 맞추는가 | 주변 분포 지표가 원 논문 수준의 품질을 보임 | 학습 하이퍼파라미터를 먼저 맞춘다 | 모든 Level |

## 운영 정의 (D17, 사용자 위임 2026-11-08)

- "조건 없는 drifting" = K5(`h_p.rule: inf`): 모든 조건이 bank 전체를 같은 가중치로 빌려와 주변 분포 p(x)를 배운다.
- 학습: B0 L1 그대로(Adam β = (0.9, 0.999), EMA 0.999, 20,000 스텝, 조건 64개 × 생성 8개, MLP 256×3, ε 4차원).
  lr ∈ {3e-4, 1e-3, 3e-3} 표준 격자. seed 5개.
- 지표: held-out 조건 256개에서 생성 512개씩을 모은 주변 분포와 참 주변 분포의 W2(2,048개 무작위 부분표본, 정확 OT).
  기준선: 같은 크기의 참 표본 두 벌 사이 W2(data–data).
- **선택**: seed 평균 주변 W2가 가장 작은 lr. **통과**: 그 값 ≤ 2 × data–data W2. ‖V‖²는 쓰지 않는다(CLAUDE.md §2.6).
- 결과: 선택된 lr을 `configs/base_B0.yaml`의 `levels.L1.train.lr`에 적고, 이후 모든 L1 arm이 같은 값을 쓴다(CLAUDE.md §2.8).
- 실패하면 L1을 멈추고 보고한다.

## baseline `drifting` (D11) — 판정 대상 아님

원 논문 Drifting(Lucid release, 정규화 포함)을 lr 1e-3으로 같은 seed로 돌린다. 조건부(자기 positive 하나) 설정이라
주변 분포 W2는 참고용이다.
