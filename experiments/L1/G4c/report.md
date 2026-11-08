# G4c — unconditional drifting (full-batch repulsion, whole-field normalisation) fits the marginal; lr fixed here: 결과 보고서

> 자동 생성: `scripts/aggregate.py G4c` · commit `b7a04fbaca` · 2026-11-08 11:26 UTC
> 사전 등록: [prereg.md](prereg.md) (결과를 본 뒤 수정 금지) · 판정 규칙: `experiment.yaml`의 `judge:`

## 판정: **PASS**

| 판정 | 검사 | 근거 |
| --- | --- | --- |
| PASS | selected uncond_lr1e-3 | w2_marginal_heldout 0.1362 [0.1293, 0.1441] vs 2.0 × w2_marginal_datadata_heldout 0.08763; all: {uncond_lr1e-4: 0.1441, uncond_lr3e-4: 0.1464, uncond_lr1e-3: 0.1362} |

## 수치 (seed 평균, 95% bootstrap CI, 5 seeds)

| arm | w2_marginal_heldout | w2_marginal_datadata_heldout | w2_marginal_train | recall_heldout | train_seconds |
| --- | --- | --- | --- | --- | --- |
| uncond_lr1e-4 | 0.1441 [0.1287, 0.1623] | 0.08763 [0.08763, 0.08763] | 0.1466 [0.1358, 0.1596] | 0.9811 [0.9793, 0.9832] | 194.5 [180.7, 208.3] |
| uncond_lr3e-4 | 0.1464 [0.1335, 0.1613] | 0.08763 [0.08763, 0.08763] | 0.1468 [0.1338, 0.1584] | 0.9796 [0.9777, 0.9813] | 244.5 [243.3, 246.2] |
| uncond_lr1e-3 | 0.1362 [0.1293, 0.1441] | 0.08763 [0.08763, 0.08763] | 0.1458 [0.1415, 0.15] | 0.9786 [0.9772, 0.9796] | 244.5 [243.3, 246.2] |
| drifting | 1.087 [1.081, 1.093] | 0.08763 [0.08763, 0.08763] | 1.086 [1.079, 1.091] | 0 [0, 0] | 67.08 [66.46, 67.42] |

누락 run: 0개

## Figures

- (아직 없음: `scripts/make_figures.py G4c`)

## 이상 징후

- `uncond_lr1e-4/seed0`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `uncond_lr1e-4/seed1`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `uncond_lr1e-4/seed2`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `uncond_lr1e-4/seed3`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `uncond_lr1e-4/seed4`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `uncond_lr3e-4/seed0`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `uncond_lr3e-4/seed1`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `uncond_lr3e-4/seed2`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `uncond_lr3e-4/seed3`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `uncond_lr3e-4/seed4`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `uncond_lr1e-3/seed0`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `uncond_lr1e-3/seed1`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `uncond_lr1e-3/seed2`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `uncond_lr1e-3/seed3`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `uncond_lr1e-3/seed4`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `drifting/seed0`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `drifting/seed1`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `drifting/seed2`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `drifting/seed3`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `drifting/seed4`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)

### 사람이 적는 이상 징후

-

## 해석 (판정 이후, 사전 등록과 구분해서)

-
