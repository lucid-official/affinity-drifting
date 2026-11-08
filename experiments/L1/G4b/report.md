# G4b — unconditional drifting (full-batch repulsion) fits the marginal; the learning rate is fixed once here: 결과 보고서

> 자동 생성: `scripts/aggregate.py G4b` · commit `0af89039be` · 2026-11-08 09:28 UTC
> 사전 등록: [prereg.md](prereg.md) (결과를 본 뒤 수정 금지) · 판정 규칙: `experiment.yaml`의 `judge:`

## 판정: **FAIL**

| 판정 | 검사 | 근거 |
| --- | --- | --- |
| FAIL | selected uncond_lr1e-3 | w2_marginal_heldout 0.1849 [0.1748, 0.1944] vs 2.0 × w2_marginal_datadata_heldout 0.08763; all: {uncond_lr1e-4: 0.1885, uncond_lr3e-4: 0.1942, uncond_lr1e-3: 0.1849} |

## 수치 (seed 평균, 95% bootstrap CI, 5 seeds)

| arm | w2_marginal_heldout | w2_marginal_datadata_heldout | w2_marginal_train | recall_heldout | train_seconds |
| --- | --- | --- | --- | --- | --- |
| uncond_lr1e-4 | 0.1885 [0.169, 0.2079] | 0.08763 [0.08763, 0.08763] | 0.1525 [0.1421, 0.1629] | 0.8672 [0.8637, 0.8707] | 208 [189.4, 226.6] |
| uncond_lr3e-4 | 0.1942 [0.1842, 0.2034] | 0.08763 [0.08763, 0.08763] | 0.1543 [0.1471, 0.1615] | 0.8573 [0.853, 0.8624] | 250.4 [244.1, 260.3] |
| uncond_lr1e-3 | 0.1849 [0.1748, 0.1944] | 0.08763 [0.08763, 0.08763] | 0.1636 [0.1573, 0.1699] | 0.8606 [0.854, 0.8669] | 250.4 [244.1, 260.4] |
| drifting | 1.087 [1.081, 1.093] | 0.08763 [0.08763, 0.08763] | 1.086 [1.079, 1.091] | 0 [0, 0] | 70.72 [67.66, 75.42] |

누락 run: 0개

## Figures

- (아직 없음: `scripts/make_figures.py G4b`)

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
