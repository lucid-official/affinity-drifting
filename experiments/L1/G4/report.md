# G4 — unconditional drifting fits the marginal; the learning rate is fixed once here: 결과 보고서

> 자동 생성: `scripts/aggregate.py G4` · commit `0e4e886044` · 2026-11-07 22:03 UTC
> 사전 등록: [prereg.md](prereg.md) (결과를 본 뒤 수정 금지) · 판정 규칙: `experiment.yaml`의 `judge:`

## 판정: **FAIL**

| 판정 | 검사 | 근거 |
| --- | --- | --- |
| FAIL | selected k5_lr3e-4 | w2_marginal_heldout 0.5781 [0.5349, 0.6218] vs 2.0 × w2_marginal_datadata_heldout 0.08763; all: {k5_lr3e-4: 0.5781, k5_lr1e-3: 0.6387, k5_lr3e-3: 0.6764} |

## 수치 (seed 평균, 95% bootstrap CI, 5 seeds)

| arm | w2_marginal_heldout | w2_marginal_datadata_heldout | w2_marginal_train | precision_heldout | recall_heldout | mmd_heldout | train_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| k5_lr3e-4 | 0.5781 [0.5349, 0.6218] | 0.08763 [0.08763, 0.08763] | 0.5674 [0.5338, 0.5882] | 0.1273 [0.1245, 0.1302] | 0.3843 [0.3161, 0.4736] | 0.09848 [0.09541, 0.1011] | 117.6 [93.67, 141.5] |
| k5_lr1e-3 | 0.6387 [0.6118, 0.6653] | 0.08763 [0.08763, 0.08763] | 0.6115 [0.5701, 0.6478] | 0.1249 [0.1205, 0.1292] | 0.4702 [0.4462, 0.5001] | 0.1022 [0.101, 0.104] | 104.4 [89.52, 129.1] |
| k5_lr3e-3 | 0.6764 [0.6497, 0.7031] | 0.08763 [0.08763, 0.08763] | 0.6555 [0.6299, 0.6814] | 0.1214 [0.119, 0.1238] | 0.3461 [0.316, 0.3753] | 0.1093 [0.1031, 0.1162] | 117.6 [93.68, 141.6] |
| drifting | 1.1 [1.097, 1.103] | 0.08763 [0.08763, 0.08763] | 1.099 [1.095, 1.103] | 0.000769 [0, 0.001971] | 0 [0, 0] | 0.8427 [0.8415, 0.8439] | 82.84 [65.77, 112.5] |

누락 run: 0개

## Figures

- (아직 없음: `scripts/make_figures.py G4`)

## 이상 징후

- `k5_lr3e-4/seed0`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `k5_lr3e-4/seed1`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `k5_lr3e-4/seed2`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `k5_lr3e-4/seed3`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `k5_lr3e-4/seed4`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `k5_lr1e-3/seed0`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `k5_lr1e-3/seed1`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `k5_lr1e-3/seed2`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `k5_lr1e-3/seed3`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `k5_lr1e-3/seed4`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `k5_lr3e-3/seed0`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `k5_lr3e-3/seed1`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `k5_lr3e-3/seed2`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `k5_lr3e-3/seed3`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `k5_lr3e-3/seed4`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `drifting/seed0`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `drifting/seed1`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `drifting/seed2`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `drifting/seed3`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `drifting/seed4`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)

### 사람이 적는 이상 징후

-

## 해석 (판정 이후, 사전 등록과 구분해서)

-
