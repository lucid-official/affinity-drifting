# E0.5 — counterfactual negatives tilt to the closed-form Gaussian: 결과 보고서

> 자동 생성: `scripts/aggregate.py E0.5` · commit `bce86ecdc3` · 2026-11-07 20:07 UTC
> 사전 등록: [prereg.md](prereg.md) (결과를 본 뒤 수정 금지) · 판정 규칙: `experiment.yaml`의 `judge:`

## 판정: **FAIL**

| 판정 | 검사 | 근거 |
| --- | --- | --- |
| PASS | lam_0:rel_err_mean | seed mean 3.804e-05 [3.393e-05, 4.23e-05] vs 0.05 |
| PASS | lam_0:rel_err_var | seed mean 6.798e-05 [4.687e-05, 8.909e-05] vs 0.05 |
| PASS | lam_0.25:rel_err_mean | seed mean 0.02776 [0.02696, 0.02843] vs 0.05 |
| PASS | lam_0.25:rel_err_var | seed mean 0.03171 [0.03087, 0.03288] vs 0.05 |
| FAIL | lam_0.5:rel_err_mean | seed mean 3145 [2914, 3382] vs 0.05 |
| FAIL | lam_0.5:rel_err_var | seed mean 2.895 [2.767, 3.114] vs 0.05 |
| FAIL | lam_0.75:rel_err_mean | seed mean 4697 [4540, 4831] vs 0.05 |
| FAIL | lam_0.75:rel_err_var | seed mean 15.94 [15.06, 16.41] vs 0.05 |
| FAIL | lam_0.9:rel_err_mean | seed mean 1.665e+05 [1.607e+05, 1.712e+05] vs 0.05 |
| FAIL | lam_0.9:rel_err_var | seed mean 7.393e+04 [7.38e+04, 7.401e+04] vs 0.05 |
| PASS | lam_1.1:diverged | 5/5 seeds diverged |

## 수치 (seed 평균, 95% bootstrap CI, 5 seeds)

| arm | rel_err_mean | rel_err_var | var_c0 | steps | final_rms_v | wall_time_s |
| --- | --- | --- | --- | --- | --- | --- |
| lam_0 | 3.804e-05 [3.393e-05, 4.23e-05] | 6.798e-05 [4.687e-05, 8.909e-05] | 0.25 [0.25, 0.25] | 1.324e+04 [1.245e+04, 1.425e+04] | 9.137e-07 [9.041e-07, 9.233e-07] | 68.63 [55.37, 82.39] |
| lam_0.25 | 0.02776 [0.02696, 0.02843] | 0.03171 [0.03087, 0.03288] | 0.4619 [0.4609, 0.463] | 5e+04 [5e+04, 5e+04] | 2.86e-06 [2.219e-06, 3.297e-06] | 436 [428, 446.3] |
| lam_0.5 | 3145 [2914, 3382] | 2.895 [2.767, 3.114] | 3.748 [3.533, 4.032] | 4.168e+04 [2.972e+04, 5e+04] | 1.095 [0.6137, 1.808] | 517.5 [368.5, 621.7] |
| lam_0.75 | 4697 [4540, 4831] | 15.94 [15.06, 16.41] | 21.25 [20.08, 21.91] | 50 [50, 50] | 1.644e+04 [1.589e+04, 1.691e+04] | 0.62 [0.6133, 0.6303] |
| lam_0.9 | 1.665e+05 [1.607e+05, 1.712e+05] | 7.393e+04 [7.38e+04, 7.401e+04] | 1.034e+05 [1.032e+05, 1.036e+05] | 50 [50, 50] | 2.53e+06 [2.441e+06, 2.601e+06] | 0.6416 [0.6343, 0.6528] |
| lam_1.1 | — | — | 1.463e+09 [1.378e+09, 1.536e+09] | 50 [50, 50] | 1.009e+09 [9.705e+08, 1.039e+09] | 0.5716 [0.5279, 0.6141] |
| drifting | — | — | — | 50 [50, 50] | — | 4.396 [4.329, 4.457] |

누락 run: 0개

## Figures

- [F0.4](../../../figures/L0/E0.5/F0.4/F0.4.png)
- [FC.1](../../../figures/L0/E0.5/FC.1/FC.1.png)

## 이상 징후

- `lam_0.25/seed0`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 3.19e-06)
- `lam_0.25/seed1`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 1.64e-06)
- `lam_0.25/seed2`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 3.29e-06)
- `lam_0.25/seed3`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 2.83e-06)
- `lam_0.25/seed4`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 3.36e-06)
- `lam_0.5/seed0`: diverged
- `lam_0.5/seed1`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 8.98e-01)
- `lam_0.5/seed2`: diverged
- `lam_0.5/seed3`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 4.40e-01)
- `lam_0.5/seed4`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 8.25e-01)
- `lam_0.75/seed0`: diverged
- `lam_0.75/seed1`: diverged
- `lam_0.75/seed2`: diverged
- `lam_0.75/seed3`: diverged
- `lam_0.75/seed4`: diverged
- `lam_0.9/seed0`: diverged
- `lam_0.9/seed1`: diverged
- `lam_0.9/seed2`: diverged
- `lam_0.9/seed3`: diverged
- `lam_0.9/seed4`: diverged
- `drifting/seed0`: diverged
- `drifting/seed1`: diverged
- `drifting/seed2`: diverged
- `drifting/seed3`: diverged
- `drifting/seed4`: diverged

### 사람이 적는 이상 징후

-

## 해석 (판정 이후, 사전 등록과 구분해서)

-
