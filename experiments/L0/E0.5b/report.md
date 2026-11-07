# E0.5b — Prop E under a Gaussian closure — algebra without finite-sample tails: 결과 보고서

> 자동 생성: `scripts/aggregate.py E0.5b` · commit `bce86ecdc3` · 2026-11-07 20:07 UTC
> 사전 등록: [prereg.md](prereg.md) (결과를 본 뒤 수정 금지) · 판정 규칙: `experiment.yaml`의 `judge:`

## 판정: **PASS**

| 판정 | 검사 | 근거 |
| --- | --- | --- |
| PASS | lam_0:rel_err_mean | seed mean 8.555e-08 [7.556e-08, 9.553e-08] vs 0.05 |
| PASS | lam_0:rel_err_var | seed mean 1.316e-07 [1.267e-07, 1.369e-07] vs 0.05 |
| PASS | lam_0.25:rel_err_mean | seed mean 1.092e-07 [9.279e-08, 1.256e-07] vs 0.05 |
| PASS | lam_0.25:rel_err_var | seed mean 6.189e-08 [5.254e-08, 7.084e-08] vs 0.05 |
| PASS | lam_0.5:rel_err_mean | seed mean 1.55e-07 [1.236e-07, 1.863e-07] vs 0.05 |
| PASS | lam_0.5:rel_err_var | seed mean 3.317e-08 [1.861e-08, 4.732e-08] vs 0.05 |
| PASS | lam_0.75:rel_err_mean | seed mean 2.069e-07 [1.119e-07, 2.796e-07] vs 0.05 |
| PASS | lam_0.75:rel_err_var | seed mean 8.758e-08 [6.407e-08, 1.042e-07] vs 0.05 |
| PASS | lam_0.9:rel_err_mean | seed mean 3.329e-06 [3.22e-06, 3.456e-06] vs 0.05 |
| PASS | lam_0.9:rel_err_var | seed mean 1.499e-07 [6.057e-08, 2.781e-07] vs 0.05 |
| PASS | lam_1.1:diverged | 5/5 seeds diverged |

## 수치 (seed 평균, 95% bootstrap CI, 5 seeds)

| arm | rel_err_mean | rel_err_var | var_c0 | steps | final_rms_v | wall_time_s |
| --- | --- | --- | --- | --- | --- | --- |
| lam_0 | 8.555e-08 [7.556e-08, 9.553e-08] | 1.316e-07 [1.267e-07, 1.369e-07] | 0.25 [0.25, 0.25] | 1050 [1050, 1050] | 3.949e-08 [3.918e-08, 3.982e-08] | 4.279 [4.027, 4.504] |
| lam_0.25 | 1.092e-07 [9.279e-08, 1.256e-07] | 6.189e-08 [5.254e-08, 7.084e-08] | 0.4344 [0.4344, 0.4344] | 1100 [1100, 1100] | 8.649e-08 [8.147e-08, 9.446e-08] | 5.358 [5.215, 5.509] |
| lam_0.5 | 1.55e-07 [1.236e-07, 1.863e-07] | 3.317e-08 [1.861e-08, 4.732e-08] | 0.6189 [0.6189, 0.6189] | 1150 [1150, 1150] | 1.387e-07 [1.194e-07, 1.675e-07] | 5.613 [5.441, 5.785] |
| lam_0.75 | 2.069e-07 [1.119e-07, 2.796e-07] | 8.758e-08 [6.407e-08, 1.042e-07] | 0.8033 [0.8033, 0.8033] | 1300 [1300, 1300] | 3.146e-07 [2.889e-07, 3.402e-07] | 6.329 [6.092, 6.524] |
| lam_0.9 | 3.329e-06 [3.22e-06, 3.456e-06] | 1.499e-07 [6.057e-08, 2.781e-07] | 0.914 [0.914, 0.914] | 5e+04 [5e+04, 5e+04] | 1.356e-06 [1.127e-06, 1.604e-06] | 253.2 [252.3, 254] |
| lam_1.1 | — | — | 62.08 [0.2784, 148.2] | 340 [310, 370] | 911.6 [778.9, 1106] | 1.609 [1.46, 1.773] |
| drifting | — | — | — | 50 [50, 50] | — | 4.373 [4.302, 4.452] |

누락 run: 0개

## Figures

- [F0.4](../../../figures/L0/E0.5b/F0.4/F0.4.png)
- [FC.1](../../../figures/L0/E0.5b/FC.1/FC.1.png)

## 이상 징후

- `lam_0.9/seed0`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 1.26e-06)
- `lam_0.9/seed1`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 1.16e-06)
- `lam_0.9/seed2`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 1.59e-06)
- `lam_0.9/seed3`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 1.01e-06)
- `lam_0.9/seed4`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 1.76e-06)
- `drifting/seed0`: diverged
- `drifting/seed1`: diverged
- `drifting/seed2`: diverged
- `drifting/seed3`: diverged
- `drifting/seed4`: diverged

### 사람이 적는 이상 징후

-

## 해석 (판정 이후, 사전 등록과 구분해서)

-
