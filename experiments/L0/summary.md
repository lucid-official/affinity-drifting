# L0 요약

> 자동 생성: `scripts/level_summary.py L0` · commit `e594287d00` · 2026-11-07 20:09 UTC
> 판정은 각 실험의 사전 등록된 `judge:` 규칙으로만 냈다. 해석은 아래 "사람이 적는 부분"과 각 report의 해석 칸에 둔다.

## 판정

| ID | Claim | 질문 | 판정 | 보고서 |
| --- | --- | --- | --- | --- |
| [E0.1](E0.1/) | Lemma 1 | V = σ²∇log(p̂/q̂) for a Gaussian kernel | **PASS** | [report](E0.1/report.md) |
| [E0.2](E0.2/) | Prop A | symmetric h_q = h_p memorises | **INCONCLUSIVE** | [report](E0.2/report.md) |
| [E0.3](E0.3/) | Prop C | h_q = 0 gives the NW mixture w̃ | **FAIL** | [report](E0.3/report.md) |
| [E0.4](E0.4/) | Prop A–C | h_q interpolates monotonically from NW mixture to memorisation | **PASS** | [report](E0.4/report.md) |
| [E0.5](E0.5/) | Prop E | counterfactual negatives tilt to the closed-form Gaussian | **FAIL** | [report](E0.5/report.md) |
| [E0.5b](E0.5b/) | Prop E (diagnostic) | Prop E under a Gaussian closure — algebra without finite-sample tails | **PASS** | [report](E0.5b/report.md) |
| [E0.6](E0.6/) | 단서 1 | joint vs separate softmax share the fixed point | **FAIL** | [report](E0.6/report.md) |

## 핵심 수치와 판정 근거 (seed 평균 [95% CI])

### E0.1 — PASS

- PASS: max_err_f64_cpu — max over 5 runs = 1.998e-15 (threshold 1e-06)

| arm | max_err_f32_metal | max_err_f64_cpu | release_max_diff |
| --- | --- | --- | --- |
| check | 1.12e-05 [5.84e-06, 1.67e-05] | 1.6e-15 [1.38e-15, 1.82e-15] | — |
| drifting | — | — | 0 [0, 0] |

### E0.2 — INCONCLUSIVE

- INCONCLUSIVE: sym/seed0 — self_mean = 0.2881, stable = False, steps = 50000
- INCONCLUSIVE: sym/seed1 — self_mean = 0.3049, stable = False, steps = 50000
- INCONCLUSIVE: sym/seed2 — self_mean = 0.3043, stable = False, steps = 50000
- INCONCLUSIVE: sym/seed3 — self_mean = 0.2946, stable = False, steps = 50000
- INCONCLUSIVE: sym/seed4 — self_mean = 0.2996, stable = False, steps = 50000
- INCONCLUSIVE: sym_hp_x0.5/seed0 — self_mean = 0.5001, stable = False, steps = 50000
- INCONCLUSIVE: sym_hp_x0.5/seed1 — self_mean = 0.5243, stable = False, steps = 50000
- INCONCLUSIVE: sym_hp_x0.5/seed2 — self_mean = 0.5027, stable = False, steps = 50000
- INCONCLUSIVE: sym_hp_x0.5/seed3 — self_mean = 0.5013, stable = False, steps = 50000
- INCONCLUSIVE: sym_hp_x0.5/seed4 — self_mean = 0.5015, stable = False, steps = 50000
- INCONCLUSIVE: sym_hp_x2/seed0 — self_mean = 0.1937, stable = False, steps = 50000
- INCONCLUSIVE: sym_hp_x2/seed1 — self_mean = 0.1978, stable = False, steps = 50000
- … 3 more in the report

| arm | final_rms_v | n_eff_mean | own_dist_mean | self_mean | self_min | steps |
| --- | --- | --- | --- | --- | --- | --- |
| sym | 1.26e-05 [9.16e-06, 1.78e-05] | 16 [16, 16] | 0.76 [0.758, 0.764] | 0.298 [0.293, 0.304] | 0 [0, 0] | 5e+04 [5e+04, 5e+04] |
| sym_hp_x0.5 | 5.58e-06 [3.33e-06, 7.83e-06] | 8.76 [8.76, 8.76] | 0.568 [0.564, 0.57] | 0.506 [0.501, 0.515] | 0 [0, 0] | 5e+04 [5e+04, 5e+04] |
| sym_hp_x2 | 9.55e-06 [8.28e-06, 1.08e-05] | 26.3 [26.3, 26.3] | 0.925 [0.921, 0.931] | 0.192 [0.186, 0.196] | 0 [0, 0] | 5e+04 [5e+04, 5e+04] |
| hq0 | 9.81e-07 [9.8e-07, 9.82e-07] | 16 [16, 16] | 1.04 [1.04, 1.04] | 0.0764 [0.0753, 0.0772] | 0.0445 [0.0406, 0.05] | 4.9e+04 [4.82e+04, 4.96e+04] |
| drifting | 1.11e-06 [1.07e-06, 1.14e-06] | 16 [16, 16] | 0.0445 [0.0441, 0.0449] | 0.951 [0.95, 0.952] | 0.932 [0.928, 0.936] | 5e+04 [5e+04, 5e+04] |

### E0.3 — FAIL

- FAIL: b0:tv_mean — seed mean 0.1469 [0.1446, 0.1491] vs fail_above 0.1

| arm | final_rms_v | n_eff_mean | own_dist_mean | self_mean | steps | tv_max |
| --- | --- | --- | --- | --- | --- | --- |
| b0 | 9.81e-07 [9.8e-07, 9.82e-07] | 16 [16, 16] | 1.04 [1.04, 1.04] | 0.0764 [0.0753, 0.0772] | 4.9e+04 [4.82e+04, 4.96e+04] | 0.21 [0.202, 0.22] |
| drifting | 1.11e-06 [1.07e-06, 1.14e-06] | 16 [16, 16] | 0.0445 [0.0441, 0.0449] | 0.951 [0.95, 0.952] | 5e+04 [5e+04, 5e+04] | 0.901 [0.899, 0.903] |

### E0.4 — PASS

- PASS: self_mean: hq_0.25 − hq_0 — paired diff 0.001514 [0.0002686, 0.002515] (expected increasing)
- PASS: self_mean: hq_0.5 − hq_0.25 — paired diff 0.02117 [0.01973, 0.02351] (expected increasing)
- PASS: self_mean: hq_1 − hq_0.5 — paired diff 0.1992 [0.1919, 0.2054] (expected increasing)
- PASS: tv_mean: hq_0.25 − hq_0 — paired diff 0.01999 [0.01811, 0.02204] (expected increasing)
- PASS: tv_mean: hq_0.5 − hq_0.25 — paired diff 0.1143 [0.1103, 0.1174] (expected increasing)
- PASS: tv_mean: hq_1 − hq_0.5 — paired diff 0.4628 [0.4597, 0.4658] (expected increasing)

| arm | final_rms_v | own_dist_mean | self_mean | steps | tv_mean | wall_time_s |
| --- | --- | --- | --- | --- | --- | --- |
| hq_0 | 9.81e-07 [9.8e-07, 9.82e-07] | 1.04 [1.04, 1.04] | 0.0764 [0.0753, 0.0772] | 4.9e+04 [4.82e+04, 4.96e+04] | 0.147 [0.145, 0.149] | 152 [142, 162] |
| hq_0.25 | 1.88e-06 [1.76e-06, 2e-06] | 1.04 [1.04, 1.04] | 0.0779 [0.0766, 0.0792] | 5e+04 [5e+04, 5e+04] | 0.167 [0.165, 0.169] | 730 [658, 809] |
| hq_0.5 | 6.12e-06 [5.81e-06, 6.47e-06] | 1.02 [1.01, 1.02] | 0.0991 [0.0977, 0.101] | 5e+04 [5e+04, 5e+04] | 0.281 [0.278, 0.284] | 796 [758, 838] |
| hq_1 | 1.26e-05 [9.16e-06, 1.78e-05] | 0.76 [0.758, 0.764] | 0.298 [0.293, 0.304] | 5e+04 [5e+04, 5e+04] | 0.744 [0.743, 0.745] | 570 [444, 696] |
| drifting | 1.11e-06 [1.07e-06, 1.14e-06] | 0.0445 [0.0441, 0.0449] | 0.951 [0.95, 0.952] | 5e+04 [5e+04, 5e+04] | 0.881 [0.88, 0.881] | 185 [174, 191] |

### E0.5 — FAIL

- FAIL: lam_0.5:rel_err_mean — seed mean 3145 [2914, 3382] vs 0.05
- FAIL: lam_0.5:rel_err_var — seed mean 2.895 [2.767, 3.114] vs 0.05
- FAIL: lam_0.75:rel_err_mean — seed mean 4697 [4540, 4831] vs 0.05
- FAIL: lam_0.75:rel_err_var — seed mean 15.94 [15.06, 16.41] vs 0.05
- FAIL: lam_0.9:rel_err_mean — seed mean 1.665e+05 [1.607e+05, 1.712e+05] vs 0.05
- FAIL: lam_0.9:rel_err_var — seed mean 7.393e+04 [7.38e+04, 7.401e+04] vs 0.05

| arm | final_rms_v | rel_err_mean | rel_err_var | steps | var_c0 | wall_time_s |
| --- | --- | --- | --- | --- | --- | --- |
| lam_0 | 9.14e-07 [9.04e-07, 9.23e-07] | 3.8e-05 [3.39e-05, 4.23e-05] | 6.8e-05 [4.69e-05, 8.91e-05] | 1.32e+04 [1.24e+04, 1.42e+04] | 0.25 [0.25, 0.25] | 68.6 [55.4, 82.4] |
| lam_0.25 | 2.86e-06 [2.22e-06, 3.3e-06] | 0.0278 [0.027, 0.0284] | 0.0317 [0.0309, 0.0329] | 5e+04 [5e+04, 5e+04] | 0.462 [0.461, 0.463] | 436 [428, 446] |
| lam_0.5 | 1.1 [0.614, 1.81] | 3.14e+03 [2.91e+03, 3.38e+03] | 2.89 [2.77, 3.11] | 4.17e+04 [2.97e+04, 5e+04] | 3.75 [3.53, 4.03] | 517 [368, 622] |
| lam_0.75 | 1.64e+04 [1.59e+04, 1.69e+04] | 4.7e+03 [4.54e+03, 4.83e+03] | 15.9 [15.1, 16.4] | 50 [50, 50] | 21.3 [20.1, 21.9] | 0.62 [0.613, 0.63] |
| lam_0.9 | 2.53e+06 [2.44e+06, 2.6e+06] | 1.66e+05 [1.61e+05, 1.71e+05] | 7.39e+04 [7.38e+04, 7.4e+04] | 50 [50, 50] | 1.03e+05 [1.03e+05, 1.04e+05] | 0.642 [0.634, 0.653] |
| lam_1.1 | 1.01e+09 [9.71e+08, 1.04e+09] | — | — | 50 [50, 50] | 1.46e+09 [1.38e+09, 1.54e+09] | 0.572 [0.528, 0.614] |
| drifting | — | — | — | 50 [50, 50] | — | 4.4 [4.33, 4.46] |

### E0.5b — PASS

- PASS: lam_0:rel_err_mean — seed mean 8.555e-08 [7.556e-08, 9.553e-08] vs 0.05
- PASS: lam_0:rel_err_var — seed mean 1.316e-07 [1.267e-07, 1.369e-07] vs 0.05
- PASS: lam_0.25:rel_err_mean — seed mean 1.092e-07 [9.279e-08, 1.256e-07] vs 0.05
- PASS: lam_0.25:rel_err_var — seed mean 6.189e-08 [5.254e-08, 7.084e-08] vs 0.05
- PASS: lam_0.5:rel_err_mean — seed mean 1.55e-07 [1.236e-07, 1.863e-07] vs 0.05
- PASS: lam_0.5:rel_err_var — seed mean 3.317e-08 [1.861e-08, 4.732e-08] vs 0.05
- PASS: lam_0.75:rel_err_mean — seed mean 2.069e-07 [1.119e-07, 2.796e-07] vs 0.05
- PASS: lam_0.75:rel_err_var — seed mean 8.758e-08 [6.407e-08, 1.042e-07] vs 0.05
- PASS: lam_0.9:rel_err_mean — seed mean 3.329e-06 [3.22e-06, 3.456e-06] vs 0.05
- PASS: lam_0.9:rel_err_var — seed mean 1.499e-07 [6.057e-08, 2.781e-07] vs 0.05
- PASS: lam_1.1:diverged — 5/5 seeds diverged

| arm | final_rms_v | rel_err_mean | rel_err_var | steps | var_c0 | wall_time_s |
| --- | --- | --- | --- | --- | --- | --- |
| lam_0 | 3.95e-08 [3.92e-08, 3.98e-08] | 8.55e-08 [7.56e-08, 9.55e-08] | 1.32e-07 [1.27e-07, 1.37e-07] | 1.05e+03 [1.05e+03, 1.05e+03] | 0.25 [0.25, 0.25] | 4.28 [4.03, 4.5] |
| lam_0.25 | 8.65e-08 [8.15e-08, 9.45e-08] | 1.09e-07 [9.28e-08, 1.26e-07] | 6.19e-08 [5.25e-08, 7.08e-08] | 1.1e+03 [1.1e+03, 1.1e+03] | 0.434 [0.434, 0.434] | 5.36 [5.21, 5.51] |
| lam_0.5 | 1.39e-07 [1.19e-07, 1.68e-07] | 1.55e-07 [1.24e-07, 1.86e-07] | 3.32e-08 [1.86e-08, 4.73e-08] | 1.15e+03 [1.15e+03, 1.15e+03] | 0.619 [0.619, 0.619] | 5.61 [5.44, 5.79] |
| lam_0.75 | 3.15e-07 [2.89e-07, 3.4e-07] | 2.07e-07 [1.12e-07, 2.8e-07] | 8.76e-08 [6.41e-08, 1.04e-07] | 1.3e+03 [1.3e+03, 1.3e+03] | 0.803 [0.803, 0.803] | 6.33 [6.09, 6.52] |
| lam_0.9 | 1.36e-06 [1.13e-06, 1.6e-06] | 3.33e-06 [3.22e-06, 3.46e-06] | 1.5e-07 [6.06e-08, 2.78e-07] | 5e+04 [5e+04, 5e+04] | 0.914 [0.914, 0.914] | 253 [252, 254] |
| lam_1.1 | 912 [779, 1.11e+03] | — | — | 340 [310, 370] | 62.1 [0.278, 148] | 1.61 [1.46, 1.77] |
| drifting | — | — | — | 50 [50, 50] | — | 4.37 [4.3, 4.45] |

### E0.6 — FAIL

- INCONCLUSIVE: tv_mean: joint_hq0 − sep_hq0 — paired diff 0.02043 [0.01772, 0.02286], margin ±0.02
- FAIL: tv_mean: joint_sym − sep_sym — paired diff -0.0398 [-0.04328, -0.03663], margin ±0.02
- INCONCLUSIVE: self_mean: joint_sym − sep_sym — paired diff -0.02661 [-0.03535, -0.01787], margin ±0.02

| arm | final_rms_v | own_dist_mean | self_mean | steps | tv_mean | wall_time_s |
| --- | --- | --- | --- | --- | --- | --- |
| sep_hq0 | 9.81e-07 [9.8e-07, 9.82e-07] | 1.04 [1.04, 1.04] | 0.0764 [0.0753, 0.0772] | 4.9e+04 [4.82e+04, 4.96e+04] | 0.147 [0.145, 0.149] | 245 [200, 269] |
| joint_hq0 | 9.81e-07 [9.8e-07, 9.82e-07] | 1.04 [1.04, 1.04] | 0.074 [0.0725, 0.0754] | 4.83e+04 [4.77e+04, 4.9e+04] | 0.167 [0.164, 0.169] | 262 [258, 267] |
| sep_sym | 1.26e-05 [9.16e-06, 1.78e-05] | 0.76 [0.758, 0.764] | 0.298 [0.293, 0.304] | 5e+04 [5e+04, 5e+04] | 0.744 [0.743, 0.745] | 859 [827, 890] |
| joint_sym | 1.07e-05 [9.16e-06, 1.19e-05] | 0.784 [0.78, 0.786] | 0.272 [0.267, 0.279] | 5e+04 [5e+04, 5e+04] | 0.704 [0.701, 0.707] | 768 [663, 860] |
| drifting | 1.11e-06 [1.07e-06, 1.14e-06] | 0.0445 [0.0441, 0.0449] | 0.951 [0.95, 0.952] | 5e+04 [5e+04, 5e+04] | 0.881 [0.88, 0.881] | 223 [191, 287] |

## Figures

- E0.1: 
- E0.2: [F0.1](../../figures/L0/E0.2/F0.1/F0.1.png), [F0.5](../../figures/L0/E0.2/F0.5/F0.5.png), [FC.1](../../figures/L0/E0.2/FC.1/FC.1.png), [FC.2](../../figures/L0/E0.2/FC.2/FC.2.png)
- E0.3: [F0.1](../../figures/L0/E0.3/F0.1/F0.1.png), [F0.2](../../figures/L0/E0.3/F0.2/F0.2.png), [FC.1](../../figures/L0/E0.3/FC.1/FC.1.png), [FC.2](../../figures/L0/E0.3/FC.2/FC.2.png)
- E0.4: [F0.3](../../figures/L0/E0.4/F0.3/F0.3.png), [FC.1](../../figures/L0/E0.4/FC.1/FC.1.png)
- E0.5: [F0.4](../../figures/L0/E0.5/F0.4/F0.4.png), [FC.1](../../figures/L0/E0.5/FC.1/FC.1.png)
- E0.5b: [F0.4](../../figures/L0/E0.5b/F0.4/F0.4.png), [FC.1](../../figures/L0/E0.5b/FC.1/FC.1.png)
- E0.6: [F0.3](../../figures/L0/E0.6/F0.3/F0.3.png), [FC.1](../../figures/L0/E0.6/FC.1/FC.1.png)

## 엔지니어 분석 (탐색적, 판정 아님)

[analysis/README.md](analysis/README.md) — 공통 원인(σ > positive 간격), E0.5 해석, L1 전 결정 사항 7개.

## 사람이 적는 부분

### 이상 징후 (사람)

-

### go/no-go (plan §9, 사용자 결정)

-
