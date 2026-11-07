# E0.6 — joint vs separate softmax share the fixed point: 결과 보고서

> 자동 생성: `scripts/aggregate.py E0.6` · commit `bce86ecdc3` · 2026-11-07 20:07 UTC
> 사전 등록: [prereg.md](prereg.md) (결과를 본 뒤 수정 금지) · 판정 규칙: `experiment.yaml`의 `judge:`

## 판정: **FAIL**

| 판정 | 검사 | 근거 |
| --- | --- | --- |
| INCONCLUSIVE | tv_mean: joint_hq0 − sep_hq0 | paired diff 0.02043 [0.01772, 0.02286], margin ±0.02 |
| PASS | self_mean: joint_hq0 − sep_hq0 | paired diff -0.002344 [-0.003149, -0.001587], margin ±0.02 |
| FAIL | tv_mean: joint_sym − sep_sym | paired diff -0.0398 [-0.04328, -0.03663], margin ±0.02 |
| INCONCLUSIVE | self_mean: joint_sym − sep_sym | paired diff -0.02661 [-0.03535, -0.01787], margin ±0.02 |

## 수치 (seed 평균, 95% bootstrap CI, 5 seeds)

| arm | tv_mean | self_mean | own_dist_mean | steps | final_rms_v | wall_time_s |
| --- | --- | --- | --- | --- | --- | --- |
| sep_hq0 | 0.1469 [0.1446, 0.1491] | 0.07639 [0.07532, 0.07722] | 1.037 [1.037, 1.037] | 4.899e+04 [4.816e+04, 4.965e+04] | 9.809e-07 [9.799e-07, 9.821e-07] | 244.7 [199.8, 269] |
| joint_hq0 | 0.1673 [0.1645, 0.1695] | 0.07405 [0.07246, 0.07542] | 1.037 [1.037, 1.037] | 4.834e+04 [4.772e+04, 4.896e+04] | 9.814e-07 [9.805e-07, 9.824e-07] | 262.1 [257.5, 266.9] |
| sep_sym | 0.7439 [0.7429, 0.7449] | 0.2983 [0.2926, 0.3036] | 0.7604 [0.7575, 0.7635] | 5e+04 [5e+04, 5e+04] | 1.261e-05 [9.162e-06, 1.779e-05] | 858.8 [827.4, 890.2] |
| joint_sym | 0.7041 [0.7013, 0.707] | 0.2717 [0.2669, 0.2789] | 0.7836 [0.7805, 0.7864] | 5e+04 [5e+04, 5e+04] | 1.073e-05 [9.161e-06, 1.188e-05] | 768.5 [662.8, 859.5] |
| drifting | 0.8808 [0.8802, 0.8813] | 0.9511 [0.9503, 0.9517] | 0.04447 [0.0441, 0.04492] | 5e+04 [5e+04, 5e+04] | 1.11e-06 [1.071e-06, 1.143e-06] | 222.8 [190.6, 287.1] |

누락 run: 0개

## Figures

- [F0.3](../../../figures/L0/E0.6/F0.3/F0.3.png)
- [FC.1](../../../figures/L0/E0.6/FC.1/FC.1.png)

## 이상 징후

- `sep_sym/seed0`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 8.39e-06)
- `sep_sym/seed1`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 2.28e-05)
- `sep_sym/seed2`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 1.13e-05)
- `sep_sym/seed3`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 1.12e-05)
- `sep_sym/seed4`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 9.35e-06)
- `joint_sym/seed0`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 7.80e-06)
- `joint_sym/seed1`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 1.07e-05)
- `joint_sym/seed2`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 1.09e-05)
- `joint_sym/seed3`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 1.17e-05)
- `joint_sym/seed4`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 1.25e-05)
- `drifting/seed0`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 1.05e-06)
- `drifting/seed1`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 1.16e-06)
- `drifting/seed2`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 1.13e-06)
- `drifting/seed3`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 1.13e-06)
- `drifting/seed4`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = 1.09e-06)

### 사람이 적는 이상 징후

-

## 해석 (판정 이후, 사전 등록과 구분해서)

-
