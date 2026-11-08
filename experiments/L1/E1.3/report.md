# E1.3 — AffD restores the mixture and beats single, K1, K2, K3: 결과 보고서

> 자동 생성: `scripts/aggregate.py E1.3` · commit `75d2109e82` · 2026-11-08 13:49 UTC
> 사전 등록: [prereg.md](prereg.md) (결과를 본 뒤 수정 금지) · 판정 규칙: `experiment.yaml`의 `judge:`

## 판정: **PASS**

| 판정 | 검사 | 근거 |
| --- | --- | --- |
| PASS | w2_heldout: AffD / single | ratio 0.3318 (pass ≤ 0.5) |
| PASS | w2_heldout: single − AffD > 0 | paired diff 0.9443 [0.9374, 0.951] |
| PASS | w2_heldout: K1 − AffD > 0 | paired diff 0.868 [0.8402, 0.8882] |
| PASS | w2_heldout: K2 − AffD > 0 | paired diff 0.5959 [0.5884, 0.6031] |
| PASS | w2_heldout: K3 − AffD > 0 | paired diff 0.9446 [0.9358, 0.9532] |

## 수치 (seed 평균, 95% bootstrap CI, 5 seeds)

| arm | w2_heldout | w2_train | mci_heldout | off_manifold_heldout | mode_coverage_heldout | memorization_train | precision_heldout | recall_heldout | coverage_heldout | mmd_heldout |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AffD | 0.4688 [0.4602, 0.4775] | 0.4791 [0.4699, 0.4886] | 0.02572 [0.02434, 0.02733] | 0.279 [0.2702, 0.2906] | 0.9997 [0.9992, 1] | 0.1653 [0.1645, 0.166] | 0.7423 [0.732, 0.7502] | 0.9397 [0.9324, 0.9457] | 0.6787 [0.671, 0.6915] | 0.005342 [0.004988, 0.005696] |
| single | 1.413 [1.411, 1.416] | 1.411 [1.409, 1.414] | 0.9996 [0.9995, 0.9997] | 1 [1, 1] | 0.3508 [0.3451, 0.3568] | 0.004434 [0.002896, 0.005972] | 0.001683 [0.0002731, 0.003093] | 1.526e-06 [0, 4.578e-06] | 1.068e-05 [4.578e-06, 1.831e-05] | 0.8408 [0.8385, 0.8433] |
| K1 | 1.337 [1.311, 1.361] | 1.336 [1.31, 1.36] | 0.9878 [0.9816, 0.9938] | 0.9993 [0.9979, 1] | 0.4849 [0.4333, 0.5409] | 0.009126 [0.00787, 0.01012] | 0.002243 [0.0009232, 0.004413] | 0.007007 [4.578e-06, 0.01596] | 0.0008484 [3.357e-05, 0.002322] | 0.7994 [0.7778, 0.819] |
| K2 | 1.065 [1.063, 1.067] | 1.045 [1.043, 1.047] | -0.2948 [-0.3006, -0.2904] | 0.9221 [0.9214, 0.9231] | 1 [1, 1] | 0.1624 [0.161, 0.1635] | 0.1007 [0.09947, 0.1015] | 0.9739 [0.971, 0.9766] | 0.2269 [0.2244, 0.2291] | 0.08366 [0.08327, 0.08413] |
| K3 | 1.413 [1.411, 1.416] | 1.412 [1.409, 1.414] | 0.9996 [0.9995, 0.9996] | 1 [1, 1] | 0.3633 [0.3547, 0.3732] | 0.002005 [0.001405, 0.002713] | 1.221e-05 [0, 3.662e-05] | 0 [0, 0] | 3.052e-06 [0, 6.104e-06] | 0.8313 [0.8292, 0.8337] |
| drifting | 1.427 [1.426, 1.427] | 1.425 [1.424, 1.426] | 1 [1, 1] | 1 [1, 1] | 0.3385 [0.3367, 0.3406] | 0.005151 [0.003728, 0.006575] | 0.002339 [0.0007767, 0.003902] | 0 [0, 0] | 6.104e-06 [3.052e-06, 7.629e-06] | 0.8458 [0.8434, 0.8486] |

누락 run: 0개

## Figures

- [F1.1](../../../figures/L1/E1.3/F1.1/F1.1.png)
- [F1.2](../../../figures/L1/E1.3/F1.2/F1.2.png)
- [F1.3](../../../figures/L1/E1.3/F1.3/F1.3.png)
- [FC.1](../../../figures/L1/E1.3/FC.1/FC.1.png)
- [FC.2](../../../figures/L1/E1.3/FC.2/FC.2.png)
- [FC.3](../../../figures/L1/E1.3/FC.3/FC.3.png)
- [FC.4](../../../figures/L1/E1.3/FC.4/FC.4.png)

## 이상 징후

- `AffD/seed0`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `AffD/seed1`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `AffD/seed2`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `AffD/seed3`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `AffD/seed4`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `single/seed0`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `single/seed1`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `single/seed2`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `single/seed3`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `single/seed4`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `K1/seed0`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `K1/seed1`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `K1/seed2`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `K1/seed3`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `K1/seed4`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `K2/seed0`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `K2/seed1`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `K2/seed2`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `K2/seed3`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `K2/seed4`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `K3/seed0`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `K3/seed1`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `K3/seed2`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `K3/seed3`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `K3/seed4`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `drifting/seed0`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `drifting/seed1`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `drifting/seed2`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `drifting/seed3`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `drifting/seed4`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)

### 사람이 적는 이상 징후

-

## 해석 (판정 이후, 사전 등록과 구분해서)

-
