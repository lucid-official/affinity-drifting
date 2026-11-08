# E1.2 — single positive collapses to the conditional mean; the oracle does not: 결과 보고서

> 자동 생성: `scripts/aggregate.py E1.2` · commit `1cbf7ca489` · 2026-11-08 12:16 UTC
> 사전 등록: [prereg.md](prereg.md) (결과를 본 뒤 수정 금지) · 판정 규칙: `experiment.yaml`의 `judge:`

## 판정: **PASS**

| 판정 | 검사 | 근거 |
| --- | --- | --- |
| PASS | single:mci_heldout | seed mean 0.9996 [0.9995, 0.9997] (pass ≥ 0.5, fail < 0.3) |
| PASS | off_manifold_heldout: single − K4 > 0 | paired diff 0.8199 [0.8134, 0.8264] |
| PASS | K4:mci_heldout | seed mean 0.09065 [0.08467, 0.09446] (pass ≤ 0.1) |
| PASS | K4:w2_heldout ≤ 2.0×w2_datadata_heldout | 0.5028 vs 2.0×0.3891 |

## 수치 (seed 평균, 95% bootstrap CI, 5 seeds)

| arm | mci_heldout | off_manifold_heldout | w2_heldout | w2_datadata_heldout | mode_coverage_heldout | mci_train | precision_heldout | recall_heldout | coverage_heldout | mmd_heldout |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single | 0.9996 [0.9995, 0.9997] | 1 [1, 1] | 1.413 [1.411, 1.416] | 0.3891 [0.3891, 0.3891] | 0.3508 [0.3451, 0.3568] | 0.9996 [0.9995, 0.9997] | 0.001683 [0.0002731, 0.003093] | 1.526e-06 [0, 4.578e-06] | 1.068e-05 [4.578e-06, 1.831e-05] | 0.8408 [0.8385, 0.8433] |
| K4 | 0.09065 [0.08467, 0.09446] | 0.1801 [0.1736, 0.1866] | 0.5028 [0.4929, 0.5102] | 0.3891 [0.3891, 0.3891] | 0.9773 [0.975, 0.9794] | 0.09714 [0.0927, 0.1012] | 0.8314 [0.8265, 0.8363] | 0.7794 [0.7585, 0.8007] | 0.5005 [0.4877, 0.5123] | 0.009931 [0.008248, 0.01113] |
| drifting | 1 [1, 1] | 1 [1, 1] | 1.427 [1.426, 1.427] | 0.3891 [0.3891, 0.3891] | 0.3385 [0.3367, 0.3406] | 1 [1, 1] | 0.002339 [0.0007767, 0.003902] | 0 [0, 0] | 6.104e-06 [3.052e-06, 7.629e-06] | 0.8458 [0.8434, 0.8486] |

누락 run: 0개

## Figures

- [F1.1](../../../figures/L1/E1.2/F1.1/F1.1.png)
- [FC.1](../../../figures/L1/E1.2/FC.1/FC.1.png)
- [FC.3](../../../figures/L1/E1.2/FC.3/FC.3.png)
- [FC.4](../../../figures/L1/E1.2/FC.4/FC.4.png)

## 이상 징후

- `single/seed0`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `single/seed1`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `single/seed2`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `single/seed3`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `single/seed4`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `K4/seed0`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `K4/seed1`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `K4/seed2`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `K4/seed3`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `K4/seed4`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `drifting/seed0`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `drifting/seed1`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `drifting/seed2`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `drifting/seed3`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)
- `drifting/seed4`: step cap reached without a stable ‖V‖ < tol (final RMS‖V‖ = nan)

### 사람이 적는 이상 징후

-

## 해석 (판정 이후, 사전 등록과 구분해서)

-
