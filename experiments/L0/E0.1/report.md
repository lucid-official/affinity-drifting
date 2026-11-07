# E0.1 — V = σ²∇log(p̂/q̂) for a Gaussian kernel: 결과 보고서

> 자동 생성: `scripts/aggregate.py E0.1` · commit `bce86ecdc3` · 2026-11-07 20:07 UTC
> 사전 등록: [prereg.md](prereg.md) (결과를 본 뒤 수정 금지) · 판정 규칙: `experiment.yaml`의 `judge:`

## 판정: **PASS**

| 판정 | 검사 | 근거 |
| --- | --- | --- |
| PASS | max_err_f64_cpu | max over 5 runs = 1.998e-15 (threshold 1e-06) |

## 수치 (seed 평균, 95% bootstrap CI, 5 seeds)

| arm | max_err_f64_cpu | max_err_f32_metal | release_max_diff |
| --- | --- | --- | --- |
| check | 1.599e-15 [1.377e-15, 1.821e-15] | 1.125e-05 [5.838e-06, 1.665e-05] | — |
| drifting | — | — | 0 [0, 0] |

누락 run: 0개

## Figures

- (아직 없음: `scripts/make_figures.py E0.1`)

## 이상 징후

- (자동 감지 없음)

### 사람이 적는 이상 징후

-

## 해석 (판정 이후, 사전 등록과 구분해서)

-
