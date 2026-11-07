# E0.5b — 사전 등록: Prop E 진단 — Gaussian closure

- 작성 2026-11-07 · 실험 엔지니어 Claude · 승인 Chan Lee (docs/deviations.md D13)
- **plan에 없는 진단 실험이다.** E0.5 smoke(seed 0, 2,000스텝)에서 AffD가 λ ≥ 0.5에서 발산한 것을 보고 추가했다.
  E0.5의 판정은 바꾸지 않는다. E0.5b의 결과는 E0.5를 해석하는 데만 쓴다.
- 이 파일과 `experiment.yaml`은 첫 run 전에 커밋하고, 이후 수정하지 않는다.

## 질문

E0.5의 실패가 Prop E의 대수(닫힌 형태) 때문인가, 유한 표본 KDE 꼬리가 만드는 동역학 불안정 때문인가?

## 설정

E0.5와 데이터·arm·seed·η·스텝 상한·σ 규칙·판정 규칙이 모두 같다. 다른 것은 AffD 필드 하나다(`level0.density: gaussian_closure`):

V = σ²[−(x − μ_c)/s̃² + (x − m_c)/(v_c + σ²) + λ (x − m_c′)/(v_c′ + σ²)]

m, v는 매 스텝 particle의 평균과 차원 평균 분산이다. Lemma 1의 V = σ²∇log(p̂/q̂_c) − λσ²∇log q̂_c′에서
p̂ = N(μ_c, s̃²), q̂ = N(m, v + σ²)로 각 KDE를 Gaussian으로 바꾼 것이다. 이 필드의 고정점이 정확히 닫힌 형태이므로,
E0.5b는 "particle 적분기가 그 고정점에 도달하는가"와 "닫힌 형태의 산술"을 확인한다.

## 예측과 반증 기준 (E0.5와 같은 수치)

- λ < 1: 5 seed 평균 `rel_err_mean`, `rel_err_var` ≤ 0.05. 넘으면 **FAIL**.
- λ = 1.1: 5 seed 모두 발산. 아니면 **FAIL**.

## 해석 규칙 (결과 전에 정함)

| E0.5 | E0.5b | 해석 |
| --- | --- | --- |
| FAIL | PASS | 닫힌 형태는 맞고, 실패 원인은 유한 표본 KDE 꼬리의 동역학 불안정이다 |
| FAIL | FAIL | Prop E의 유도 또는 그 구현을 다시 본다 |
| PASS | PASS | 두 형태 모두 예측과 일치 |
| PASS | FAIL | 구현 오류를 의심한다(closure 필드) |

## baseline `drifting` (D11, D14) — 판정 대상 아님

E0.5와 같은 baseline을 같은 seed로 돌린다(raw release, M = 2048). E0.5 smoke에서는 발산했고, 발산하면 그대로 보고한다.

## figure

- FC.1 (모든 arm), F0.4 (모든 arm).

## 알려진 위험

- closure는 particle 분포의 모양을 강제하지 않는다. 평균과 분산만 맞고 분포가 Gaussian이 아닐 수 있다. 판정은 평균과 분산만 본다.
