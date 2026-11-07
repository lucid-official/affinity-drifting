# Affinity Drifting (AffD) 실험 저장소

## 0. 이 저장소의 목적
Affinity Drifting(AffD)의 핵심 주장: drifting의 attraction을 단일 정답 회귀에서 "의미가 비슷한 조건들의 positive에 대한
mean-shift"로 바꾸면, 조건당 positive가 하나인 데이터에서도 생성기가 조건부 평균이 아닌 조건부 분포를 학습한다.
- 기준 논문: Deng et al., "Generative Modeling via Drifting", arXiv 2602.04770.
- 실험 계획의 단일 진실 원천(source of truth)은 `docs/plan.md`다. 실험 ID(E0.1, E1.3, ...), 통제군(K1–K6),
  Gate(G0–G4), figure ID(FC.*, F0.*, F1.*, ...)는 모두 그 문서의 정의를 따른다.
- 너의 역할은 **실험 엔지니어**다. 연구 방향을 바꾸지 않는다. 계획과 다르게 해야 할 이유가 생기면
  `docs/deviations.md`에 (날짜, 무엇을, 왜, 결과에 미칠 영향)을 적고 **나에게 승인받은 뒤** 진행한다.

## 1. 저장소 구조
affdrift/
  core/kernels.py        # Gaussian / Laplace(명시적 product) kernel, 캡션 kernel w_i(c)
  core/drift.py          # V+, V-, negative 항, normalization. single positive도 같은 함수(h_p→0, one-hot w)
  core/bandwidth.py      # N_eff(c) >= N* 를 만족하는 h_p(c) 탐색(bisection)
  data/particles.py      # Level 0
  data/toy2d.py          # Level 1 (조건부 3-mode GMM)
  data/shapes_unique.py  # Level 2 (clift의 shapes 생성기를 참고해 확장)
  models/generator.py    # MLP f(eps, g(c)); c 양자화 δ, spectral-norm Lipschitz L 옵션
  eval/metrics.py        # W2, MCI, off-manifold, coverage, memorization, TV, bootstrap CI
  eval/figures.py        # docs/plan.md 10장의 figure 전부
configs/base_B0.yaml     # Base config B0 (plan 2.1)
configs/exp/<ID>.yaml    # 각 실험은 B0를 상속하고 Δ만 적는다
probe/                   # 학습 전에 고정한 probe set (조건 16 × eps 시드 16). 생성 후 커밋, 이후 수정 금지
scripts/run.py           # 단일 run
scripts/sweep.py         # arm × seed 실행
scripts/make_figures.py  # run 디렉터리 → figure 세트
scripts/aggregate.py     # seed 집계, paired bootstrap CI, PASS/FAIL 판정
runs/<ID>/<arm>/seed<k>/ # resolved config, config hash, Δ-diff, metrics.json, logs, figs/
results/<ID>.md          # 실험 보고서 (사전 등록 → 결과 → 판정)
tests/

## 2. 단일 요인 attribution 규칙 (절대 어기지 말 것)
1. **Δ 검사.** run.py는 resolved config와 B0의 diff를 자동 계산해 `delta.json`으로 저장한다.
   diff 키가 `configs/exp/<ID>.yaml`에 선언한 `allowed_delta`와 다르면 실행을 거부한다.
   factorial 실험(E1.4, E1.4b)만 두 키를 허용한다.
2. **같은 코드 경로.** single positive는 별도 구현 금지. `h_p → 0`(w를 one-hot)으로 만든다.
   통제군 K1(k_x≡1), K2(무작위 이웃, 같은 N_eff), K3(잡음 single, 분산 매칭), K4(oracle), K5(h_p→∞)도
   같은 `drift.py`의 옵션으로 구현한다.
3. **σ 고정.** k_x의 σ는 B0 초기화 때 한 번 계산해 `sigma.json`으로 저장하고, 같은 실험의 모든 arm이
   그 값을 읽는다. arm마다 다시 추정하지 않는다.
4. **같은 시드, 같은 probe.** 모든 arm은 같은 seed 집합과 같은 probe set을 쓴다. 비교는 paired로 한다.
5. **스케일 매칭.** normalization을 켜고 항별 ‖V‖를 매 스텝 로깅한다.
   Gate G2: 항별 기울기 norm 비가 [0.5, 2]를 벗어나면 run을 `invalid`로 표시하고 집계에서 뺀다.
6. **‖V‖²는 선택 기준이 아니다.** 모델 선택, early stopping, h 튜닝에 ‖V‖²를 쓰지 않는다.
7. **frozen encoder.** ψ와 φ는 `requires_grad=False`를 assert한다.
8. **하이퍼파라미터 공유.** lr, 스텝, batch 등 공통 하이퍼파라미터는 G4(unconditional)에서 한 번 정하고
   모든 arm에 같은 값을 쓴다. arm별 튜닝 금지.
9. **사전 등록.** 실험을 돌리기 전에 `results/<ID>.md`에 plan의 예측과 반증 기준을 그대로 복사하고 커밋한다.
   결과를 본 뒤에는 이 부분을 고치지 않는다. 판정은 PASS / FAIL / INCONCLUSIVE 중 하나로, CI와 함께 적는다.
10. **통계.** 시드: L0–L1은 5개, L2 이상은 3개. 95% paired bootstrap CI(10,000 resample).
    "차이 없음"은 차이의 CI가 사전 정한 동등 범위 안에 있을 때만 주장한다.

## 3. Figure 규칙
- 평가가 끝날 때마다 make_figures.py가 plan 10장의 해당 figure를 자동 생성한다. 손으로 고른 샘플 금지.
- 공통 figure FC.1–FC.4는 모든 run에서 낸다. FC.4(eps 보간 strip)는 반드시 포함한다.
- arm 순서와 색은 전역 고정: single, K1, K2, K3, AffD, K4, K5 (+K6). `eval/figures.py`의 상수 하나로 관리한다.
- 비교 figure마다 arm 라벨을 섞은 blind 버전(`*_blind.png`)과 키 파일(`*_key.json`)을 따로 저장한다.
- 각 figure 파일 옆에 `<figure>.md`로 "plan의 판독 기준"과 "관찰 3줄" 칸을 만든다. 관찰 칸은 비워 둔다(내가 채운다).

## 4. 수식 참조 (구현 기준)
- 캡션 가중치: w_i(c) = exp(-‖ψ(c)-ψ(c_i)‖² / 2h_p²), 변형 (B): w_i(c) = exp(γ·t·⟨ψ(c), φ_img(y_i)⟩)
- attraction: V+(x|c) = Σ_i w_i k_σ(x,y_i)(y_i - x) / Σ_i w_i k_σ(x,y_i)
- repulsion: 같은 형태로 생성 샘플에 대해, 캡션 bandwidth는 h_q (B0: h_q = 0 → 같은 프롬프트의 M개 샘플만, 자기 자신 제외)
- 손실: ‖f(ε,c) - stopgrad(f(ε,c) + V)‖², V = V+ - V- (- λ·V_neg ...)
- N_eff(c) = (Σw)² / Σw², h_p(c)는 N_eff ≥ N* (B0: 16)를 만족하는 최소값을 bisection으로
- Lemma 0 (G2 검사): single positive에서 repulsion을 끄면 기울기가 ‖f - y_i‖²의 기울기와 같아야 한다(오차 < 1e-5)
- Lemma 1 (E0.1): Gaussian kernel이면 V = σ²∇_x[log p̂(x|c) - log q̂(x|c)]. autograd와 비교(오차 < 1e-6)
- Prop E (E0.5): p_c = N(μ1, s²), p_c' = N(μ2, s²), 서로 counterfactual negative λ이면
  평형 q̂_c = N((μ1 - λμ2)/(1 - λ), (1 + λ)·s̃²), s̃² = s² + σ², λ < 1. λ ≥ 1이면 발산.
- normalization: 원 논문(2602.04770)의 feature normalization과 drift normalization 구현을 먼저 확인해
  그대로 따른다. 확인이 안 되면 근사(feature는 평균 쌍거리가 √C가 되게 스케일, 각 V 항은 배치에서
  E‖V‖²/C = 1이 되게 스케일)를 쓰고 `docs/deviations.md`에 기록한다.
- pos/neg softmax: 따로 정규화(B0)와 합쳐서 정규화를 옵션으로 둔다(E0.6에서 비교).

## 5. 기본 데이터 사양 (plan과 충돌하면 plan이 우선)
- Level 0: 조건 K = 32, ψ_k ∈ [0,1] 등간격, 조건당 y_k ∈ R² 하나(고정 시드). particle M = 256, x ← x + η·V,
  ‖V‖ < 1e-6 또는 스텝 상한에서 정지.
- Level 1: c ∈ [0,1]², 학습 쌍 N = 10k(각자 고유 c). p(x|c) = 3-mode GMM,
  중심 각도 2π·c1 + 2πj/3, 반지름 1 + c2, mode 표준편차 0.1, mixture 비율은 c2에 따라 softmax로 변화.
  held-out c 256개, 조건당 평가 샘플 512개. 용량 손잡이: δ ∈ {0, 1/64, 1/16, 1/4}, L ∈ {1, 4, 16, 64}.

## 6. 작업 방식
- 저장소는 `lucid-official/affine-drifting`이다. `main`은 팀원의 브랜치이므로 커밋, merge, rebase, force push를 하지 않는다.
  모든 작업은 `main`과 히스토리를 공유하지 않는 orphan 브랜치 `chan/affd-exp`에서만 한다.
- device는 cuda > mps > cpu 순으로 자동 선택한다(개발은 Apple Silicon Mac, L3 이상은 연구실 CUDA 서버). float32.
- 실험 하나의 순서: 구현 → 단위 테스트 → 1 seed·짧은 스텝 smoke run → 전체 run → aggregate → figures → 보고서.
- **Level이 끝날 때마다 멈추고** 결과 요약(판정 표, 핵심 figure 경로, 이상 징후)을 보여 준다.
  plan 9장의 go/no-go 판단은 내가 한다. 다음 Level로 혼자 넘어가지 않는다.
- PyTorch, type hint, pytest. 전역 상태 금지. 실험 하나당 커밋 하나(메시지에 실험 ID).
- 결과를 보고 예측이나 반증 기준을 고치지 않는다. 실패도 그대로 보고한다.
