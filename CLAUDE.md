# Affinity Drifting (AffD) 실험 저장소

## 0. 이 저장소의 목적
Affinity Drifting(AffD)의 핵심 주장: drifting의 attraction을 단일 정답 회귀에서 "의미가 비슷한 조건들의 positive에 대한
mean-shift"로 바꾸면, 조건당 positive가 하나인 데이터에서도 생성기가 조건부 평균이 아닌 조건부 분포를 학습한다.
- 기준 논문: Deng et al., "Generative Modeling via Drifting", arXiv 2602.04770.
- 실험 계획의 단일 진실 원천(source of truth)은 `docs/plan.md`다. 실험 ID(E0.1, E1.3, ...), 통제군(K1–K6),
  Gate(G0–G4), figure ID(FC.*, F0.*, F1.*, ...)는 모두 그 문서의 정의를 따른다.
- 너의 역할은 **실험 엔지니어**다. 연구 방향을 바꾸지 않는다. 계획과 다르게 해야 할 이유가 생기면
  `docs/deviations.md`에 (날짜, 무엇을, 왜, 결과에 미칠 영향)을 적고 **나에게 승인받은 뒤** 진행한다.

## 1. 저장소 구조 (실험 계층 = 디렉터리 계층)
디렉터리는 실험 계층 Level → Experiment → arm → seed를 그대로 따른다. 코드(`affdrift/`)와 기록(`experiments/`)을 섞지 않는다.

affdrift/                         # 라이브러리 (코드만)
  core/kernels.py                 # k_x (Gaussian / Laplace / none=K1), 캡션 kernel w_i(c), K2 shuffle
  core/bandwidth.py               # N_eff(c) >= N* 를 만족하는 h_p(c) 탐색(bisection)
  core/drift.py                   # DriftLoss(= Lucid release loss) ← AffinityDrift(override). single·K1–K5는 옵션
  core/identities.py              # Lemma 1 autograd 기준, Prop E 닫힌 형태
  data/particles.py               # Level 0
  data/toy2d.py                   # Level 1 (조건부 3-mode GMM)
  data/shapes_unique.py           # Level 2 (clift의 shapes 생성기를 참고해 확장)
  models/generator.py             # MLP f(eps, g(c)); c 양자화 δ, spectral-norm Lipschitz L 옵션
  eval/metrics.py, stats.py       # 지표, paired bootstrap CI
  eval/judge.py                   # 사전 등록된 판정 규칙의 실행기 (PASS / FAIL / INCONCLUSIVE)
  eval/figures/                   # style.py(arm 순서·색 상수), catalog.py(plan 10장 판독 기준), writer.py, level0.py, ...
  levels/level0.py, ...           # Level별 runner: resolved config + seed → metrics
  tracking/                       # run 디렉터리, JSONL + wandb 로깅, EXPERIMENTS.md ledger
  config.py, pipeline.py, runtime.py
configs/base_B0.yaml              # Base config B0 (plan 2.1)와 Level별 B0 섹션
experiments/
  <Level>/README.md               # Level 카드: 목적, 실험 목록, Gate, go/no-go 기록
  <Level>/summary.md              # Level 종료 요약 (판정 표, 핵심 figure, 이상 징후)
  <Level>/<ID>/experiment.yaml    # arms, allowed_delta, reference arm, judge, figures — B0 대비 Δ만 적는다
  <Level>/<ID>/prereg.md          # 사전 등록 (plan의 예측·반증 기준 원문 + 운영 정의). 커밋 후 수정 금지
  <Level>/<ID>/sigma.json         # 실험당 한 번 고정한 σ
  <Level>/<ID>/runs/<arm>/seed<k>/        # config.yaml, config_hash.txt, delta.json, env.json, status.json,
                                          # log.jsonl, metrics.json, arrays.npz(git 제외)
  <Level>/<ID>/runs_smoke/<arm>/seed0/    # smoke run (집계에 쓰지 않음)
  <Level>/<ID>/runs_superseded/<D>/       # 승인된 수정(D번호)으로 무효가 된 run. 지우지 않고 옮겨 둔다(집계 안 함)
  <Level>/<ID>/aggregate.json     # 판정과 seed 집계 (자동)
  <Level>/<ID>/report.md          # 결과 보고서 (자동 생성 + 사람이 적는 칸)
figures/                          # 모든 figure는 여기에만. 실험 계층을 그대로 따른다
  <Level>/<ID>/<FIG>/             # 실험 단위: <FIG>.png, <FIG>_blind.png, <FIG>_key.json, <FIG>.md
  <Level>/<ID>/runs/<arm>/seed<k>/<FIG>/  # run 단위 figure
probe/                            # 학습 전에 고정한 probe set. 생성 후 커밋, 이후 수정 금지
scripts/                          # run.py, sweep.py, aggregate.py, make_figures.py, make_probe.py, setup_env.sh
docs/plan.md, docs/deviations.md
EXPERIMENTS.md                    # 실험 노트: 인덱스(자동) + 시간순 로그(append-only)
CHANGELOG.md                      # 코드·config 변경 이력 (Keep a Changelog)
tests/                            # affdrift/ 구조를 그대로 미러링

## 2. 단일 요인 attribution 규칙 (절대 어기지 말 것)
1. **Δ 검사.** run.py는 resolved config와 B0의 diff를 자동 계산해 `delta.json`으로 저장한다.
   diff 키가 `experiments/<Level>/<ID>/experiment.yaml`에 선언한 `allowed_delta`와 다르면 실행을 거부한다.
   또 arm의 reference arm 대비 바뀐 손잡이는 하나여야 한다. factorial 실험(E1.4, E1.4b)만 `max_knobs: 2`로 두 키를 허용한다.
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
9. **사전 등록.** 실험을 돌리기 전에 `experiments/<Level>/<ID>/prereg.md`에 plan의 예측과 반증 기준을 그대로 복사하고,
   판정 규칙(`experiment.yaml`의 `judge:`)과 함께 커밋한다. `prereg.md`나 `experiment.yaml`이 커밋되지 않았거나 HEAD와 다르면
   run·sweep은 실행을 거부한다(smoke 포함).
   결과를 본 뒤에는 이 부분을 고치지 않는다. 판정은 PASS / FAIL / INCONCLUSIVE 중 하나로, CI와 함께 적는다.
10. **통계.** 시드: L0–L1은 5개, L2 이상은 3개. 95% paired bootstrap CI(10,000 resample).
    "차이 없음"은 차이의 CI가 사전 정한 동등 범위 안에 있을 때만 주장한다.
11. **원 논문 Drifting baseline.** 모든 실험은 아무리 작은 toy라도 원 논문 Drifting(Lucid release의 drift loss를 그대로 쓰는
    `DriftLoss`, class-conditional positive, 캡션 빌려오기 없음)을 arm `drifting`(`field: release`)으로 같은 seed·probe로 돌린다.
    sweep은 이 arm이 없으면 실행을 거부한다. baseline은 plan의 예측을 판정하는 대상이 아니라 같이 보고하는 기준점이며,
    Level 공통 설정(예: L0의 D4)만 똑같이 적용한다.

## 3. Figure 규칙
- 평가가 끝날 때마다 make_figures.py가 plan 10장의 해당 figure를 자동 생성한다. 손으로 고른 샘플 금지.
- figure는 루트의 `figures/<Level>/<ID>/` 아래에만 둔다. `experiments/`나 run 디렉터리 안에 figure를 만들지 않는다.
- 공통 figure FC.1–FC.4는 모든 run에서 낸다. FC.4(eps 보간 strip)는 반드시 포함한다.
- arm 순서와 색은 전역 고정: drifting(baseline), single, K1, K2, K3, AffD, K4, K5 (+K6). `eval/figures/style.py`의 상수 하나로 관리한다.
- 모든 비교 figure에 baseline `drifting`을 항상 같이 그린다(검은색). baseline이 정의되지 않는 figure는 `<figure>.md`에 이유를 적는다.
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
- 저장소는 `lucid-official/affinity-drifting`이다. `main`은 팀원의 브랜치이므로 커밋, merge, rebase, force push를 하지 않는다.
  모든 작업은 `main`과 히스토리를 공유하지 않는 orphan 브랜치 `affinity-drifting`에서만 한다. 다른 사람의 브랜치는 건드리지 않는다.
- device는 `metal` > `cpu` 순으로 자동 선택한다(Lucid는 이 두 이름만 받는다. 개발 머신은 M4 Max 36GB Mac Studio). float32.
  float64가 필요한 수치 검증만 cpu에서 한다(Metal은 float64 미지원). L3 이상의 CUDA 서버 사용은 L3 전에 따로 정한다.
- 실험 하나의 순서: 구현 → 단위 테스트 → 1 seed·짧은 스텝 smoke run → 전체 run → aggregate → figures → 보고서.
- **Level이 끝날 때마다 멈추고** 결과 요약(판정 표, 핵심 figure 경로, 이상 징후)을 보여 준다.
  plan 9장의 go/no-go 판단은 내가 한다. 다음 Level로 혼자 넘어가지 않는다.
- Lucid(`lucid-dl==3.16.4`), type hint, pytest. 전역 상태 금지. 실험마다 커밋은 사전 등록 하나, 결과 하나(메시지에 실험 ID).
- 결과를 보고 예측이나 반증 기준을 고치지 않는다. 실패도 그대로 보고한다.

## 7. 코드 규칙 (strict)
1. **프레임워크는 Lucid.** `lucid-dl==3.16.4`만 쓴다. PyTorch·JAX를 import하지 않는다. NumPy/SciPy는 host 쪽 정적 계산
   (캡션 가중치, bisection, 지표)에만 쓴다.
2. **Lucid 구현체를 상속한다.** Lucid에 있는 것(Drifting의 loss·generator·feature encoder·queue 등)을 그대로 쓸 수 없으면
   이 저장소 안에 새 클래스를 만들되, 반드시 Lucid 구현체를 상속하고 바꿔야 하는 부분만 override한다(`@override` 표기).
   Lucid가 함수로만 제공하는 것은 그 함수들을 단계별 메서드로 그대로 위임하는 기반 클래스로 감싸고, 그 클래스가 Lucid
   원본과 비트 단위로 같다는 테스트를 둔다(예: `DriftLoss` ↔ `lucid...drifting._model._drift_loss`). 복붙 재구현 금지.
3. **이름에 `Lucid`를 넣지 않는다.** 클래스·함수·모듈 이름은 하는 일로 짓는다(`DriftLoss`, `AffinityDrift`).
4. **Metal은 lazy(MLX)다.** 반복 루프에서는 결과를 평가한다. 텐서 하나는 `t.eval()`(예: `loss.eval()`),
   여러 개는 `lucid.eval(a, b, ...)`로 한 번에. 손실만이 아니라 갱신된 파라미터, EMA, 옵티마이저 상태까지 평가해야 한다
   (손실만 평가하다 G4가 1.8만 스텝에서 Metal 자원 한도에 걸렸다). particle 적분은 매 스텝, 학습 루프는 고정 간격
   `train.eval_every`(기본 10스텝; 매 스텝 11.3ms → 10스텝 3.7ms)로 평가한다.
5. **`lucid.compile`을 먼저 시도한다.** 반복되는 스텝 함수는 compile하되, 시작할 때 eager와 결과가 같은지 확인하고
   다르거나 실패하면 eager로 돌린 뒤 그 사실을 run 기록(`metrics.json`의 `notes`)에 남긴다.
6. **최신 Python(3.14).** PEP 695 `type` 별칭과 제네릭 문법, `match`, `StrEnum`, `@dataclass(frozen=True, slots=True, kw_only=True)`,
   `typing.override`/`Self`, walrus, `pathlib`을 쓴다. `from __future__` 불필요. `Any`는 경계(설정·JSON)에서만.
7. **`__pycache__` 금지.** `scripts/setup_env.sh`가 venv에 `sys.dont_write_bytecode = True`를 거는 `.pth`를 설치한다.
   pytest는 `-p no:cacheprovider`. 테스트는 이 설정이 꺼져 있으면 시작하지 않는다. 생긴 캐시는 즉시 지운다.
8. **테스트.** 새 수식·옵션마다 단위 테스트를 같이 커밋한다. 수식 검증(Lemma 0/1, release 동등성)은 허용 오차를 테스트에 명시한다.

## 8. 기록·로깅·figure 관리 (strict)
1. **모든 run은 자기 디렉터리를 갖는다.** `experiments/<Level>/<ID>/runs/<arm>/seed<k>/`에 1장의 파일 전부가 없으면 그 run은
   없는 것이다. 손으로 만든 결과 파일, 디렉터리 밖의 결과는 인정하지 않는다. complete run은 `--force` 없이 덮어쓰지 않는다.
2. **로그.** 진실 원천은 `log.jsonl`과 `metrics.json`이다. wandb(project `affinity-drifting`, group = 실험 ID, job_type = arm)는
   거울이다. 기본 `AFFD_WANDB=offline`: 로컬에만 쌓이고, 업로드(`wandb sync`)는 내가 정할 때만 한다.
3. **상태.** `status.json`은 running / complete / failed / invalid 중 하나다. 실패도 지우지 않고 이유(traceback)와 함께 남긴다.
   invalid run은 집계에서 빠지지만 보고서의 이상 징후에 나온다.
4. **EXPERIMENTS.md.** 인덱스(자동 재생성)와 로그(append-only)로 이뤄진다. 사전 등록, smoke, sweep, aggregate, figure 생성은
   스크립트가 자동으로 항목을 붙인다. 그 밖의 결정·관찰·문제는 사람이(또는 너가) 날짜, 실험 ID, commit과 함께 손으로 붙인다.
   기존 항목은 고치지 않는다. 정정은 새 항목으로 쓴다.
5. **CHANGELOG.md.** 코드, config(B0 포함), 디렉터리 규약, 규칙이 바뀌면 같은 커밋에 Keep a Changelog 형식으로 적는다.
   결과에 영향을 줄 수 있는 변경은 영향받는 실험 ID와 `docs/deviations.md` 항목을 링크한다.
6. **deviations.md.** plan과 다르게 하는 모든 것은 D번호로 (날짜, 무엇을, 왜, 결과에 미칠 영향, 승인 여부)를 적는다.
   승인 전에는 그 변경으로 실험을 돌리지 않는다. 코드와 config 주석은 D번호로 이 문서를 가리킨다.
7. **figure.** figure는 `make_figures.py`만 만들고, 루트 `figures/`에만 둔다. 실험 단위는 `figures/<Level>/<ID>/<FIG>/`,
   run 단위는 `figures/<Level>/<ID>/runs/<arm>/seed<k>/<FIG>/`. 경로는 `Experiment.figure_dir`·`run_figure_dir` 하나로만 정한다.
   모든 figure 옆의 `<FIG>.md`에 원천 run과 config hash, plan의 판독 기준 원문, 관찰 칸을 둔다. 다시 만들 때
   `## 관찰` 아래는 보존된다. arm 비교 figure는 blind 버전과 key를 같이 낸다. 색·순서는 `eval/figures/style.py`에서만 정한다.
8. **보고서.** `report.md`는 `aggregate.py`가 판정·수치·figure 목록·자동 이상 징후를 쓰고, 사람이 적는 이상 징후와 해석 칸을 남긴다.
   해석은 판정과 섞지 않는다. Level이 끝나면 `experiments/<Level>/summary.md`를 쓰고 멈춘다.
9. **git.** 커밋 단위: 인프라/코드 변경 → 실험별 사전 등록 → 실험별 결과(runs의 json·figure·report). `arrays.npz`와 `wandb/`는
   커밋하지 않는다. 커밋 메시지는 실험 ID로 시작한다(예: `E0.3: pre-register`, `E0.3: results (PASS)`).

