# clift

CLIP 텍스트 조건을 [Generative Modeling via Drifting](https://arxiv.org/abs/2602.04770)(Deng et al.)에 통합한 구현입니다. [Lucid](https://github.com/lucid-official/lucid) 3.16.3으로 작성했고 Apple Silicon에서 돌아갑니다.

논문은 클래스 라벨만 조건으로 다룹니다. 여기서는 텍스트 임베딩을 생성기 조건과 loss의 positive 선택에 함께 씁니다. 프롬프트 하나에 이미지 하나뿐인 상황에서 텍스트 이웃을 분포처럼 모으는 것이 핵심입니다.

```
k((x,t),(y,s)) = exp(-‖φ(x)-φ(y)‖/τ_x - ‖e(t)-e(s)‖/τ_t)
```

**검증 범위는 합성 데이터 `shapes`와 Flickr30k 32×32입니다.** Flickr 결과는 [results/report.md](results/report.md)에 있습니다. FID는 재지 않았고, CLIP 임베딩 공간의 FD만 쟀습니다.

## 설치

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest
```

Python 3.14 이상이 필요합니다. 첫 학습 때 CLIP ViT-B/32 가중치(577MB)와 토크나이저를 내려받습니다.

## 사용법

```bash
# 1. φ로 쓸 MAE 학습 (shapes 기준 약 13분. checkpoints/에 이미 있음)
python -m clipdrift.mae --steps 2000 --out checkpoints/mae_shapes.lucid

# 2. 생성기 학습
python -m clipdrift.train --mode pseudo --features pixel --out runs/pseudo      # 로드맵 1단계
python -m clipdrift.train --mode joint  --features pixel --out runs/joint       # 로드맵 2단계
python -m clipdrift.train --mode joint  --features pixel,mae --pixel-weight 300 --out runs/x  # 로드맵 3단계

# 3. 분포에 맞는 텍스트 찾기
python -m clipdrift.inverse --checkpoint runs/joint --out results/inverse.json

# 4. 저장된 실행 재평가, 비교 실험 전체 재현
python -m clipdrift.evaluate runs/joint
bash experiments/run_shapes.sh pixel
bash experiments/run_shapes.sh features
python experiments/toy_text_kernel.py
python experiments/toy_phi_collapse.py
```

`.venv/bin/python`으로 실행하세요. 실제 데이터는 `{"image": 경로, "caption": 문장}` 형식의 JSONL을 `--data`로 넘기면 됩니다(경로는 JSONL 파일 기준 상대 경로). 이 경로는 코드만 있고 아직 실행해 보지 않았습니다. MAE도 그 데이터로 다시 학습해야 합니다.

## 구조

| 파일 | 내용 | 근거 |
|---|---|---|
| [clipdrift/drift.py](clipdrift/drift.py) | `compute_V`, 정규화 S·λ, 다중 temperature, joint kernel, CFG 가중치 | 논문 Alg.2, A.6, A.7 |
| [clipdrift/sampling.py](clipdrift/sampling.py) | 텍스트 키 스케일, 의사 클래스, kNN 큐, 텍스트 드롭아웃 | 로드맵 1·2단계 |
| [clipdrift/generator.py](clipdrift/generator.py) | `f(ε, e(t), α)`. Lucid `DiTBlock` 재사용 | 논문 A.2 |
| [clipdrift/clip_encoder.py](clipdrift/clip_encoder.py) | 동결 CLIP: 텍스트 토큰, 이미지 패치 특징 | 로드맵 3단계 |
| [clipdrift/mae.py](clipdrift/mae.py) | ResNet-style MAE (Lucid에 없어 직접 구현) | 논문 A.3 |
| [clipdrift/features.py](clipdrift/features.py) | 다중 스케일·위치 특징 벡터 | 논문 A.5 |
| [clipdrift/train.py](clipdrift/train.py) | 학습 루프 | 논문 A.9 |
| [clipdrift/inverse.py](clipdrift/inverse.py) | 분포 → 텍스트 탐색, textual inversion | 아이디어 문서 §6 |
| [clipdrift/data.py](clipdrift/data.py), [clipdrift/evaluate.py](clipdrift/evaluate.py) | `shapes` 데이터, 규칙 판정기, 평가 | 검증용 |

## 검증 결과

`shapes`는 직접 만든 32×32 합성 데이터입니다. 색 4 × 도형 3 × 배경 2, 이미지당 캡션 1개, 고유 캡션 100개이고, 색-도형 2개 조합(노란 삼각형, 파란 원)은 학습에서 뺐습니다. 정답 속성을 알아 조건 정확도를 규칙으로 잴 수 있습니다. 실제 이미지에서 판정기의 정확도는 1.00입니다. 단순한 데이터이므로 아래 수치로 실제 이미지 품질을 말할 수는 없습니다.

### 텍스트 kernel (toy, 시드 3개)

프롬프트별 조건부 energy distance입니다. 낮을수록 좋습니다.

| 방법 | gmm | ring |
|---|---|---|
| blind (loss에 텍스트 없음) | 2.40 | 1.62 |
| pair (N_pos=1) | 0.033 | 0.084 |
| 의사 클래스 K=8 / 32 | 0.014 / 0.018 | 0.089 / 0.058 |
| kNN + joint τ_t=0.1 | 0.015 | 0.055 |
| kNN만 (τ_t=100) | 0.025 | 0.050 |
| kernel만 τ_t=0.1 | 0.023 | 0.082 |
| kernel만 τ_t=0.5 | 0.014 | 0.214 |
| kernel만 τ_t=2 | 0.013 | 0.506 |
| kernel만 τ_t=100 | 2.23 | 0.949 |

- loss에 텍스트가 들어가야 합니다. blind는 전체 분포만 맞추고 조건은 배우지 못합니다.
- τ_t는 실제로 분포의 입도를 정합니다. kernel만 쓰면 τ_t가 커질수록 blind로 수렴합니다.
- kNN으로 묶으면 τ_t는 거의 영향이 없습니다. soft kernel의 이득은 kNN 없이 쓸 때만 뚜렷합니다.
- 프롬프트당 positive가 1개여도(pair) 생각보다 잘 됩니다. 저차원 toy에서는 생성기의 일반화가 이를 메꿉니다. 이미지에서는 다릅니다(아래).

### φ를 같이 학습하면 붕괴하는가 (toy, 시드 3개)

| φ | cond_ed | raw drift | effective rank |
|---|---|---|---|
| 고정 | 0.013 ± 0.005 | 2.9e-3 | 2.81 |
| φ도 ‖V‖²를 직접 줄임 | 0.144 ± 0.092 | 7.0e-4 | 1.08 |
| φ는 InfoNCE로만 갱신 + EMA | 0.018 ± 0.000 | 3.6e-3 | 5.27 |

공모 붕괴가 재현됩니다. drift는 더 작은데 분포는 약 11배 나쁘고, φ가 한 축으로 눌립니다. contrastive + EMA는 안정적이지만 이 toy에서 고정보다 낫지는 않았습니다.

### shapes 이미지 조건 (픽셀 공간, α=1, 학습에 있던 조합)

| 방식 | 색 | 배경 | 도형 | 연결성 | IoU |
|---|---|---|---|---|---|
| 실제 이미지 | 1.00 | 1.00 | 1.00 | 1.00 | 0.995 |
| blind, 3000스텝 (3시드) | 0.26 | 0.49 | 0.35 ± 0.03 | 0.88 | 0.68 |
| pseudo, 3000스텝 (3시드) | 1.00 | 1.00 | 0.38 ± 0.01 | 0.99 | 0.79 |
| joint, 3000스텝 (3시드) | 1.00 | 1.00 | 0.36 ± 0.01 | 0.99 | 0.78 |
| blind, 6000스텝 (1시드) | 0.29 | 0.47 | 0.35 | 0.96 | 0.74 |
| pseudo, 6000스텝 (1시드) | 1.00 | 1.00 | 0.53 | 0.99 | 0.81 |
| joint, 6000스텝 (1시드) | 1.00 | 1.00 | 0.45 | 0.99 | 0.80 |

도형의 우연 수준은 0.33입니다. 색과 배경은 학습에 없던 조합과 문장 표현에서도 거의 1.00입니다(joint 3000스텝: 0.995 이상). 학습에서 뺀 조합의 도형은 0.35 ± 0.05로 우연 수준이므로, 도형까지 일반화된다고는 말할 수 없습니다.

면적은 텍스트를 잘 따릅니다(6000스텝 joint: 삼각형 0.110, 원 0.166, 사각형 0.221 / 실제 0.102, 0.159, 0.204 / blind는 모두 약 0.18). pseudo와 joint는 3000스텝 3시드에서 차이가 없습니다.

### φ 비교: MAE, CLIP (3000스텝, α=1, 시드 1개)

전부 joint입니다. `해독`은 위치에 둔감한 진폭 스펙트럼에 로지스틱 회귀를 맞춰 프롬프트의 도형을 맞히는 비율입니다. 한 색을 빼고 학습해 그 색에서 평가합니다. 우연 0.33, blind 0.35, 실제 1.00입니다([experiments/phi_diagnostics.py](experiments/phi_diagnostics.py), 결과는 [results/phi_diagnostics.txt](results/phi_diagnostics.txt)). "픽셀만" 행의 판정기 값은 시드 3개 평균이고 해독은 시드 0입니다.

| φ | 색 | 배경 | 도형(판정기) | 도형(해독) | 연결성 | IoU | 위치 std |
|---|---|---|---|---|---|---|---|
| 실제 이미지 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.995 | 0.142 |
| 픽셀만 | 1.00 | 1.00 | 0.36 | 0.52 | 0.99 | 0.78 | 0.134 |
| 픽셀 + MAE | 0.99 | 1.00 | 0.33 | 0.67 | 0.94 | 0.72 | 0.098 |
| 픽셀 + CLIP | 0.64 | 0.50 | 측정 불가 | **0.96** | 측정 불가 | 측정 불가 | 측정 불가 |
| 픽셀 + MAE + CLIP | 0.98 | 1.00 | 0.33 | 0.40 | 0.90 | 0.71 | 0.084 |
| 픽셀 ×300 + MAE | 1.00 | 1.00 | 0.40 | 0.54 | 0.99 | 0.80 | 0.133 |
| 픽셀 ×300 + CLIP | 0.96 | 1.00 | 0.38 | 0.46 | 0.73 | 0.65 | 0.084 |

**이 조건에서 MAE와 CLIP은 도형 학습을 앞당기지 못했습니다.** 해독이 픽셀만 쓴 값보다 뚜렷이 높은 실행은 전부 이미지 품질을 잃었고, 품질을 지킨 실행의 해독은 픽셀만 쓴 값(시드 0, 1에서 0.52, 0.56)과 같은 범위입니다. 판정기 값 0.40은 픽셀만(0.36 ± 0.01)보다 높지만 시드가 1개이고, 학습에서 뺀 조합의 도형은 0.19로 우연 수준 아래입니다.

가중치 1일 때는 픽셀 항이 사실상 꺼져 있었습니다. 이미지 공간 기울기가 φ마다 크게 다릅니다.

| 패밀리 | 키 수 | loss | 기울기 (픽셀 대비) |
|---|---|---|---|
| 픽셀 | 1 | 약 6 | 1 |
| MAE | 63 | 약 400 | 330~1,440배 |
| CLIP | 29 | 약 180 | 320~1,030배 |

- **MAE**: 픽셀 항을 300배 하면 품질 손실이 사라집니다(IoU 0.72 → 0.80, 위치 std 0.098 → 0.133). 그래서 위 사용법 예시에 넣었습니다. 기본값은 논문의 단순 합과 같은 1입니다.
- **CLIP, 가중치 1**: 프롬프트의 도형과 같은 모양의 작은 글리프를 그려 해독이 0.96까지 오릅니다. 그러나 크기가 틀리고 배경이 회색이라 이미지로는 실패입니다. 배경 분리가 안 되어(면적 0.53) 판정기 값은 쓸 수 없습니다.
- **CLIP, 가중치 300**: 색과 배경은 돌아오지만 글리프가 사라지고 해독이 0.46으로 떨어집니다. 도형 정보가 전체 크기로 옮겨 가지 않았고 이미지는 조각납니다(연결성 0.73).
- 픽셀만 쓴 실행도 도형 정보를 판정기 값보다 많이 담고 있습니다(3000스텝 해독 0.52, 6000스텝 0.60).

CLIP 결과를 실제 이미지로 일반화하면 안 됩니다. 32×32를 224로 키워 넣으므로 CLIP의 패치 하나가 원본 약 4.6픽셀에 해당합니다. CLIP이 설계된 해상도와 거리가 먼 조건입니다. 가중치는 300 하나만 시험했습니다.

### 분포 → 텍스트 탐색 (후보 24개)

| 방법 | top-1 (학습 조합 20개) | top-1 (제외 조합 4개) |
|---|---|---|
| CLIP 정렬 점수 + 배경 보정(PMI) | **1.00** | 1.00 |
| CLIP 정렬 점수 | 0.65 | 1.00 |
| softmax 분류 | 0.65 | 1.00 |
| 생성기 + `E‖V‖²` 최소 | 0.60 | 1.00 |
| 생성기 + energy distance 최소 (대조군) | 0.95 | 1.00 |

- PMI 보정은 효과가 뚜렷합니다.
- `E‖V‖²`는 선택 기준으로 약합니다. 분포가 멀면 kernel이 작아져 V도 작아지므로 "drift가 작다"가 "잘 맞는다"를 뜻하지 않습니다.
- textual inversion(임베딩 직접 최적화)은 목표 6개 모두에서 색과 배경을 1.00으로 복원했지만(시작 0.03~0.47), 도형은 복원하지 못했습니다. 생성기 자체의 도형 표현력이 낮은 영향도 있습니다.

## 논문·아이디어 문서와 다른 점

측정 결과에 따라 바꾼 것입니다.

- **whitening 기본 끔** (문서 §4와 다름). 고유 캡션 수가 차원보다 적으면 정단체가 되어 의미 구조가 사라집니다(같은 조합/다른 조합 거리비 0.605 → 0.895, kNN 순도 0.72 → 0.51). 표본이 충분한 toy ring에서도 끄는 쪽이 낫습니다(kernel만 τ_t=0.1: 0.173 → 0.082). 평균 쌍거리를 1로 맞추는 정규화는 유지합니다.
- **입력 x² 항 제외** (논문 A.5와 다름). 기울기가 픽셀 부호를 증폭하고 목표값이 +0.9와 -0.9에서 같아서, 프롬프트와 무관하게 출력 전체가 한 극으로 붕괴했습니다. 800스텝 진단에서 "검은 배경"/"흰 배경" 프롬프트의 평균 밝기가 x²를 넣으면 -0.86/-0.86, 빼면 -0.77/+0.73이었습니다. 논문은 평균이 0 근처인 잠재 공간에서 씁니다.
- **style 임베딩 초기값 0.2, 출력층 비영 초기화**. 전역 노이즈가 약하면 이미지가 패치 단위로 조각납니다(연결성 0.78 → 0.96). DiT처럼 출력층을 0으로 두면 모든 샘플이 한 점에서 출발해 drift도 똑같아집니다.
- **EMA 웜업**. 0.999로 바로 시작하면 1500스텝 뒤에도 초기 가중치가 22% 남습니다.

규모 때문에 줄이거나 뺀 것입니다.

- 픽셀 공간 32×32만 지원합니다(SD-VAE 잠재 공간 없음).
- SwiGLU, RoPE, RMSNorm, QK-Norm을 쓰지 않습니다(Lucid `DiTBlock` 표준 구성).
- MAE 폭 32(논문 256/640), 2000스텝, 분류 미세조정·증강 없음.
- 그룹 4 × N 16 × P 64(논문 64 × 64 × 64), 특징당 위치 32개로 무작위 축소, 생성기 128폭 3층.
- 클래스별 샘플 큐 대신 임의 접근 데이터셋을 씁니다(논문 A.8도 데이터 로더가 더 원칙적이라고 밝힘).

논문 내부 불일치 하나: Table 8은 "τ별 loss 합"이라 쓰고 A.6은 "τ별 field 합"이라 씁니다. 더 자세한 A.6을 따랐습니다.

## 미구현과 미해결

- **로드맵 5단계**(생성 샘플을 CLIP에 되먹임)는 구현하지 않았습니다. **4단계**(CLIP을 contrastive로 갱신)는 toy에서만 검증했습니다.
- **φ의 효과**는 합성 32×32에서 시드 1개로 본 것이 전부입니다. 실제 이미지 해상도와 더 긴 학습에서는 다를 수 있고, CLIP은 그 조건에서 다시 봐야 합니다.
- **그룹별 S·λ 정규화**는 실험 후 코드에서 뺐지만 결론이 나지 않았습니다. 6000스텝 시드 1개에서 joint 도형 정확도가 올랐고(α=1 0.49 vs 0.45, α=2 0.56 vs 0.47) 면적 오차는 커졌습니다. 결과는 `runs/pixel_joint_pergroup`에 있습니다. 다시 볼 가치가 있습니다.
- 패키지 이름은 `clipdrift`로 남아 있습니다.

## 참고: Lucid 3.16.3 버그

`CLIPTokenizer.from_file`은 생성자가 받지 않는 `pre_tokenizer` 인자를 넘겨 항상 `TypeError`가 납니다. 여기서는 `vocab.json`과 `merges.txt`를 직접 읽어 생성자를 호출합니다([clipdrift/clip_encoder.py](clipdrift/clip_encoder.py)). `lucid.models.generative._common._transformers`의 비공개 모듈도 쓰므로 버전을 고정했습니다.
