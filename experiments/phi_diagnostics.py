"""φ 실행 진단. 규칙 판정기가 보지 못하는 두 가지를 잰다.

1. 도형 해독: 생성 이미지에 프롬프트의 도형 정보가 들어 있는가. 규칙 판정기는 이미지 전체를 템플릿과
   맞추므로 도형이 작은 무늬로 나타나면 놓친다. 위치에 둔감한 진폭 스펙트럼에 로지스틱 회귀를 맞추고,
   한 색을 빼고 학습해 그 색에서 평가한다(색 단서 차단). 우연 수준은 0.33이다.
2. 패밀리별 기울기: 픽셀, MAE, CLIP loss 각각의 이미지 공간 기울기 크기.

    python experiments/phi_diagnostics.py --decode runs/a runs/b
    python experiments/phi_diagnostics.py --gradients runs/a
"""

import argparse
import json
from pathlib import Path

import lucid
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from clipdrift.clip_encoder import FrozenCLIP
from clipdrift.data import COLORS, load_dataset
from clipdrift.drift import DEFAULT_TEMPERATURES, feature_drift_loss
from clipdrift.evaluate import EvalPrompt, eval_prompts, generate
from clipdrift.features import pixel_features, subsample_locations
from clipdrift.generator import GeneratorConfig, TextConditionedGenerator
from clipdrift.mae import MAEFeatures, load_mae
from clipdrift.text_bank import TextBank, build_text_bank
from clipdrift.train import to_grouped

DEVICE = "metal"
SAMPLES = 48
GROUP_NEG, GROUP_POS = 16, 64


def load_generator(run: Path) -> TextConditionedGenerator:
    config = GeneratorConfig(**json.loads((run / "generator.json").read_text()))
    model = TextConditionedGenerator(config)
    model.load_state_dict(lucid.load(run / "generator_ema.lucid"))
    return model.to(DEVICE).eval()


def real_images(dataset, prompt: EvalPrompt, count: int, rng: np.random.Generator) -> np.ndarray:
    index = [
        i
        for i, attr in enumerate(dataset.attributes)
        if (attr["color"], attr["shape"], attr["background"])
        == (prompt.color, prompt.shape, prompt.background)
    ]
    return dataset.images[rng.choice(index, count)]


def generated_images(model, bank: TextBank, prompt: EvalPrompt, count: int) -> np.ndarray:
    index = bank.index(prompt.text)
    return generate(
        model,
        np.repeat(bank.pooled[index][None], count, axis=0),
        np.repeat(bank.tokens[index][None], count, axis=0),
        alpha=1.0,
        device=DEVICE,
    )


def decode_shape(images: np.ndarray, shapes: np.ndarray, colors: np.ndarray) -> float:
    gray = images.mean(axis=1)
    gray = gray - gray.mean(axis=(1, 2), keepdims=True)
    spectrum = np.log1p(np.abs(np.fft.rfft2(gray))).reshape(len(images), -1)
    scores = []
    for color in COLORS:
        test = colors == color
        scaler = StandardScaler().fit(spectrum[~test])
        classifier = LogisticRegression(max_iter=2000, C=0.1)
        classifier.fit(scaler.transform(spectrum[~test]), shapes[~test])
        scores.append(classifier.score(scaler.transform(spectrum[test]), shapes[test]))
    return float(np.mean(scores))


def decode_all(runs: list[Path], dataset, bank: TextBank, seen: list[EvalPrompt]) -> None:
    shapes = np.repeat([prompt.shape for prompt in seen], SAMPLES)
    colors = np.repeat([prompt.color for prompt in seen], SAMPLES)
    rng = np.random.default_rng(0)
    images = np.concatenate([real_images(dataset, prompt, SAMPLES, rng) for prompt in seen])
    print(f"real {decode_shape(images, shapes, colors):.3f}", flush=True)
    for run in runs:
        model = load_generator(run)
        lucid.manual_seed(0)
        images = np.concatenate([generated_images(model, bank, prompt, SAMPLES) for prompt in seen])
        print(f"{run} {decode_shape(images, shapes, colors):.3f}", flush=True)


def family_gradient(extractor, generated: np.ndarray, real: np.ndarray, groups: int):
    x = lucid.tensor(generated, device=DEVICE, requires_grad=True)
    features_x = to_grouped(extractor(x), groups, GROUP_NEG)
    with lucid.no_grad():
        features_pos = to_grouped(extractor(lucid.tensor(real, device=DEVICE)), groups, GROUP_POS)
    features_x, features_pos = subsample_locations((features_x, features_pos), 32)
    total = None
    for name, phi_x in features_x.items():
        loss, _ = feature_drift_loss(phi_x, features_pos[name], None, None, DEFAULT_TEMPERATURES)
        total = loss if total is None else total + loss
    total.eval()
    total.backward()
    grad = np.asarray(x.grad.numpy() if hasattr(x.grad, "numpy") else x.grad)
    return float(total.item()), float(np.sqrt((grad**2).sum(axis=(1, 2, 3))).mean()), len(features_x)


def gradients_all(runs: list[Path], dataset, bank: TextBank, seen: list[EvalPrompt], mae: Path) -> None:
    chosen = seen[:: len(seen) // 4][:4]
    rng = np.random.default_rng(0)
    real = np.concatenate([real_images(dataset, prompt, GROUP_POS, rng) for prompt in chosen])
    extractors = {
        "pixel": pixel_features,
        "mae": MAEFeatures(load_mae(mae, DEVICE)),
        "clip": FrozenCLIP(device=DEVICE).features,
    }
    for run in runs:
        model = load_generator(run)
        lucid.manual_seed(0)
        generated = np.concatenate(
            [generated_images(model, bank, prompt, GROUP_NEG) for prompt in chosen]
        )
        print(run, flush=True)
        base = None
        for family, extractor in extractors.items():
            loss, grad, keys = family_gradient(extractor, generated, real, len(chosen))
            base = base or grad
            print(
                f"  {family:6s} keys={keys:3d} loss={loss:8.2f} grad={grad:8.4f} ratio={grad / base:7.1f}",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--gradients", action="store_true")
    parser.add_argument("--mae", type=Path, default=Path("checkpoints/mae_shapes.lucid"))
    args = parser.parse_args()

    config = json.loads((args.runs[0] / "config.json").read_text())
    dataset = load_dataset(config["data"], count=config["dataset_size"], seed=config["seed"])
    prompts = list(dict.fromkeys([*dataset.captions, *[p.text for p in eval_prompts()]]))
    bank, _ = build_text_bank(prompts, num_tokens=config["num_text_tokens"], device=DEVICE)
    seen = [prompt for prompt in eval_prompts() if prompt.split == "seen"]
    if args.decode:
        decode_all(args.runs, dataset, bank, seen)
    if args.gradients:
        gradients_all(args.runs, dataset, bank, seen, args.mae)


if __name__ == "__main__":
    main()
