"""Flickr30k에서 패밀리별(픽셀, MAE, CLIP) 이미지 공간 기울기 비율을 잰다.

phi_diagnostics.family_gradient를 그대로 쓰되, 양성은 텍스트가 가장 가까운 학습 이미지 64장(joint의 kNN과 같은 방식)이다.

    PYTHONPATH=.:experiments .venv/bin/python experiments/flickr_gradients.py runs/flickr_s10000_pixel_joint_seed0 checkpoints/mae_flickr32.lucid
"""

import json
import sys
from pathlib import Path

import lucid
import numpy as np

from clipdrift.clip_encoder import FrozenCLIP
from clipdrift.data import read_manifest
from clipdrift.evaluate import generate
from clipdrift.features import pixel_features
from clipdrift.mae import MAEFeatures, load_mae
from clipdrift.text_bank import build_text_bank
from phi_diagnostics import DEVICE, GROUP_NEG, GROUP_POS, family_gradient, load_generator

run, mae = Path(sys.argv[1]), Path(sys.argv[2])
config = json.loads((run / "config.json").read_text())
train = read_manifest(Path(config["data"]))
clip = FrozenCLIP(device=DEVICE)
bank, _ = build_text_bank(train.captions, num_tokens=config["num_text_tokens"], clip=clip, device=DEVICE)
keys = bank.pooled[train.caption_ids]  # 이미지별 텍스트 키

model = load_generator(run)
extractors = {"pixel": pixel_features, "mae": MAEFeatures(load_mae(mae, DEVICE)), "clip": clip.features}
for trial, start in enumerate((0, 7)):
    chosen = np.arange(start, len(train.images), len(train.images) // 4)[:4]
    real = np.concatenate([train.images[np.argsort(-(keys @ keys[i]))[:GROUP_POS]] for i in chosen])
    lucid.manual_seed(trial)
    generated = np.concatenate(
        [
            generate(
                model,
                np.repeat(keys[i][None], GROUP_NEG, axis=0),
                np.repeat(bank.tokens[train.caption_ids[i]][None], GROUP_NEG, axis=0),
                alpha=1.0,
                device=DEVICE,
            )
            for i in chosen
        ]
    )
    print(f"trial {trial} images {chosen.tolist()}", flush=True)
    base = None
    for family, extractor in extractors.items():
        loss, grad, count = family_gradient(extractor, generated, real, len(chosen))
        base = base or grad
        print(f"  {family:6s} keys={count:3d} loss={loss:8.2f} grad={grad:8.4f} ratio={grad / base:7.1f}", flush=True)
