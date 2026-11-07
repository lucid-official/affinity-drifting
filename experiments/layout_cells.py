"""4×4 칸별 배치 상관. 이미지의 어느 위치에 텍스트 조건이 들어가는지 본다.

`caption_eval`과 같은 시드와 순서로 샘플을 다시 만들므로, 칸별 값의 평균은 `caption_eval.json`의
`layout_corr`(α=1)과 같다. 각 실행에 `layout_cells.json`을, 캡션 검색 기준은 `results/layout_cells_retrieval.json`에 쓴다.
텍스트 뱅크는 `caption_eval`이 만든 캐시를 읽으므로 CLIP을 띄우지 않는다.

    PYTHONPATH=. .venv/bin/python experiments/layout_cells.py runs/flickr_s10000_*
"""

import argparse
import json
from pathlib import Path

import lucid
import numpy as np

from clipdrift.caption_eval import SAMPLES_PER_PROMPT, layout, layout_corr, layout_corr_cells, load_generator
from clipdrift.data import read_manifest
from clipdrift.evaluate import generate
from clipdrift.text_bank import build_text_bank


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--test", type=Path, default=Path("data/flickr30k_32/test.jsonl"))
    parser.add_argument("--device", default="metal")
    args = parser.parse_args()

    config = json.loads((args.runs[0] / "config.json").read_text())
    tokens = config["num_text_tokens"]
    test = read_manifest(args.test)
    bank, _ = build_text_bank(test.captions, num_tokens=tokens, device=args.device)
    target = layout(test.images)
    rows = test.caption_ids
    channels = test.images.shape[1]

    train = read_manifest(Path(config["data"]))
    train_bank, _ = build_text_bank(train.captions, num_tokens=tokens, device=args.device)
    similarity = bank.pooled[rows] @ train_bank.pooled[train.caption_ids].T
    nearest = np.argsort(-similarity, axis=1)[:, :SAMPLES_PER_PROMPT]
    retrieved = layout(train.images[nearest.reshape(-1)]).reshape(len(rows), SAMPLES_PER_PROMPT, -1).mean(1)
    cells = layout_corr_cells(retrieved, target).reshape(channels, 4, 4)
    Path("results/layout_cells_retrieval.json").write_text(json.dumps({"alpha=1.0": cells.tolist()}))
    print(f"retrieval mean={cells.mean():.3f}", flush=True)

    for run in args.runs:
        model = load_generator(run, args.device)
        lucid.manual_seed(0)  # caption_eval.evaluate_run과 같은 시드와 순서
        repeated = np.repeat(rows, SAMPLES_PER_PROMPT)
        samples = generate(model, bank.pooled[repeated], bank.tokens[repeated], alpha=1.0, device=args.device)
        predicted = layout(samples).reshape(len(rows), SAMPLES_PER_PROMPT, -1).mean(1)
        cells = layout_corr_cells(predicted, target).reshape(channels, 4, 4)
        (run / "layout_cells.json").write_text(json.dumps({"alpha=1.0": cells.tolist()}))
        reported = json.loads((run / "caption_eval.json").read_text())["alpha=1.0"]["layout_corr"]
        print(f"{run} mean={layout_corr(predicted, target):.3f} (caption_eval {reported:.3f})", flush=True)


if __name__ == "__main__":
    main()
