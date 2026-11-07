"""`runs/*/caption_eval.json`을 시드별로 묶어 평균 ± 표준편차로 출력한다.

    python experiments/summarize_flickr.py 'runs/flickr_s10000_*'
"""

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = ("layout_corr", "layout_r_precision", "clip_score", "r_precision", "fd", "diversity")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("pattern", nargs="?", default="runs/flickr_s10000_*")
    args = parser.parse_args()

    grouped: dict[str, list[dict]] = defaultdict(list)
    reference: dict | None = None
    for run in sorted(glob.glob(args.pattern)):
        path = Path(run) / "caption_eval.json"
        if path.exists():
            result = json.loads(path.read_text())
            reference = reference or {key: result[key] for key in ("real", "retrieval") if key in result}
            grouped[re.sub(r"_seed\d+$", "", Path(run).name)].append(result)

    for name, values in (reference or {}).items():
        print(f"{name:28s}", " ".join(f"{k}={v:.3f}" for k, v in values.items()))
    for alpha in ("alpha=1.0", "alpha=2.0"):
        print(alpha)
        for name, rows in grouped.items():
            cells = [
                f"{metric}={np.mean(values):.3f}±{np.std(values):.3f}"
                for metric in METRICS
                for values in [[row[alpha][metric] for row in rows]]
            ]
            print(f"  {name:26s} n={len(rows)}", " ".join(cells))


if __name__ == "__main__":
    main()
