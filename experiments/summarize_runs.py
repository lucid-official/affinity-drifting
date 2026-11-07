"""`runs/*/summary.json`을 시드별로 묶어 평균 ± 표준편차로 출력한다.

    python experiments/summarize_runs.py 'runs/s3000_*'
"""

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = ("color", "background", "shape", "joint", "center_std", "coherence", "iou", "area")
SPLITS = ("seen", "unseen_wording", "held_out_combo", "seen/circle", "seen/square", "seen/triangle")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("pattern", nargs="?", default="runs/s3000_*")
    args = parser.parse_args()

    grouped: dict[str, list[dict]] = defaultdict(list)
    for run in sorted(glob.glob(args.pattern)):
        path = Path(run) / "summary.json"
        if path.exists():
            name = re.sub(r"_seed\d+$", "", Path(run).name)
            grouped[name].append(json.loads(path.read_text()))

    for name, rows in grouped.items():
        print(f"{name}  n={len(rows)} {np.mean([row['seconds'] for row in rows]):.0f}s")
        for alpha in ("alpha=1.0", "alpha=2.0"):
            for split in SPLITS:
                cells = [
                    f"{metric[:5]}={np.mean(values):.3f}±{np.std(values):.3f}"
                    for metric in METRICS
                    for values in [[row[alpha][split][metric] for row in rows]]
                ]
                print(f"   {alpha} {split:15s}", " ".join(cells))


if __name__ == "__main__":
    main()
