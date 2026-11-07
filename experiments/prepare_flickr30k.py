"""Flickr30k를 32×32로 줄여 JSONL 매니페스트로 만든다.

    python -I experiments/prepare_flickr30k.py data/raw/flickr30k data/flickr30k_32

입력 폴더에는 Hugging Face `nlphuji/flickr30k`의 `flickr30k-images.zip`과 `flickr_annotations_30k.csv`가
있어야 한다. zip은 풀지 않고 직접 읽는다. 분할은 CSV의 Karpathy 분할(train 29000 / val 1014 / test 1000)을 따른다.

- 이미지는 중앙 정사각형으로 자른 뒤 BICUBIC(Pillow는 축소 시 안티앨리어싱한다)으로 32×32로 줄인다.
  `clipdrift.data.read_manifest`와 같은 방식이고, 32×32 입력에는 그 함수의 크롭·리사이즈가 아무것도 바꾸지 않는다.
- `caption`은 이미지당 첫 번째 캡션 하나다(프롬프트당 이미지 하나라는 아이디어 문서의 전제). 다섯 개 전부는
  `captions`에 남긴다. 캡션 다섯 개를 모두 학습에 쓰면 텍스트 토큰 캐시가 약 5GB가 된다.
"""

import argparse
import csv
import io
import json
import zipfile
from pathlib import Path

from PIL import Image

SPLITS = ("train", "val", "test")


def to_square(image: Image.Image, size: int) -> Image.Image:
    image = image.convert("RGB")
    side = min(image.size)
    left, top = (image.width - side) // 2, (image.height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    return image.resize((size, size), Image.Resampling.BICUBIC)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--size", type=int, default=32)
    args = parser.parse_args()

    with (args.source / "flickr_annotations_30k.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    manifests: dict[str, list[dict]] = {split: [] for split in SPLITS}

    with zipfile.ZipFile(args.source / "flickr30k-images.zip") as archive:
        # zip 안의 경로는 쓰지 않고 이름만으로 찾는다. 출력 파일명은 img_id로 만든다.
        members = {Path(name).name: name for name in archive.namelist() if name.lower().endswith(".jpg")}
        for count, row in enumerate(rows, start=1):
            captions = json.loads(row["raw"])
            name = f"images/{int(row['img_id']):05d}.png"
            with Image.open(io.BytesIO(archive.read(members[row["filename"]]))) as image:
                to_square(image, args.size).save(args.out / name)
            manifests[row["split"]].append({"image": name, "caption": captions[0], "captions": captions})
            if count % 5000 == 0:
                print(f"{count}/{len(rows)}", flush=True)

    for split, items in manifests.items():
        with (args.out / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"{split}: {len(items)}", flush=True)


if __name__ == "__main__":
    main()
