#!/usr/bin/env bash
# shapes 비교 실험을 하나씩 순서대로 돌린다(18GB 메모리에서 Metal 학습은 동시에 띄우지 않는다).
#
#   bash experiments/run_shapes.sh pixel     # blind, pseudo, joint (φ 없음)
#   bash experiments/run_shapes.sh features  # joint에 MAE, CLIP 특징 추가
set -euo pipefail
cd "$(dirname "$0")/.."

STEPS="${STEPS:-3000}"
PREFIX="${PREFIX:-s${STEPS}_}"
PY=.venv/bin/python

run() {
  local name="$1"
  shift
  local out="runs/$PREFIX$name"
  mkdir -p "$out"
  echo "== $name: $*"
  "$PY" -m clipdrift.train --steps "$STEPS" --log-every 500 --sample-every 2000 --out "$out" "$@" \
    >"$out/train.log" 2>&1
}

case "${1:-}" in
  pixel)
    run pixel_blind --mode blind --features pixel
    run pixel_pseudo --mode pseudo --features pixel
    run pixel_joint --mode joint --features pixel
    ;;
  features)
    run mae_joint --mode joint --features pixel,mae
    run clip_joint --mode joint --features pixel,clip
    run mae_clip_joint --mode joint --features pixel,mae,clip
    ;;
  *)
    echo "usage: $0 {pixel|features}" >&2
    exit 2
    ;;
esac
