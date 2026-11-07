#!/usr/bin/env bash
# shapes 비교 실험을 하나씩 순서대로 돌린다(18GB 메모리에서 Metal 학습은 동시에 띄우지 않는다).
#
#   bash experiments/run_shapes.sh pixel     # blind, pseudo, joint (φ 없음), 시드 3개
#   bash experiments/run_shapes.sh features  # joint에 MAE, CLIP 특징 추가, 시드 1개 (오래 걸림)
#
# 실행 하나가 실패해도(예: 메모리 부족) 나머지는 계속 돌리고, 실패한 이름을 출력한다.
set -uo pipefail
cd "$(dirname "$0")/.."

STEPS="${STEPS:-3000}"
PY=.venv/bin/python

run() {
  local name="$1" seed="$2"
  shift 2
  local out="runs/s${STEPS}_${name}_seed${seed}"
  mkdir -p "$out"
  echo "== $out: $*"
  "$PY" -m clipdrift.train --steps "$STEPS" --seed "$seed" --log-every 500 \
    --sample-every 1000 --out "$out" "$@" >"$out/train.log" 2>&1 || echo "FAILED: $out"
}

case "${1:-}" in
  pixel)
    for seed in 0 1 2; do
      run pixel_blind "$seed" --mode blind --features pixel
      run pixel_pseudo "$seed" --mode pseudo --features pixel
      run pixel_joint "$seed" --mode joint --features pixel
    done
    ;;
  features)
    run mae_joint 0 --mode joint --features pixel,mae
    run clip_joint 0 --mode joint --features pixel,clip
    run mae_clip_joint 0 --mode joint --features pixel,mae,clip
    ;;
  *)
    echo "usage: $0 {pixel|features}" >&2
    exit 2
    ;;
esac
