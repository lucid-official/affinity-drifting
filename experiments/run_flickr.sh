#!/usr/bin/env bash
# Flickr30k 32×32 비교 실험. 하나씩 순서대로 돌린다(18GB 메모리에서 Metal 작업은 동시에 띄우지 않는다).
#
#   python -I experiments/prepare_flickr30k.py data/raw/flickr30k data/flickr30k_32   # 먼저 한 번
#   SEEDS="0 1 2" bash experiments/run_flickr.sh pixel   # blind, pair, pseudo(K=32, 256), joint (φ 없음)
#   python -m clipdrift.mae --data data/flickr30k_32/train.jsonl --out checkpoints/mae_flickr32.lucid --steps 10000
#   bash experiments/run_flickr.sh mae   # pseudo(K=256), joint + 고정 MAE φ (PIXEL_WEIGHT 기본 1000)
#
# 이미 끝난 실행(generator_ema.lucid가 있음)은 건너뛴다. 학습이 모두 끝나면 테스트 캡션 1000개로
# 전부 다시 평가한다(caption_eval.json, samples_test.png).
set -uo pipefail
cd "$(dirname "$0")/.."

STEPS="${STEPS:-10000}"
SEEDS="${SEEDS:-0}"
DATA=data/flickr30k_32
PY=.venv/bin/python
RUNS=()

run() {
  local name="$1" seed="$2"
  shift 2
  local out="runs/flickr_s${STEPS}_${name}_seed${seed}"
  if [ -f "$out/generator_ema.lucid" ]; then
    echo "== $out: done, skipping"
    RUNS+=("$out")
    return
  fi
  mkdir -p "$out"
  echo "== $out: $*"
  if "$PY" -m clipdrift.train --data "$DATA/train.jsonl" --steps "$STEPS" --seed "$seed" \
    --log-every 500 --out "$out" "$@" >"$out/train.log" 2>&1; then
    RUNS+=("$out")
  else
    echo "FAILED: $out"
  fi
}

case "${1:-}" in
  pixel)
    for seed in $SEEDS; do
      run pixel_blind "$seed" --mode blind --features pixel
      run pixel_pair "$seed" --mode pair --features pixel
      run pixel_pseudo "$seed" --mode pseudo --features pixel --num-clusters 32
      run pixel_pseudo256 "$seed" --mode pseudo --features pixel --num-clusters 256
      run pixel_joint "$seed" --mode joint --features pixel
    done
    ;;
  mae)
    # Flickr30k로 학습해 고정한 MAE를 φ로 붙인다. 픽셀 가중치는 기울기 비율에 맞춘다.
    # Flickr의 pixel joint 생성기에서 MAE 기울기는 픽셀의 약 1,800배다(results/phi_diagnostics.txt). shapes에서는
    # MAE를 붙여 학습하면 비율이 0.59배로 줄었으므로 1000을 골랐다. 실제로 이 값으로 학습한 joint 생성기에서 다시
    # 재 보니 비율은 약 2,040배로 줄지 않았다. 즉 1000에서는 MAE 항의 기울기가 픽셀 항의 약 2배다.
    W="${PIXEL_WEIGHT:-1000}"
    for seed in $SEEDS; do
      for mode in pseudo256 joint; do
        cluster=()
        [ "$mode" = pseudo256 ] && cluster=(--mode pseudo --num-clusters 256) || cluster=(--mode joint)
        run "mae_w${W}_${mode}" "$seed" "${cluster[@]}" --features pixel,mae --pixel-weight "$W" \
          --mae-checkpoint checkpoints/mae_flickr32.lucid
      done
    done
    ;;
  *)
    echo "usage: $0 {pixel|mae}" >&2
    exit 2
    ;;
esac

if [ "${#RUNS[@]}" -gt 0 ]; then
  "$PY" -m clipdrift.caption_eval "${RUNS[@]}" --test "$DATA/test.jsonl" 2>&1 | grep -v Warning
fi
