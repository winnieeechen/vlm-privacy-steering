#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

GPU_ID="${GPU_ID:-1}"
RUNNER="scripts/14_free/01_run_dual_adaptive_steering.py"
BASE_CSV="outputs/14_free/base/base_freeform_qwen_val_gpt4o_mini.csv"
BASELINE="outputs/14_free/val/14_free_val_a1.0-1.5_c3.0-4.0.csv"
OUT_DIR="outputs/14_free/val_confidence_sweep"
LOG_DIR="outputs/14_free/logs"

mkdir -p "$OUT_DIR" "$LOG_DIR"

if [[ -z "${LOCAL_JUDGE_API_KEY:-}" ]]; then
  echo "LOCAL_JUDGE_API_KEY is not set." >&2
  echo "Export it before starting this sweep." >&2
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

configs=(
  "0.50 0.55"
  "0.45 0.65"
  "0.50 0.65"
  "0.55 0.65"
)

for config in "${configs[@]}"; do
  read -r threshold_a threshold_c <<< "$config"
  tag="ta${threshold_a}_tc${threshold_c}"
  output="$OUT_DIR/14_free_val_${tag}.csv"
  log="$LOG_DIR/val_confidence_${tag}.log"

  echo "===== Starting ${tag} on GPU ${GPU_ID} ====="
  CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 python -u "$RUNNER" \
    --split val \
    --base-csv "$BASE_CSV" \
    --assets outputs/14_free/assets/training_free_assets.pt \
    --vector-dir outputs/14_free/assets/vectors \
    --alpha-a-low 1.0 \
    --alpha-a-high 1.5 \
    --alpha-c-low 3.0 \
    --alpha-c-high 4.0 \
    --min-confidence-a "$threshold_a" \
    --min-confidence-c "$threshold_c" \
    --prototype-blend 0.25 \
    --expert-neighbors 24 \
    --expert-temperature 0.05 \
    --temperature 0.0 \
    --judge-provider local \
    --judge-base-url http://hl279-cmp-01.egr.duke.edu:4000 \
    --judge-model gpt-4o-mini \
    --reuse-compatible-csv "$BASELINE" \
    --output-csv "$output" \
    --resume \
    2>&1 | tee -a "$log"
done

python scripts/14_free/03_summarize_confidence_val.py
