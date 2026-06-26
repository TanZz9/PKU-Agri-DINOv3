#!/usr/bin/env bash
# Evaluate one DirectFinetune (or exported) DINOv3 checkpoint on Agri downstream tasks.
#
# Usage:
#   CHECKPOINT=/path/to/checkpoint-2.pth bash run_dino_cls_single.sh
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON="${PYTHON:-python}"
cd "$(dirname "$0")"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to your DINOv3 .pth file}"
MODEL_ID="${MODEL_ID:-dinov3_vitb16}"
METRICS_OUT="${METRICS_OUT:-results/metrics_dinov3_cls.jsonl}"
SUMMARY_OUT="${SUMMARY_OUT:-results/summary_dinov3_cls.csv}"
DATASETS="${DATASETS:-all}"
BATCH_SIZE="${BATCH_SIZE:-64}"
WORKERS="${WORKERS:-8}"
USE_CACHE="${USE_CACHE:-1}"
REFRESH_CACHE="${REFRESH_CACHE:-0}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}"
  exit 1
fi

CACHE_ARGS=()
[[ "${USE_CACHE}" == "1" ]] && CACHE_ARGS+=(--use-cache)
[[ "${REFRESH_CACHE}" == "1" ]] && CACHE_ARGS+=(--refresh-cache)

name="$(basename "${CHECKPOINT}" .pth)"
echo "DINOv3 linear probe | ${name}"
echo "  checkpoint: ${CHECKPOINT}"
echo "  dinov3 repo: ${DINOV3_REPO:-../dinov3}"
echo "  agri root: ${AGRI_ROOT:-<default>}"

"${PYTHON}" run_eval.py \
  --models "${MODEL_ID}" \
  --datasets "${DATASETS}" \
  --checkpoint "${CHECKPOINT}" \
  --batch-size "${BATCH_SIZE}" \
  --workers "${WORKERS}" \
  --metrics-path "${METRICS_OUT}" \
  "${CACHE_ARGS[@]}"

"${PYTHON}" scripts/summarize_results.py \
  --metrics-path "${METRICS_OUT}" \
  --output "${SUMMARY_OUT}"

echo "Done: ${METRICS_OUT}"
