#!/usr/bin/env bash
# Extract Agri-DINOv3 features from one image or a folder.
#
# Usage:
#   CHECKPOINT=/path/to/checkpoint-2.pth INPUT=/path/to/image.jpg bash run_infer_single.sh
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON="${PYTHON:-python}"
cd "$(dirname "$0")"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to your DINOv3 .pth file}"
INPUT="${INPUT:?Set INPUT to an image file or image directory}"
OUTPUT="${OUTPUT:-results/features.npz}"
MODEL_ID="${MODEL_ID:-dinov3_vitb16}"
BATCH_SIZE="${BATCH_SIZE:-32}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}"
  exit 1
fi

if [[ ! -e "${INPUT}" ]]; then
  echo "Input not found: ${INPUT}"
  exit 1
fi

echo "Agri-DINOv3 inference"
echo "  checkpoint: ${CHECKPOINT}"
echo "  input: ${INPUT}"
echo "  output: ${OUTPUT}"

"${PYTHON}" infer_features.py \
  --input "${INPUT}" \
  --checkpoint "${CHECKPOINT}" \
  --model-id "${MODEL_ID}" \
  --output "${OUTPUT}" \
  --batch-size "${BATCH_SIZE}"

echo "Done: ${OUTPUT}"
