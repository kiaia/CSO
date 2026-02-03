#!/bin/bash

# Verify high-scoring PRM candidates and generate verified DPO data.
# This script continues execution with high-scoring candidates and only keeps
# DPO pairs where the high-scoring path leads to correct answers.

set -euo pipefail

# ===== Model endpoints
export LLM_URL=${LLM_URL:-"http://your-model-endpoint:8081/v1/chat/completions"}
export VLM_URL=${VLM_URL:-"http://your-vlm-endpoint:8081/v1/chat/completions"}

# ===== Azure OpenAI (if using Azure)
export AZURE_OPENAI_API_KEY=${AZURE_OPENAI_API_KEY:-"your_azure_key"}
export AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT:-"your_azure_endpoint"}
export AZURE_OPENAI_API_VERSION=${AZURE_OPENAI_API_VERSION:-"2025-01-01-preview"}
export OPENAI_API_TYPE=${OPENAI_API_TYPE:-"azure_ai"}

# ===== Python path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/../..:${PYTHONPATH:-}"

# ===== Inputs/Outputs
# Input should be PRM-resampled data (output from run_resample_prm.sh)
INPUT=${INPUT:-"data/prm_resampled.jsonl"}
OUTPUT=${OUTPUT:-"data/verified_cso_dpo.jsonl"}

# ===== DPO filtering thresholds
GOOD_MIN=${GOOD_MIN:-0.6}  # Minimum score for chosen candidate
BAD_MAX=${BAD_MAX:-0.5}    # Maximum score for rejected candidate

# ===== Execution parameters
MAX_ADDITIONAL_STEPS=${MAX_ADDITIONAL_STEPS:-10}  # Max steps for continuation
LIMIT=${LIMIT:-0}  # 0 = process all items

mkdir -p "$(dirname "$OUTPUT")"

echo "=========================================="
echo "Verify and Generate DPO:"
echo "INPUT:  $INPUT"
echo "OUTPUT: $OUTPUT"
echo "=========================================="
echo "GOOD_MIN (chosen threshold):   $GOOD_MIN"
echo "BAD_MAX (rejected threshold):  $BAD_MAX"
echo "MAX_ADDITIONAL_STEPS:          $MAX_ADDITIONAL_STEPS"
echo "LLM_URL:                       $LLM_URL"
echo "=========================================="

# ===== Run verification and DPO generation
python3 verify_and_generate_dpo.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --good-min "$GOOD_MIN" \
  --bad-max "$BAD_MAX" \
  --max-additional-steps "$MAX_ADDITIONAL_STEPS" \
  --phase action \
  --resume \
  --jsonl \
  $([ "$LIMIT" -gt 0 ] && echo "--limit $LIMIT" || echo "") \
  2>&1 | tee -a "$(dirname "$OUTPUT")/_log_verify_dpo"

echo "=========================================="
echo "Done. Verified DPO data at:"
echo "$OUTPUT"
echo "=========================================="

