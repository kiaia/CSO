#!/bin/bash

# Resample PRM candidates per step on an existing GAIA log using Claude 3.7 as the PRM model.
# Other hyperparameters mirror the GAIA L1/PRM scripts; only PRM model is switched to Claude 3.7.

set -euo pipefail

# ===== Model endpoints
export LLM_URL="http://your-vllm-endpoint:8081/v1/chat/completions"
export VLM_URL="http://your-vllm-endpoint:8081/v1/chat/completions" 
export DATABRICKS_TOKEN="your_databricks_token_here"
export DATABRICKS_BASE_URL="your_databricks_base_url_here"

# ===== PRM configuration (same as before, except RPM_MODEL_URL -> Claude 3.7)
export USE_PROCESS_REWARD=${USE_PROCESS_REWARD:-True}
export ENABLE_SUBAGENT_PRM=${ENABLE_SUBAGENT_PRM:-False}
export RPM_MODEL_URL=${RPM_MODEL_URL:-azure_claude:databricks-claude-3-7-sonnet}  # Use Databricks Claude 3.7 for PRM
export NUM_ACTION_CANDIDATES=${NUM_ACTION_CANDIDATES:-3}
export SAMPLING_TEMPERATURE=${SAMPLING_TEMPERATURE:-0}
export MIN_SCORE_THRESHOLD=${MIN_SCORE_THRESHOLD:-0.3}
export ENABLE_PARALLEL_SAMPLING=${ENABLE_PARALLEL_SAMPLING:-False}
export ENABLE_PARALLEL_EVALUATION=${ENABLE_PARALLEL_EVALUATION:-False}
export RPM_EVALUATION_TIMEOUT=${RPM_EVALUATION_TIMEOUT:-60}
export PRM_USAGE_STRATEGY=${PRM_USAGE_STRATEGY:-always}

# Planning PRM
export ENABLE_PLANNING_PRM=${ENABLE_PLANNING_PRM:-True}
export PLANNING_SCORE_WEIGHT=${PLANNING_SCORE_WEIGHT:-0.7}
export DIVERSITY_WEIGHT=${DIVERSITY_WEIGHT:-0.3}

# Diverse Sequential Sampling
export ENABLE_DIVERSE_SAMPLING=${ENABLE_DIVERSE_SAMPLING:-True}
export SEQUENTIAL_MODE=${SEQUENTIAL_MODE:-True}
export DIVERSITY_THRESHOLD=${DIVERSITY_THRESHOLD:-0.4}
export MAX_SAMPLING_ATTEMPTS=${MAX_SAMPLING_ATTEMPTS:-3}
export DIVERSITY_PROMPT_STRENGTH=${DIVERSITY_PROMPT_STRENGTH:-medium}

# Adaptive Sampling
export ENABLE_ADAPTIVE_SAMPLING=${ENABLE_ADAPTIVE_SAMPLING:-False}
export ADAPTIVE_SCORE_THRESHOLD=${ADAPTIVE_SCORE_THRESHOLD:-0.75}
export ADAPTIVE_MIN_CANDIDATES=${ADAPTIVE_MIN_CANDIDATES:-3}
# ===== Azure OpenAI (required for gpt:* routing)
export AZURE_OPENAI_API_KEY=${AZURE_OPENAI_API_KEY:-your_azure_key_here}
export AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT:-your_azure_endpoint_here}
export AZURE_OPENAI_API_VERSION=${AZURE_OPENAI_API_VERSION:-2025-01-01-preview}
export OPENAI_API_TYPE=${OPENAI_API_TYPE:-azure_ai}
export AZURE_INFERENCE_ENDPOINT=$AZURE_OPENAI_ENDPOINT
export AZURE_INFERENCE_CREDENTIAL=$AZURE_OPENAI_API_KEY

# ===== Python path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/../..:${PYTHONPATH:-}"

# ===== Inputs/Outputs
INPUT=${INPUT:-"data/failed_trajectories.jsonl"}  # Failed trajectories from policy model
OUTPUT=${OUTPUT:-"data/prm_resampled.jsonl"}  # PRM resampled output

mkdir -p "$(dirname "$OUTPUT")"

echo "=========================================="
echo "PRM Resample (Claude 3.7) on:"
echo "INPUT:  $INPUT"
echo "OUTPUT: $OUTPUT"
echo "=========================================="
echo "USE_PROCESS_REWARD:      $USE_PROCESS_REWARD"
echo "ENABLE_PLANNING_PRM:     $ENABLE_PLANNING_PRM"
echo "NUM_ACTION_CANDIDATES:   $NUM_ACTION_CANDIDATES"
echo "RPM_MODEL_URL:           $RPM_MODEL_URL"
echo "PRM_USAGE_STRATEGY:      $PRM_USAGE_STRATEGY"
echo "ENABLE_DIVERSE_SAMPLING: $ENABLE_DIVERSE_SAMPLING"
echo "SEQUENTIAL_MODE:         $SEQUENTIAL_MODE"
echo "=========================================="

# ===== PRM prompt override (optional)
# If you want to supply your own custom PRM prompt, set RPM_PROMPT_FILE before running.
export RPM_PROMPT_FILE=${RPM_PROMPT_FILE:-""}
if [[ -f "$RPM_PROMPT_FILE" ]]; then
  echo "[PRM PROMPT] Using RPM_PROMPT_FILE: $RPM_PROMPT_FILE"
else
  echo "[PRM PROMPT] WARNING: RPM_PROMPT_FILE not found: $RPM_PROMPT_FILE (default PRM prompt will be used)"
  unset RPM_PROMPT_FILE
fi

# ===== Run resampler with id-based resume (append new ids)
python3.12 -u -m ckv3.ck_main.scripts.resample_prm_from_log \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --phase both \
  --resume \
  --num_action_candidates "$NUM_ACTION_CANDIDATES" \
  2>&1 | tee -a "$(dirname "$OUTPUT")/_log_resample_prm_part7_claude_ck_v2"

echo "=========================================="
echo "Done. PRM-annotated (Claude 3.7) log at:"
echo "$OUTPUT"
echo "=========================================="


