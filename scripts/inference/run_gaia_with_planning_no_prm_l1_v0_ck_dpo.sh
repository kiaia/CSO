#!/bin/bash

# GAIA script using CK model without Process Reward Model (PRM)
# Keeps evaluator/reward on GPT-4.1; disables PRM and planning PRM

# Using CK endpoint for LLM, GPT-4.1 for VLM/evaluator
export LLM_URL=http://your-vllm-endpoint:8081/v1/chat/completions  # Your trained model endpoint
export VLM_URL=http://your-vlm-endpoint:8081/v1/chat/completions  # Vision model endpoint
export DATABRICKS_TOKEN="your_databricks_token_here"
export DATABRICKS_BASE_URL="your_databricks_base_url_here"
export EVALUATOR_LLM=http://your-evaluator-endpoint:8081/v1/chat/completions

# Process Reward Model Configuration - DISABLED
export USE_PROCESS_REWARD=False  # Disable Process Reward Model
export RPM_MODEL_URL=gpt:gpt-4.1  # Reward/evaluator model remains GPT-4.1
export NUM_ACTION_CANDIDATES=3
export SAMPLING_TEMPERATURE=0
export MIN_SCORE_THRESHOLD=0.3
export ENABLE_PARALLEL_SAMPLING=False
export ENABLE_PARALLEL_EVALUATION=False
export RPM_EVALUATION_TIMEOUT=60
export PRM_USAGE_STRATEGY=always

# Planning PRM - DISABLED
export ENABLE_PLANNING_PRM=False
export PLANNING_SCORE_WEIGHT=0.7
export DIVERSITY_WEIGHT=0.3

# Diverse Sequential Sampling - ENABLED (independent of PRM)
export ENABLE_DIVERSE_SAMPLING=True
export SEQUENTIAL_MODE=True
export DIVERSITY_THRESHOLD=0.4
export MAX_SAMPLING_ATTEMPTS=3
export DIVERSITY_PROMPT_STRENGTH="medium"

# Adaptive Sampling - DISABLED
export ENABLE_ADAPTIVE_SAMPLING=False
export ADAPTIVE_SCORE_THRESHOLD=0.75
export ADAPTIVE_MIN_CANDIDATES=3

export ENABLE_SUBAGENT_PRM=False
# Configure Playwright backend
export PLAYWRIGHT_BACKEND=local
export AZURE_OPENAI_API_KEY=your_azure_key_here
export AZURE_OPENAI_ENDPOINT=your_azure_endpoint_here
export AZURE_OPENAI_API_VERSION=2025-01-01-preview

### langchain
# export EVALUATOR_LLM=gpt:gpt-4.1
# export LANGCHAIN_LLM=gpt-4.1
export AZURE_OPENAI_API_VERSION=2025-01-01-preview
export OPENAI_API_TYPE=azure_ai
export AZURE_INFERENCE_ENDPOINT=$AZURE_OPENAI_ENDPOINT
export AZURE_INFERENCE_CREDENTIAL=$AZURE_OPENAI_API_KEY

# Setup search engine
export SEARCH_BACKEND="Google"
export SEARCH_API_KEY="your_google_search_api_key"
export SEARCH_CSE_ID="your_custom_search_engine_id"

# Step 5: run
# Set PYTHONPATH to the root directory of this project
export PYTHONPATH=${PYTHONPATH:-$(cd "$(dirname "$0")/../.." && pwd)}

# Web browser configuration
WEB_DIR=${WEB_SERVICE_DIR:-"${PYTHONPATH}/system/ckv3/ck_web/_web"}
export WEB_SERVICE_FOLDER=$WEB_DIR
WEB_PORT=${WEB_PORT:-3004}
export LISTEN_PORT=$WEB_PORT
export WEB_IP=localhost:${LISTEN_PORT}
lsof -ti tcp:$LISTEN_PORT | xargs kill -9 2>/dev/null || true

### file agent
export MAX_FILE_READ_TOKENS=10000
export MAX_FILE_SCREENSHOT=5

export TOKENIZERS_PARALLELISM=false

# Build Process Reward and feature configuration (PRM disabled)
PROCESS_REWARD_CONFIG="'process_reward': {'enable_process_reward': ${USE_PROCESS_REWARD}, 'enable_subagent_prm': ${ENABLE_SUBAGENT_PRM}, 'num_action_candidates': ${NUM_ACTION_CANDIDATES}, 'enable_parallel_sampling': ${ENABLE_PARALLEL_SAMPLING}, 'sampling_temperature': ${SAMPLING_TEMPERATURE}, 'min_score_threshold': ${MIN_SCORE_THRESHOLD}, 'prm_usage_strategy': '${PRM_USAGE_STRATEGY}', 'rpm_model_config': {'call_target': '${RPM_MODEL_URL}', 'enable_parallel_evaluation': ${ENABLE_PARALLEL_EVALUATION}, 'evaluation_timeout': ${RPM_EVALUATION_TIMEOUT}, 'enable_planning_prm': ${ENABLE_PLANNING_PRM}, 'planning_score_weight': ${PLANNING_SCORE_WEIGHT}, 'diversity_weight': ${DIVERSITY_WEIGHT}}}"

DIVERSE_SAMPLING_CONFIG="'diverse_sequential_sampling': {'enable_diverse_sampling': ${ENABLE_DIVERSE_SAMPLING}, 'sequential_mode': ${SEQUENTIAL_MODE}, 'diversity_threshold': ${DIVERSITY_THRESHOLD}, 'max_sampling_attempts': ${MAX_SAMPLING_ATTEMPTS}, 'diversity_prompt_strength': '${DIVERSITY_PROMPT_STRENGTH}', 'enable_adaptive_sampling': ${ENABLE_ADAPTIVE_SAMPLING}, 'adaptive_score_threshold': ${ADAPTIVE_SCORE_THRESHOLD}, 'adaptive_min_candidates': ${ADAPTIVE_MIN_CANDIDATES}}"

export MAIN_ARGS="{'web_agent': {'max_steps': 20, 'model': {'call_target': '${LLM_URL}'}, 'model_multimodal': {'call_target': '${VLM_URL}'}, 'web_env_kwargs': {'web_ip': '${WEB_IP}', 'web_command': 'cd ${WEB_SERVICE_FOLDER}; LISTEN_PORT=${LISTEN_PORT} npm start'}}, 'file_agent': {'max_steps': 20, 'model': {'call_target': '${LLM_URL}'}, 'model_multimodal': {'call_target': '${VLM_URL}'}}, 'model': {'call_target': '${LLM_URL}'}, 'max_steps': 12}"

echo "=========================================="
echo "Starting GAIA (CK, No PRM)"
echo "=========================================="
echo "PRM disabled; running baseline without PRM-specific configs"
echo "=========================================="

# Set input/output paths
INPUT_FILE=${INPUT_FILE:-"path/to/your/input_data.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"./output"}
OUTPUT_FILE=${OUTPUT_FILE:-"${OUTPUT_DIR}/gaia_dev_output_no_prm_dpo.jsonl"}
LOG_FILE=${LOG_FILE:-"_log_gaia_dev_no_prm_dpo"}

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# Run the main command and redirect both stdout and stderr to the log file
python3 -u -m ckv3.ck_main.main --updates "${MAIN_ARGS}" \
    --inference-time-evaluation-method gpt_judge \
    --max_retry_num 3 \
    --evaluation-metric llm_score_prompt \
    --input ${INPUT_FILE} \
    --output ${OUTPUT_FILE} \
    2>&1 | tee -a ${LOG_FILE}

echo "=========================================="
echo "GAIA execution completed (CK DPO, No PRM)"
echo "Check the output file: ${OUTPUT_FILE}"
echo "Check the log file: ${LOG_FILE}"
echo "=========================================="

# Analyze and check the output
python -m ckv3.ck_main.scripts.analyze -f ${OUTPUT_FILE} -b 0


