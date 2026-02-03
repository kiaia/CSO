#!/bin/bash
# Example script for continuing execution from a historical checkpoint
# 
# Usage:
#   bash run_continue_from_checkpoint.sh [input_file] [output_file] [step_num] [strategy]
#
# Examples:
#   # Replace from step 3 (default behavior)
#   bash run_continue_from_checkpoint.sh input.jsonl output.jsonl 3
#
#   # Continue after step 3 (keep step 3, add after it)
#   bash run_continue_from_checkpoint.sh input.jsonl output.jsonl 3 continue_after
#
#   # Branch from step 3 (keep before step 3, try alternative)
#   bash run_continue_from_checkpoint.sh input.jsonl output.jsonl 3 branch_from

set -e  # Exit on error

# Configuration from environment or defaults
export LLM_URL="${LLM_URL:-gpt:gpt-4o}"
export RPM_MODEL_URL="${RPM_MODEL_URL:-claude:claude-3.7}"

# Process Reward Model settings
export USE_PROCESS_REWARD="${USE_PROCESS_REWARD:-false}"
export ENABLE_SUBAGENT_PRM="${ENABLE_SUBAGENT_PRM:-false}"
export NUM_ACTION_CANDIDATES="${NUM_ACTION_CANDIDATES:-3}"
export SAMPLING_TEMPERATURE="${SAMPLING_TEMPERATURE:-0.0}"
export MIN_SCORE_THRESHOLD="${MIN_SCORE_THRESHOLD:-0.3}"

# Planning PRM settings
export ENABLE_PLANNING_PRM="${ENABLE_PLANNING_PRM:-true}"
export PLANNING_SCORE_WEIGHT="${PLANNING_SCORE_WEIGHT:-0.7}"
export DIVERSITY_WEIGHT="${DIVERSITY_WEIGHT:-0.3}"

# Diverse sampling settings
export ENABLE_DIVERSE_SAMPLING="${ENABLE_DIVERSE_SAMPLING:-true}"
export SEQUENTIAL_MODE="${SEQUENTIAL_MODE:-true}"
export DIVERSITY_THRESHOLD="${DIVERSITY_THRESHOLD:-0.4}"
export MAX_SAMPLING_ATTEMPTS="${MAX_SAMPLING_ATTEMPTS:-3}"
export DIVERSITY_PROMPT_STRENGTH="${DIVERSITY_PROMPT_STRENGTH:-medium}"

# Parse command line arguments
INPUT_FILE="${1:-}"
OUTPUT_FILE="${2:-}"
CONTINUE_FROM_STEP="${3:-0}"
STRATEGY="${4:-replace_from}"

# Validate required arguments
if [ -z "$INPUT_FILE" ] || [ -z "$OUTPUT_FILE" ]; then
    echo "Error: Missing required arguments"
    echo ""
    echo "Usage: $0 <input_file> <output_file> <step_num> [strategy]"
    echo ""
    echo "Arguments:"
    echo "  input_file        : Input JSONL file with existing sessions"
    echo "  output_file       : Output JSONL file for continued sessions"
    echo "  step_num          : Step number to continue from (0-indexed)"
    echo "  strategy          : Continuation strategy (replace_from|continue_after|branch_from)"
    echo ""
    echo "Environment Variables:"
    echo "  LLM_URL           : Main model endpoint (default: gpt:gpt-4o)"
    echo "  RPM_MODEL_URL     : Reward model endpoint (default: claude:claude-3.7)"
    echo "  USE_PROCESS_REWARD: Enable PRM (default: false)"
    echo ""
    echo "Examples:"
    echo "  # Replace from step 3 with PRM enabled"
    echo "  USE_PROCESS_REWARD=true $0 input.jsonl output.jsonl 3 replace_from"
    echo ""
    echo "  # Continue after step 5 without PRM"
    echo "  $0 input.jsonl output.jsonl 5 continue_after"
    exit 1
fi

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SYSTEM_DIR="$(dirname "$SCRIPT_DIR")"

echo "========================================="
echo "Continue Execution from Checkpoint"
echo "========================================="
echo "Input file:      $INPUT_FILE"
echo "Output file:     $OUTPUT_FILE"
echo "Continue from:   Step $CONTINUE_FROM_STEP"
echo "Strategy:        $STRATEGY"
echo "Main model:      $LLM_URL"
echo "PRM enabled:     $USE_PROCESS_REWARD"
if [ "$USE_PROCESS_REWARD" = "true" ]; then
    echo "PRM model:       $RPM_MODEL_URL"
    echo "Action samples:  $NUM_ACTION_CANDIDATES"
fi
echo "========================================="
echo ""

# Change to system directory
cd "$SYSTEM_DIR"

# Run the continuation script
python -m ckv3.ck_main.scripts.continue_from_checkpoint \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    --continue_from_step "$CONTINUE_FROM_STEP" \
    --strategy "$STRATEGY" \
    "${@:5}"  # Pass any additional arguments

echo ""
echo "========================================="
echo "Continuation completed!"
echo "Output saved to: $OUTPUT_FILE"
echo "========================================="







