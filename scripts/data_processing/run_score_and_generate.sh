#!/bin/bash

# Script to score trajectory steps and generate alternative candidates using PRM

# Set up environment variables
export AZURE_OPENAI_API_KEY="your_azure_key"
export AZURE_OPENAI_ENDPOINT="your_azure_endpoint"
export AZURE_OPENAI_API_VERSION="2025-01-01-preview"

# Configure langchain (optional)
export EVALUATOR_LLM="gpt:gpt-4.1"
export LANGCHAIN_LLM="gpt-4.1"
export OPENAI_API_TYPE="azure_ai"
export AZURE_INFERENCE_ENDPOINT="$AZURE_OPENAI_ENDPOINT"
export AZURE_INFERENCE_CREDENTIAL="$AZURE_OPENAI_API_KEY"

# Set PYTHONPATH
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/../..:${PYTHONPATH:-}"

# Set tokenizer parallelism
export TOKENIZERS_PARALLELISM="false"

# Input and output paths
INPUT_FILE=${INPUT_FILE:-"data/trajectories.jsonl"}
OUTPUT_FILE=${OUTPUT_FILE:-"data/scored_alternatives.jsonl"}

# Number of items to process (optional, comment out to process all)
# MAX_ITEMS=10

# Run the script
cd "$(dirname "$OUTPUT_FILE")"

echo "=========================================="
echo "Scoring Trajectories and Generating Alternatives"
echo "=========================================="
echo "Input file: $INPUT_FILE"
echo "Output file: $OUTPUT_FILE"
echo "=========================================="

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found: $INPUT_FILE"
    exit 1
fi

# Check file size
FILE_SIZE=$(wc -l < "$INPUT_FILE")
echo "Input file has $FILE_SIZE lines"

# Run the Python script
if [ -z "$MAX_ITEMS" ]; then
    python3 score_and_generate_candidates.py \
        --input "$INPUT_FILE" \
        --output "$OUTPUT_FILE" \
        --llm-endpoint "gpt:gpt-4.1" \
        --rpm-endpoint "gpt:gpt-4.1"
else
    python3 score_and_generate_candidates.py \
        --input "$INPUT_FILE" \
        --output "$OUTPUT_FILE" \
        --llm-endpoint "gpt:gpt-4.1" \
        --rpm-endpoint "gpt:gpt-4.1" \
        --max-items "$MAX_ITEMS"
fi

echo "=========================================="
echo "Done! Check output file: $OUTPUT_FILE"
echo "=========================================="

