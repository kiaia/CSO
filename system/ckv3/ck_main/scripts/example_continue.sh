#!/bin/bash
# Quick example demonstrating continue_from_checkpoint usage
# This is a minimal working example for testing

set -e

# Configure environment
export LLM_URL="gpt:gpt-4o-mini"
export USE_PROCESS_REWARD=false

# Example 1: Simple replace from step 2
echo "========================================="
echo "Example 1: Replace from step 2"
echo "========================================="
python -m ckv3.ck_main.scripts.continue_from_checkpoint \
    --input example_input.jsonl \
    --output example_output_replace.jsonl \
    --continue_from_step 2 \
    --strategy replace_from \
    --limit 1

echo ""
echo "✓ Example 1 completed"
echo ""

# Example 2: Continue after step 3
echo "========================================="
echo "Example 2: Continue after step 3"
echo "========================================="
python -m ckv3.ck_main.scripts.continue_from_checkpoint \
    --input example_input.jsonl \
    --output example_output_continue.jsonl \
    --continue_from_step 3 \
    --strategy continue_after \
    --max_additional_steps 2 \
    --limit 1

echo ""
echo "✓ Example 2 completed"
echo ""

# Example 3: Branch from step 1 with PRM
echo "========================================="
echo "Example 3: Branch from step 1 (with PRM)"
echo "========================================="
export USE_PROCESS_REWARD=true
export RPM_MODEL_URL="claude:claude-3.7"
export NUM_ACTION_CANDIDATES=3

python -m ckv3.ck_main.scripts.continue_from_checkpoint \
    --input example_input.jsonl \
    --output example_output_branch.jsonl \
    --continue_from_step 1 \
    --strategy branch_from \
    --enable_prm \
    --limit 1

echo ""
echo "✓ Example 3 completed"
echo ""

echo "========================================="
echo "All examples completed successfully!"
echo "========================================="
echo ""
echo "Output files created:"
echo "  - example_output_replace.jsonl   (replace_from strategy)"
echo "  - example_output_continue.jsonl  (continue_after strategy)"
echo "  - example_output_branch.jsonl    (branch_from strategy with PRM)"
echo ""
echo "Compare the results to see how different strategies work!"







