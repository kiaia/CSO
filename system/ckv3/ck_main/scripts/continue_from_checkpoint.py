#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Continue execution from a specific historical checkpoint/step.

Unlike resample_prm_from_log.py which only regenerates PRM candidates without execution,
this script actually continues the agent's execution from a specified step.

Usage:
    python continue_from_checkpoint.py \
        --input existing_output.jsonl \
        --output continued_output.jsonl \
        --continue_from_step 3 \
        --strategy replace_from
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional

from ...agents.session import AgentSession
from ...agents.utils import rprint, zwarn, my_json_dumps
from ..process_reward_agent import ProcessRewardCKAgent
from ..agent import CKAgent


def _parse_bool_env(name: str, default: bool) -> bool:
	val = os.getenv(name)
	if val is None:
		return default
	v = str(val).strip().lower()
	if v in {"1", "true", "t", "yes", "y", "on"}:
		return True
	if v in {"0", "false", "f", "no", "n", "off"}:
		return False
	return default


def _parse_int_env(name: str, default: int) -> int:
	val = os.getenv(name)
	if val is None:
		return default
	try:
		return int(str(val).strip())
	except Exception:
		return default


def _parse_float_env(name: str, default: float) -> float:
	val = os.getenv(name)
	if val is None:
		return default
	try:
		return float(str(val).strip())
	except Exception:
		return default


def build_ck_configs_from_env() -> Dict[str, Any]:
	"""Build CK agent configuration from environment variables."""
	llm_url = os.getenv("LLM_URL", "gpt:gpt-4o-mini")
	
	use_process_reward = _parse_bool_env("USE_PROCESS_REWARD", False)
	enable_subagent_prm = _parse_bool_env("ENABLE_SUBAGENT_PRM", False)
	rpm_model_url = os.getenv("RPM_MODEL_URL", "claude:claude-3.7")
	num_action_candidates = _parse_int_env("NUM_ACTION_CANDIDATES", 3)
	sampling_temperature = _parse_float_env("SAMPLING_TEMPERATURE", 0.0)
	min_score_threshold = _parse_float_env("MIN_SCORE_THRESHOLD", 0.3)
	enable_parallel_sampling = _parse_bool_env("ENABLE_PARALLEL_SAMPLING", False)
	enable_parallel_evaluation = _parse_bool_env("ENABLE_PARALLEL_EVALUATION", False)
	rpm_evaluation_timeout = _parse_int_env("RPM_EVALUATION_TIMEOUT", 60)
	prm_usage_strategy = os.getenv("PRM_USAGE_STRATEGY", "always")
	
	enable_planning_prm = _parse_bool_env("ENABLE_PLANNING_PRM", True)
	planning_score_weight = _parse_float_env("PLANNING_SCORE_WEIGHT", 0.7)
	diversity_weight = _parse_float_env("DIVERSITY_WEIGHT", 0.3)
	
	enable_diverse_sampling = _parse_bool_env("ENABLE_DIVERSE_SAMPLING", True)
	sequential_mode = _parse_bool_env("SEQUENTIAL_MODE", True)
	diversity_threshold = _parse_float_env("DIVERSITY_THRESHOLD", 0.4)
	max_sampling_attempts = _parse_int_env("MAX_SAMPLING_ATTEMPTS", 3)
	diversity_prompt_strength = os.getenv("DIVERSITY_PROMPT_STRENGTH", "medium")
	
	enable_adaptive_sampling = _parse_bool_env("ENABLE_ADAPTIVE_SAMPLING", False)
	adaptive_score_threshold = _parse_float_env("ADAPTIVE_SCORE_THRESHOLD", 0.75)
	adaptive_min_candidates = _parse_int_env("ADAPTIVE_MIN_CANDIDATES", 3)
	
	configs: Dict[str, Any] = {
		"model": {"call_target": llm_url},
		"max_steps": 12,
		"process_reward": {
			"enable_process_reward": use_process_reward,
			"enable_subagent_prm": enable_subagent_prm,
			"num_action_candidates": num_action_candidates,
			"enable_parallel_sampling": enable_parallel_sampling,
			"sampling_temperature": sampling_temperature,
			"min_score_threshold": min_score_threshold,
			"prm_usage_strategy": prm_usage_strategy,
			"rpm_model_config": {
				"call_target": rpm_model_url,
				"enable_parallel_evaluation": enable_parallel_evaluation,
				"evaluation_timeout": rpm_evaluation_timeout,
				"enable_planning_prm": enable_planning_prm,
				"planning_score_weight": planning_score_weight,
				"diversity_weight": diversity_weight,
			},
		},
		"diverse_sequential_sampling": {
			"enable_diverse_sampling": enable_diverse_sampling,
			"sequential_mode": sequential_mode,
			"diversity_threshold": diversity_threshold,
			"max_sampling_attempts": max_sampling_attempts,
			"diversity_prompt_strength": diversity_prompt_strength,
			"enable_adaptive_sampling": enable_adaptive_sampling,
			"adaptive_score_threshold": adaptive_score_threshold,
			"adaptive_min_candidates": adaptive_min_candidates,
		},
	}
	return configs


def extract_state_from_steps(steps: List[Dict[str, Any]], step_idx: int) -> Dict[str, Any]:
	"""
	Extract the progress state from a specific step or the last available step before it.
	
	Args:
		steps: List of step dictionaries
		step_idx: The step index to extract state from (or before)
		
	Returns:
		The extracted state dictionary
	"""
	if not steps or step_idx < 0:
		return {}
	
	# Look backwards from step_idx to find the most recent state
	search_range = min(step_idx, len(steps) - 1)
	for i in range(search_range, -1, -1):
		try:
			step = steps[i]
			if "plan" in step and "state" in step["plan"]:
				state = step["plan"]["state"]
				if isinstance(state, dict):
					rprint(f"[continue] Extracted state from step {i}")
					return state
		except Exception as e:
			zwarn(f"[continue] Failed to extract state from step {i}: {e}")
	
	rprint("[continue] No state found in history, using empty state")
	return {}


def continue_from_checkpoint(
	item: Dict[str, Any],
	agent: Any,  # CKAgent or ProcessRewardCKAgent
	continue_from_step: int,
	strategy: str = "replace_from",
	max_additional_steps: Optional[int] = None,
) -> Dict[str, Any]:
	"""
	Continue execution from a historical checkpoint.
	
	Args:
		item: Input item with task and session
		agent: The CK agent to use for continuation
		continue_from_step: Step index to continue from (0-indexed)
		strategy: Continuation strategy:
			- "replace_from": Discard step at continue_from_step and all after, then continue
			- "continue_after": Keep all steps up to and including continue_from_step, then continue
			- "branch_from": Keep steps before continue_from_step, create a new branch
		max_additional_steps: Maximum additional steps to execute (None = use agent default)
		
	Returns:
		Updated item with new session
	"""
	inst = dict(item)
	task = inst.get("task", "")
	
	# Validate input
	session_obj: Optional[Dict[str, Any]] = inst.get("session")
	if not session_obj:
		zwarn(f"[continue] No session found for id={inst.get('id')}, cannot continue")
		return inst
	
	steps = session_obj.get("steps", [])
	if not isinstance(steps, list) or not steps:
		zwarn(f"[continue] Empty steps for id={inst.get('id')}, cannot continue")
		return inst
	
	if continue_from_step < 0 or continue_from_step >= len(steps):
		zwarn(f"[continue] Invalid continue_from_step={continue_from_step} for {len(steps)} steps")
		return inst
	
	# Apply continuation strategy
	if strategy == "replace_from":
		# Keep steps before continue_from_step, discard from that point
		kept_steps = steps[:continue_from_step]
		state_extract_idx = continue_from_step - 1
		rprint(f"[continue] Strategy: replace_from - keeping {len(kept_steps)} steps, replacing from step {continue_from_step}")
	
	elif strategy == "continue_after":
		# Keep all steps up to and including continue_from_step
		kept_steps = steps[:continue_from_step + 1]
		state_extract_idx = continue_from_step
		rprint(f"[continue] Strategy: continue_after - keeping {len(kept_steps)} steps, continuing after step {continue_from_step}")
	
	elif strategy == "branch_from":
		# Keep steps strictly before continue_from_step
		kept_steps = steps[:continue_from_step]
		state_extract_idx = continue_from_step - 1
		rprint(f"[continue] Strategy: branch_from - keeping {len(kept_steps)} steps, branching from step {continue_from_step}")
	
	else:
		zwarn(f"[continue] Unknown strategy '{strategy}', using 'replace_from'")
		kept_steps = steps[:continue_from_step]
		state_extract_idx = continue_from_step - 1
	
	# Extract state from the kept steps
	state = extract_state_from_steps(steps, state_extract_idx)
	
	# Rebuild session with kept steps
	session = AgentSession.init_from_data(
		task=task,
		steps=kept_steps,
		**session_obj.get("info", {})
	)
	
	rprint(f"[continue] Rebuilt session with {len(kept_steps)} steps, state keys: {list(state.keys())}")
	
	# Calculate remaining steps
	if max_additional_steps is not None:
		max_steps = max_additional_steps
	else:
		# Use agent's default but account for steps already taken
		max_steps = agent.max_steps - len(kept_steps)
		if max_steps <= 0:
			rprint(f"[continue] Warning: No remaining steps (already at {len(kept_steps)}/{agent.max_steps})")
			max_steps = 1  # Allow at least one more step
	
	rprint(f"[continue] Continuing execution for up to {max_steps} additional steps...")
	
	# Continue execution - the key is to pass the existing session
	try:
		# Run from the current state
		result_session = agent.run(
			task=task,
			session=session,
			max_steps=max_steps,
		)
		
		# Update instance with new session
		inst["session"] = result_session.to_dict()
		inst["continued_from_step"] = continue_from_step
		inst["continuation_strategy"] = strategy
		
		rprint(f"[continue] Completed continuation, total steps: {result_session.num_of_steps()}")
		
	except Exception as e:
		zwarn(f"[continue] Failed to continue execution: {e}")
		import traceback
		traceback.print_exc()
		# Keep original session on failure
		inst["session"] = session_obj
		inst["continuation_error"] = str(e)
	
	return inst


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Continue agent execution from a specific historical checkpoint.",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  # Replace from step 3 onwards (discard step 3 and after, then continue)
  python continue_from_checkpoint.py --input log.jsonl --output continued.jsonl --continue_from_step 3

  # Continue after step 3 (keep step 3, add new steps after)
  python continue_from_checkpoint.py --input log.jsonl --output continued.jsonl \\
      --continue_from_step 3 --strategy continue_after

  # Branch from step 3 (keep before step 3, create alternative path)
  python continue_from_checkpoint.py --input log.jsonl --output continued.jsonl \\
      --continue_from_step 3 --strategy branch_from

  # Process only specific task IDs
  python continue_from_checkpoint.py --input log.jsonl --output continued.jsonl \\
      --continue_from_step 3 --task_ids task001,task002

  # Limit additional execution steps
  python continue_from_checkpoint.py --input log.jsonl --output continued.jsonl \\
      --continue_from_step 3 --max_additional_steps 5
		"""
	)
	
	parser.add_argument(
		"--input",
		type=str,
		required=True,
		help="Input JSONL file with existing sessions"
	)
	
	parser.add_argument(
		"--output",
		type=str,
		required=True,
		help="Output JSONL file with continued sessions"
	)
	
	parser.add_argument(
		"--continue_from_step",
		type=int,
		required=True,
		help="Step index to continue from (0-indexed)"
	)
	
	parser.add_argument(
		"--strategy",
		type=str,
		choices=["replace_from", "continue_after", "branch_from"],
		default="replace_from",
		help="Continuation strategy (default: replace_from)"
	)
	
	parser.add_argument(
		"--max_additional_steps",
		type=int,
		default=None,
		help="Maximum additional steps to execute (default: agent's remaining steps)"
	)
	
	parser.add_argument(
		"--task_ids",
		type=str,
		default=None,
		help="Comma-separated task IDs to process (default: all)"
	)
	
	parser.add_argument(
		"--limit",
		type=int,
		default=0,
		help="Limit number of items to process (default: 0 = all)"
	)
	
	parser.add_argument(
		"--resume",
		action="store_true",
		help="Resume: skip items whose ids already exist in output"
	)
	
	parser.add_argument(
		"--enable_prm",
		action="store_true",
		help="Enable Process Reward Model for continuation (overrides env)"
	)
	
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	
	# Ensure output directory exists
	os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
	
	# Build configs
	configs = build_ck_configs_from_env()
	
	# Override PRM setting if requested
	if args.enable_prm:
		if "process_reward" in configs:
			configs["process_reward"]["enable_process_reward"] = True
			rprint("[continue] PRM enabled via --enable_prm flag")
	
	# Initialize agent
	use_prm = configs.get("process_reward", {}).get("enable_process_reward", False)
	if use_prm:
		rprint("[continue] Initializing ProcessRewardCKAgent")
		agent = ProcessRewardCKAgent(**configs)
	else:
		rprint("[continue] Initializing standard CKAgent")
		ck_configs = {k: v for k, v in configs.items() if k != "process_reward"}
		agent = CKAgent(**ck_configs)
	
	# Parse task IDs filter
	task_ids_filter = None
	if args.task_ids:
		task_ids_filter = set(tid.strip() for tid in args.task_ids.split(","))
		rprint(f"[continue] Filtering for task IDs: {task_ids_filter}")
	
	# Resume support
	existing_ids = set()
	write_mode = "w"
	if args.resume and os.path.exists(args.output):
		write_mode = "a"
		rprint(f"[continue] Resume enabled, scanning existing output: {args.output}")
		try:
			with open(args.output, "r", encoding="utf-8") as ef:
				for line in ef:
					line = line.strip()
					if not line:
						continue
					try:
						obj = json.loads(line)
						_id = obj.get("id") or obj.get("task_id")
						if _id:
							existing_ids.add(str(_id))
					except Exception:
						continue
			rprint(f"[continue] Found {len(existing_ids)} existing ids")
		except Exception as e:
			zwarn(f"[continue] Failed to read existing output for resume: {e}")
	
	# Process items
	processed = 0
	skipped = 0
	
	with open(args.input, "r", encoding="utf-8") as rf, \
		 open(args.output, write_mode, encoding="utf-8") as wf:
		
		for line_num, line in enumerate(rf, 1):
			line = line.strip()
			if not line:
				continue
			
			try:
				item = json.loads(line)
			except Exception as e:
				zwarn(f"[continue] Failed to parse line {line_num}: {e}")
				continue
			
			item_id = item.get("id") or item.get("task_id")
			
			# Check filters
			if args.resume and item_id and str(item_id) in existing_ids:
				skipped += 1
				continue
			
			if task_ids_filter and item_id not in task_ids_filter:
				skipped += 1
				continue
			
			# Process item
			rprint(f"\n[continue] ===== Processing item {item_id} (line {line_num}) =====")
			
			try:
				new_item = continue_from_checkpoint(
					item=item,
					agent=agent,
					continue_from_step=args.continue_from_step,
					strategy=args.strategy,
					max_additional_steps=args.max_additional_steps,
				)
				
				wf.write(my_json_dumps(new_item, ensure_ascii=False) + "\n")
				wf.flush()  # Ensure written immediately
				processed += 1
				
			except Exception as e:
				zwarn(f"[continue] Failed to process item {item_id}: {e}")
				import traceback
				traceback.print_exc()
			
			# Check limit
			if args.limit > 0 and processed >= args.limit:
				rprint(f"[continue] Reached limit of {args.limit} items")
				break
	
	rprint(f"\n[continue] ===== Summary =====")
	rprint(f"[continue] Processed: {processed} items")
	rprint(f"[continue] Skipped: {skipped} items")
	rprint(f"[continue] Output: {args.output}")


if __name__ == "__main__":
	main()







