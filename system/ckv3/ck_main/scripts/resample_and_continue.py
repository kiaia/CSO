#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resample from historical nodes and continue execution to completion.

Unlike resample_prm_from_log.py which only samples without execution,
and continue_from_checkpoint.py which continues without resampling,
this script combines both:
1. Resample PRM candidates at a specific historical step
2. Select the best candidate
3. Continue execution from there to completion
4. Check if the final result is correct
5. Store the result in standard CK format

Usage:
    python resample_and_continue.py \
        --input existing_output.jsonl \
        --output resampled_and_continued.jsonl \
        --resample_step 2 \
        --phase action
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from ...agents.session import AgentSession
from ...agents.utils import rprint, zwarn, my_json_dumps
from ..process_reward_agent import ProcessRewardCKAgent


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
	
	use_process_reward = _parse_bool_env("USE_PROCESS_REWARD", True)
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
	"""Extract the progress state from a specific step or the last available step before it."""
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
					rprint(f"[resample_continue] Extracted state from step {i}")
					return state
		except Exception as e:
			zwarn(f"[resample_continue] Failed to extract state from step {i}: {e}")
	
	rprint("[resample_continue] No state found in history, using empty state")
	return {}


def _prepare_common_input_kwargs_for(ck: ProcessRewardCKAgent, session: AgentSession, state: Dict[str, Any]) -> Dict[str, Any]:
	"""Use the agent's helper to prepare inputs exactly as in normal run"""
	return ck._prepare_common_input_kwargs(session, state)  # type: ignore[attr-defined]


def resample_and_select_best(
	ck: ProcessRewardCKAgent,
	session: AgentSession,
	state: Dict[str, Any],
	phase: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Dict[str, Any]]:
	"""
	Resample candidates at the current step and select the best one.
	
	Returns:
		(best_plan, best_action, prm_metadata)
		- best_plan: Best planning candidate if phase includes planning
		- best_action: Best action candidate if phase includes action
		- prm_metadata: Metadata about the resampling process
	"""
	do_planning = phase in ("planning", "both")
	do_action = phase in ("action", "both")
	
	best_plan: Optional[Dict[str, Any]] = None
	best_action: Optional[Dict[str, Any]] = None
	prm_metadata: Dict[str, Any] = {}
	
	# Add a placeholder current step for template preparation
	session.add_step({"step_idx": session.num_of_steps()})
	input_kwargs = _prepare_common_input_kwargs_for(ck, session, state)
	
	# Resample Planning candidates
	if do_planning and ck.enable_planning_prm:
		try:
			rprint("[resample_continue] Resampling planning candidates...")
			plan_candidates = ck._generate_plan_candidates(input_kwargs, session)  # type: ignore[attr-defined]
			
			if len(plan_candidates) >= 2:
				scores, evals = ck._evaluate_plan_candidates(session, state, plan_candidates)  # type: ignore[attr-defined]
				
				# Select best
				best_idx = max(range(len(scores)), key=lambda i: scores[i]) if scores else 0
				best_plan = plan_candidates[best_idx]
				
				# Safe copy of candidates
				safe_candidates = []
				for cand in plan_candidates:
					if isinstance(cand, dict):
						safe = {"thought": cand.get("thought", ""), "code": cand.get("code", "")}
						if "diversity_info" in cand:
							safe["diversity_info"] = cand["diversity_info"]
						safe_candidates.append(safe)
					else:
						safe_candidates.append({"thought": str(cand), "code": ""})
				
				prm_metadata["planning"] = {
					"all_candidates": safe_candidates,
					"scores": scores,
					"selected_idx": best_idx,
					"rpm_evaluations": evals,
					"num_candidates": len(plan_candidates),
				}
				
				rprint(f"[resample_continue] Selected planning candidate {best_idx} with score {scores[best_idx]:.3f}")
			elif len(plan_candidates) == 1:
				best_plan = plan_candidates[0]
				rprint("[resample_continue] Only 1 planning candidate, using it")
		except Exception as e:
			zwarn(f"[resample_continue] Planning resample failed: {e}")
	
	# Resample Action candidates
	if do_action:
		try:
			rprint("[resample_continue] Resampling action candidates...")
			
			# Build action kwargs
			action_kwargs = input_kwargs.copy()
			try:
				import json as _json
				action_kwargs["state"] = _json.dumps(state, ensure_ascii=False, indent=2)
			except Exception:
				action_kwargs["state"] = "{}"
			
			action_candidates = ck._generate_action_candidates(action_kwargs, session)  # type: ignore[attr-defined]
			
			if len(action_candidates) >= 2:
				scores, evals = ck._evaluate_action_candidates(session, state, action_candidates)  # type: ignore[attr-defined]
				
				# Select best
				best_idx = max(range(len(scores)), key=lambda i: scores[i]) if scores else 0
				best_action = action_candidates[best_idx]
				
				# Safe copy of candidates
				safe_candidates = []
				for cand in action_candidates:
					if isinstance(cand, dict):
						safe = {"thought": cand.get("thought", ""), "code": cand.get("code", "")}
						if "diversity_info" in cand:
							safe["diversity_info"] = cand["diversity_info"]
						safe_candidates.append(safe)
					else:
						safe_candidates.append({"thought": str(cand), "code": ""})
				
				prm_metadata["action"] = {
					"all_candidates": safe_candidates,
					"scores": scores,
					"selected_idx": best_idx,
					"rpm_evaluations": evals,
					"num_candidates": len(action_candidates),
				}
				
				rprint(f"[resample_continue] Selected action candidate {best_idx} with score {scores[best_idx]:.3f}")
			elif len(action_candidates) == 1:
				best_action = action_candidates[0]
				rprint("[resample_continue] Only 1 action candidate, using it")
		except Exception as e:
			zwarn(f"[resample_continue] Action resample failed: {e}")
	
	# Remove the placeholder step
	try:
		if session.num_of_steps() > 0:
			session.steps.pop()
	except Exception:
		pass
	
	return best_plan, best_action, prm_metadata


def check_result_correctness(
	item: Dict[str, Any],
	session: AgentSession,
) -> Dict[str, Any]:
	"""
	Check if the final result is correct by comparing with ground truth.
	
	Returns:
		Dictionary with correctness information
	"""
	result = {
		"is_correct": False,
		"predicted_answer": None,
		"ground_truth": None,
		"check_method": "unknown",
	}
	
	# Extract predicted answer from final output
	final_step = None
	if session.steps:
		for step in reversed(session.steps):
			if "action" in step and step.get("action"):
				final_step = step
				break
	
	if final_step:
		action = final_step.get("action", {})
		# Try to extract answer from observation or code
		observation = action.get("observation", "")
		code = action.get("code", "")
		
		# Look for stop() or final answer patterns
		if isinstance(observation, str):
			if "Final Answer:" in observation:
				parts = observation.split("Final Answer:")
				if len(parts) > 1:
					result["predicted_answer"] = parts[1].strip()
			elif "Answer:" in observation:
				parts = observation.split("Answer:")
				if len(parts) > 1:
					result["predicted_answer"] = parts[1].strip()
		
		if result["predicted_answer"] is None and isinstance(code, str):
			if "stop(" in code:
				# Extract from stop(answer)
				try:
					start = code.index("stop(") + 5
					end = code.index(")", start)
					result["predicted_answer"] = code[start:end].strip().strip('"').strip("'")
				except:
					pass
	
	# Extract ground truth
	ground_truth = item.get("answer") or item.get("ground_truth") or item.get("expected_answer")
	result["ground_truth"] = ground_truth
	
	# Compare
	if result["predicted_answer"] is not None and ground_truth is not None:
		pred = str(result["predicted_answer"]).strip().lower()
		truth = str(ground_truth).strip().lower()
		result["is_correct"] = (pred == truth)
		result["check_method"] = "exact_match"
		
		# Try numeric comparison if exact match fails
		if not result["is_correct"]:
			try:
				pred_num = float(pred.replace(",", ""))
				truth_num = float(truth.replace(",", ""))
				result["is_correct"] = abs(pred_num - truth_num) < 1e-6
				result["check_method"] = "numeric_match"
			except:
				pass
	
	rprint(f"[resample_continue] Correctness check: {result['is_correct']} "
	       f"(predicted={result['predicted_answer']}, ground_truth={result['ground_truth']})")
	
	return result


def resample_and_continue_item(
	item: Dict[str, Any],
	ck: ProcessRewardCKAgent,
	resample_step: int,
	phase: str,
	max_additional_steps: Optional[int] = None,
) -> Dict[str, Any]:
	"""
	Resample at a specific step and continue execution to completion.
	
	Args:
		item: Input item with task and session
		ck: The CK agent to use
		resample_step: Step index to resample at (0-indexed)
		phase: Which phase to resample ("planning", "action", or "both")
		max_additional_steps: Maximum additional steps to execute
		
	Returns:
		Updated item with new session and correctness check
	"""
	inst = dict(item)
	task = inst.get("task", "")
	
	# Validate input
	session_obj: Optional[Dict[str, Any]] = inst.get("session")
	if not session_obj:
		zwarn(f"[resample_continue] No session found for id={inst.get('id')}")
		return inst
	
	steps = session_obj.get("steps", [])
	if not isinstance(steps, list) or not steps:
		zwarn(f"[resample_continue] Empty steps for id={inst.get('id')}")
		return inst
	
	if resample_step < 0 or resample_step >= len(steps):
		zwarn(f"[resample_continue] Invalid resample_step={resample_step} for {len(steps)} steps")
		return inst
	
	# Keep steps before resample_step
	kept_steps = steps[:resample_step]
	state = extract_state_from_steps(steps, resample_step - 1)
	
	rprint(f"[resample_continue] Keeping {len(kept_steps)} steps, resampling at step {resample_step}")
	
	# Rebuild session with kept steps
	session = AgentSession.init_from_data(
		task=task,
		steps=kept_steps,
		**session_obj.get("info", {})
	)
	
	# Resample and select best candidate at this step
	best_plan, best_action, prm_metadata = resample_and_select_best(
		ck=ck,
		session=session,
		state=state,
		phase=phase,
	)
	
	# If we have a best candidate, we need to "inject" it into the continuation
	# We do this by manually executing one step with the selected candidate
	if best_action or best_plan:
		rprint("[resample_continue] Executing resampled step...")
		try:
			# Create a new step
			new_step = {"step_idx": session.num_of_steps()}
			session.add_step(new_step)
			
			# Add plan if we resampled it
			if best_plan:
				# Execute plan code if present
				if isinstance(best_plan, dict) and "code" in best_plan and best_plan["code"]:
					try:
						exec(best_plan["code"], {"state": state})
					except Exception as e:
						best_plan["obs"] = f"Error in planning: {e}"
						zwarn(f"[resample_continue] Error executing plan code: {e}")
				
				# Store state in plan
				if isinstance(best_plan, dict):
					best_plan["state"] = state.copy()
				
				new_step["plan"] = best_plan
			
			# Execute action if we resampled it
			if best_action:
				# Execute the action
				input_kwargs = _prepare_common_input_kwargs_for(ck, session, state)
				try:
					observation = ck.step_action(best_action, input_kwargs)
					if isinstance(best_action, dict):
						best_action["observation"] = observation
				except Exception as e:
					zwarn(f"[resample_continue] Error executing action: {e}")
					if isinstance(best_action, dict):
						best_action["observation"] = f"Error: {e}"
				
				new_step["action"] = best_action
				
				# Check for termination
				if ck._check_action_termination(best_action):  # type: ignore[attr-defined]
					new_step["terminated"] = True
					rprint("[resample_continue] Resampled step terminated the task")
			
			# Store PRM metadata in the resampled step
			new_step["resample_metadata"] = prm_metadata
			
		except Exception as e:
			zwarn(f"[resample_continue] Failed to execute resampled step: {e}")
			import traceback
			traceback.print_exc()
	
	# Calculate remaining steps
	if max_additional_steps is not None:
		max_steps = max_additional_steps
	else:
		max_steps = ck.max_steps - session.num_of_steps()
		if max_steps <= 0:
			max_steps = 1
	
	# Continue execution if not terminated
	if not (session.steps and session.steps[-1].get("terminated")):
		rprint(f"[resample_continue] Continuing execution for up to {max_steps} additional steps...")
		try:
			result_session = ck.run(
				task=task,
				session=session,
				max_steps=max_steps,
			)
			session = result_session
		except Exception as e:
			zwarn(f"[resample_continue] Failed to continue execution: {e}")
			import traceback
			traceback.print_exc()
	else:
		rprint("[resample_continue] Task terminated at resampled step, no continuation needed")
	
	# Check result correctness
	correctness = check_result_correctness(item, session)
	
	# Update instance
	inst["session"] = session.to_dict()
	inst["resampled_at_step"] = resample_step
	inst["resample_phase"] = phase
	inst["correctness_check"] = correctness
	inst["is_correct"] = correctness["is_correct"]
	
	rprint(f"[resample_continue] Completed, total steps: {session.num_of_steps()}, "
	       f"correct: {correctness['is_correct']}")
	
	return inst


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Resample from historical node and continue to completion.",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  # Resample action at step 2 and continue
  python resample_and_continue.py --input log.jsonl --output result.jsonl --resample_step 2

  # Resample both planning and action at step 1
  python resample_and_continue.py --input log.jsonl --output result.jsonl \\
      --resample_step 1 --phase both

  # Process only specific task IDs
  python resample_and_continue.py --input log.jsonl --output result.jsonl \\
      --resample_step 2 --task_ids task001,task002

  # Limit additional execution steps
  python resample_and_continue.py --input log.jsonl --output result.jsonl \\
      --resample_step 2 --max_additional_steps 5
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
		help="Output JSONL file with resampled and continued sessions"
	)
	
	parser.add_argument(
		"--resample_step",
		type=int,
		required=True,
		help="Step index to resample at (0-indexed)"
	)
	
	parser.add_argument(
		"--phase",
		type=str,
		choices=["planning", "action", "both"],
		default="action",
		help="Which phase to resample (default: action)"
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
		"--num_action_candidates",
		type=int,
		default=-1,
		help="Override NUM_ACTION_CANDIDATES (default: use env)"
	)
	
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	
	# Ensure output directory exists
	os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
	
	# Build configs
	configs = build_ck_configs_from_env()
	
	# Override num_action_candidates if specified
	if args.num_action_candidates > 0:
		if "process_reward" in configs:
			configs["process_reward"]["num_action_candidates"] = args.num_action_candidates
	
	# Ensure PRM is enabled
	if "process_reward" in configs:
		configs["process_reward"]["enable_process_reward"] = True
	
	# Initialize agent
	rprint("[resample_continue] Initializing ProcessRewardCKAgent")
	ck = ProcessRewardCKAgent(**configs)
	
	# Parse task IDs filter
	task_ids_filter = None
	if args.task_ids:
		task_ids_filter = set(tid.strip() for tid in args.task_ids.split(","))
		rprint(f"[resample_continue] Filtering for task IDs: {task_ids_filter}")
	
	# Resume support
	existing_ids = set()
	write_mode = "w"
	if args.resume and os.path.exists(args.output):
		write_mode = "a"
		rprint(f"[resample_continue] Resume enabled, scanning existing output: {args.output}")
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
			rprint(f"[resample_continue] Found {len(existing_ids)} existing ids")
		except Exception as e:
			zwarn(f"[resample_continue] Failed to read existing output for resume: {e}")
	
	# Process items
	processed = 0
	skipped = 0
	correct_count = 0
	
	with open(args.input, "r", encoding="utf-8") as rf, \
		 open(args.output, write_mode, encoding="utf-8") as wf:
		
		for line_num, line in enumerate(rf, 1):
			line = line.strip()
			if not line:
				continue
			
			try:
				item = json.loads(line)
			except Exception as e:
				zwarn(f"[resample_continue] Failed to parse line {line_num}: {e}")
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
			rprint(f"\n[resample_continue] ===== Processing item {item_id} (line {line_num}) =====")
			
			try:
				new_item = resample_and_continue_item(
					item=item,
					ck=ck,
					resample_step=args.resample_step,
					phase=args.phase,
					max_additional_steps=args.max_additional_steps,
				)
				
				# Track correctness
				if new_item.get("is_correct"):
					correct_count += 1
				
				wf.write(my_json_dumps(new_item, ensure_ascii=False) + "\n")
				wf.flush()
				processed += 1
				
			except Exception as e:
				zwarn(f"[resample_continue] Failed to process item {item_id}: {e}")
				import traceback
				traceback.print_exc()
			
			# Check limit
			if args.limit > 0 and processed >= args.limit:
				rprint(f"[resample_continue] Reached limit of {args.limit} items")
				break
	
	# Print summary
	rprint(f"\n[resample_continue] ===== Summary =====")
	rprint(f"[resample_continue] Processed: {processed} items")
	rprint(f"[resample_continue] Skipped: {skipped} items")
	rprint(f"[resample_continue] Correct: {correct_count}/{processed} ({100*correct_count/processed:.1f}%)" if processed > 0 else "[resample_continue] Correct: N/A")
	rprint(f"[resample_continue] Output: {args.output}")


if __name__ == "__main__":
	main()


