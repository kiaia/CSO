#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify high-scoring PRM candidates by continuing execution and generate verified DPO data.

This script combines candidate verification with DPO generation:
1. Read PRM-resampled data (from resample_prm_from_log.py)
2. For each step with candidates, filter pairs where:
   - Original/selected candidate score < bad_max (default 0.5)
   - Best candidate score > good_min (default 0.6)
3. For each qualified pair:
   - Continue execution from that step using the high-scoring candidate
   - Check if final answer is correct
   - If correct, generate DPO pair for that step only (not subsequent steps)
4. Output verified DPO data in LLaMAFactory format

This creates a stricter, verified version of DPO data where we confirm
that choosing the high-scoring candidate actually leads to correct answers.

Usage:
    python verify_and_generate_dpo.py \
        --input prm_resampled.jsonl \
        --output verified_dpo.jsonl \
        --good-min 0.6 \
        --bad-max 0.5
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from system.ckv3.agents.session import AgentSession
from system.ckv3.agents.utils import rprint, zwarn, my_json_dumps
from system.ckv3.ck_main.process_reward_agent import ProcessRewardCKAgent


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
	
	# Disable PRM for continuation (we're not resampling, just executing)
	use_process_reward = False
	
	configs: Dict[str, Any] = {
		"model": {"call_target": llm_url},
		"max_steps": 12,
		"process_reward": {
			"enable_process_reward": use_process_reward,
		},
	}
	return configs


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
	"""Iterate over JSONL file."""
	with open(path, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			try:
				obj = json.loads(line)
			except Exception:
				continue
			yield obj


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
					return state
		except Exception:
			pass
	
	return {}


def check_result_correctness(
	predicted_answer: Optional[str],
	ground_truth: Optional[str],
) -> bool:
	"""
	Check if predicted answer matches ground truth.
	
	Returns:
		True if correct, False otherwise
	"""
	if predicted_answer is None or ground_truth is None:
		return False
	
	pred = str(predicted_answer).strip().lower()
	truth = str(ground_truth).strip().lower()
	
	# Exact match
	if pred == truth:
		return True
	
	# Try numeric comparison
	try:
		pred_num = float(pred.replace(",", ""))
		truth_num = float(truth.replace(",", ""))
		return abs(pred_num - truth_num) < 1e-6
	except:
		pass
	
	return False


def extract_final_answer_from_session(session: AgentSession) -> Optional[str]:
	"""Extract the final answer from the completed session."""
	if not session.steps:
		return None
	
	# Look for final answer in reverse order
	for step in reversed(session.steps):
		if "action" not in step:
			continue
		
		action = step.get("action", {})
		observation = action.get("observation", "")
		code = action.get("code", "")
		
		# Try to extract from observation
		if isinstance(observation, str):
			if "Final Answer:" in observation:
				parts = observation.split("Final Answer:")
				if len(parts) > 1:
					return parts[1].strip()
			elif "Answer:" in observation:
				parts = observation.split("Answer:")
				if len(parts) > 1:
					return parts[1].strip()
		
		# Try to extract from code (stop() call)
		if isinstance(code, str) and "stop(" in code:
			try:
				start = code.index("stop(") + 5
				end = code.index(")", start)
				answer = code[start:end].strip().strip('"').strip("'")
				if answer:
					return answer
			except:
				pass
	
	return None


def _prepare_common_input_kwargs_for(ck: ProcessRewardCKAgent, session: AgentSession, state: Dict[str, Any]) -> Dict[str, Any]:
	"""Use the agent's helper to prepare inputs exactly as in normal run"""
	return ck._prepare_common_input_kwargs(session, state)  # type: ignore[attr-defined]


def continue_from_step_with_candidate(
	item: Dict[str, Any],
	ck: ProcessRewardCKAgent,
	step_idx: int,
	chosen_candidate: Dict[str, Any],
	max_additional_steps: Optional[int] = None,
) -> Tuple[Optional[AgentSession], Optional[str]]:
	"""
	Continue execution from a specific step using a chosen candidate.
	
	Args:
		item: Original item with task and session
		ck: CK agent for execution
		step_idx: Step index to continue from (0-indexed)
		chosen_candidate: The high-scoring candidate to use
		max_additional_steps: Maximum additional steps to execute
		
	Returns:
		(completed_session, final_answer) or (None, None) on failure
	"""
	task = item.get("task", "")
	session_obj = item.get("session")
	
	if not session_obj:
		return None, None
	
	steps = session_obj.get("steps", [])
	if not isinstance(steps, list) or not steps:
		return None, None
	
	if step_idx < 0 or step_idx >= len(steps):
		return None, None
	
	# Keep steps before step_idx
	kept_steps = steps[:step_idx]
	state = extract_state_from_steps(steps, step_idx - 1)
	
	# Rebuild session with kept steps
	session = AgentSession.init_from_data(
		task=task,
		steps=kept_steps,
		**session_obj.get("info", {})
	)
	
	# Execute the chosen candidate at step_idx
	try:
		# Create a new step
		new_step = {"step_idx": session.num_of_steps()}
		session.add_step(new_step)
		
		# Check if we have plan info in the candidate (for "both" phase resampling)
		# For action-only resampling, we skip plan execution
		
		# Execute the chosen action candidate
		if "thought" in chosen_candidate or "code" in chosen_candidate:
			# Prepare input kwargs
			input_kwargs = _prepare_common_input_kwargs_for(ck, session, state)
			
			# Execute the action
			try:
				observation = ck.step_action(chosen_candidate, input_kwargs)
				chosen_candidate_copy = dict(chosen_candidate)
				chosen_candidate_copy["observation"] = observation
				new_step["action"] = chosen_candidate_copy
				
				# Check for termination
				if ck._check_action_termination(chosen_candidate_copy):  # type: ignore[attr-defined]
					new_step["terminated"] = True
			except Exception as e:
				zwarn(f"[verify_dpo] Error executing chosen candidate: {e}")
				chosen_candidate_copy = dict(chosen_candidate)
				chosen_candidate_copy["observation"] = f"Error: {e}"
				new_step["action"] = chosen_candidate_copy
		
	except Exception as e:
		zwarn(f"[verify_dpo] Failed to execute chosen candidate at step {step_idx}: {e}")
		return None, None
	
	# Continue execution if not terminated
	if not (session.steps and session.steps[-1].get("terminated")):
		if max_additional_steps is None:
			max_steps = ck.max_steps - session.num_of_steps()
			if max_steps <= 0:
				max_steps = 1
		else:
			max_steps = max_additional_steps
		
		try:
			result_session = ck.run(
				task=task,
				session=session,
				max_steps=max_steps,
			)
			session = result_session
		except Exception as e:
			zwarn(f"[verify_dpo] Failed to continue execution: {e}")
			return None, None
	
	# Extract final answer
	final_answer = extract_final_answer_from_session(session)
	
	return session, final_answer


def find_candidate_pairs_for_verification(
	item: Dict[str, Any],
	good_min: float,
	bad_max: float,
	phase: str = "action",
) -> List[Tuple[int, Dict[str, Any], Dict[str, Any], Any]]:
	"""
	Find candidate pairs in the item that need verification.
	
	Args:
		item: Item with PRM-resampled session
		good_min: Minimum score for chosen candidate
		bad_max: Maximum score for rejected candidate
		phase: Which phase to extract from ("action" or "both")
		
	Returns:
		List of (step_idx, chosen_candidate, rejected_candidate, llm_input) tuples
	"""
	pairs: List[Tuple[int, Dict[str, Any], Dict[str, Any], Any]] = []
	
	session = item.get("session", {})
	steps = session.get("steps", [])
	
	if not isinstance(steps, list):
		return pairs
	
	for step_idx, step in enumerate(steps):
		# Extract action PRM info
		action = step.get("action", {})
		prm_info = action.get("prm_info", {})
		
		if not prm_info:
			continue
		
		all_candidates = prm_info.get("all_candidates", [])
		scores = prm_info.get("scores", [])
		selected_idx = prm_info.get("selected_idx")
		llm_input = action.get("llm_input")
		
		if not isinstance(all_candidates, list) or not isinstance(scores, list):
			continue
		
		if len(all_candidates) < 2 or len(scores) < 2:
			continue
		
		if len(all_candidates) != len(scores):
			continue
		
		# Check if original selected candidate has low score
		if selected_idx is None:
			continue
		
		try:
			original_score = float(scores[selected_idx])
		except:
			continue
		
		# Original must be bad (< bad_max)
		if not (original_score < bad_max):
			continue
		
		# Find best candidate
		try:
			best_idx = max(range(len(scores)), key=lambda i: float(scores[i]))
			best_score = float(scores[best_idx])
		except:
			continue
		
		# Best must be good (> good_min)
		if not (best_score > good_min):
			continue
		
		# Don't compare candidate with itself
		if best_idx == selected_idx:
			continue
		
		# Get candidates
		chosen_cand = all_candidates[best_idx] if isinstance(all_candidates[best_idx], dict) else {}
		rejected_cand = all_candidates[selected_idx] if isinstance(all_candidates[selected_idx], dict) else {}
		
		# Check both have non-empty content
		if not candidate_output_nonempty(chosen_cand) or not candidate_output_nonempty(rejected_cand):
			continue
		
		pairs.append((step_idx, chosen_cand, rejected_cand, llm_input))
	
	return pairs


def candidate_output_nonempty(cand: Dict[str, Any]) -> bool:
	"""Check if candidate has non-empty thought or code."""
	thought = (cand.get("thought") or "").strip()
	code = (cand.get("code") or "").strip()
	return bool(thought or code)


def build_answer_string(thought: Optional[str], code: Optional[str]) -> str:
	"""Build answer string from thought and code."""
	parts: List[str] = []
	t = (thought or "").strip()
	c = (code or "").strip()
	if t:
		parts.append(f"Thought: {t}")
	if c:
		parts.append("Code:\n" + c)
	return "\n".join(parts).strip()


def flatten_llm_input_to_user_instruction(llm_input: Any, fallback_task: str, step_idx: Optional[int]) -> str:
	"""
	Flatten CK llm_input (list of role/content messages) into a single 'human' message.
	"""
	if isinstance(llm_input, list) and llm_input:
		lines: List[str] = []
		for msg in llm_input:
			if isinstance(msg, dict):
				role = str(msg.get("role", "user")).strip()
				content = str(msg.get("content", "")).strip()
				if content:
					lines.append(f"[{role}] {content}")
			else:
				m = str(msg).strip()
				if m:
					lines.append(m)
		if lines:
			return "\n".join(lines)
	
	# Fallback: synthesized
	prefix = f"[user] Task: {fallback_task}".strip()
	if step_idx is not None:
		return f"{prefix}\n[meta] step_index={step_idx}"
	return prefix


def to_llamafactory_sample(user_instruction: str, chosen_text: str, rejected_text: str) -> Dict[str, Any]:
	"""Convert to LLaMAFactory DPO format."""
	return {
		"conversations": [
			{"from": "human", "value": user_instruction}
		],
		"chosen": {"from": "gpt", "value": chosen_text},
		"rejected": {"from": "gpt", "value": rejected_text},
	}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Verify high-scoring candidates and generate verified DPO data.",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  # Basic usage
  python verify_and_generate_dpo.py \\
      --input prm_resampled.jsonl \\
      --output verified_dpo.jsonl

  # Custom thresholds
  python verify_and_generate_dpo.py \\
      --input prm_resampled.jsonl \\
      --output verified_dpo.jsonl \\
      --good-min 0.7 \\
      --bad-max 0.4

  # Limit processing
  python verify_and_generate_dpo.py \\
      --input prm_resampled.jsonl \\
      --output verified_dpo.jsonl \\
      --limit 100
		"""
	)
	
	parser.add_argument(
		"--input",
		type=str,
		required=True,
		help="Input PRM-resampled JSONL file"
	)
	
	parser.add_argument(
		"--output",
		type=str,
		required=True,
		help="Output verified DPO JSONL file"
	)
	
	parser.add_argument(
		"--good-min",
		type=float,
		default=0.6,
		help="Minimum score for chosen candidate (default: 0.6)"
	)
	
	parser.add_argument(
		"--bad-max",
		type=float,
		default=0.5,
		help="Maximum score for rejected candidate (default: 0.5)"
	)
	
	parser.add_argument(
		"--phase",
		type=str,
		choices=["action", "planning", "both"],
		default="action",
		help="Which phase to extract candidates from (default: action)"
	)
	
	parser.add_argument(
		"--max-additional-steps",
		type=int,
		default=None,
		help="Maximum additional steps for continuation (default: agent's remaining steps)"
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
		help="Resume: skip items already in output"
	)
	
	parser.add_argument(
		"--jsonl",
		action="store_true",
		default=True,
		help="Output JSONL format (default: True)"
	)
	
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	
	# Ensure output directory exists
	os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
	
	# Build agent config
	configs = build_ck_configs_from_env()
	
	# Initialize agent (without PRM, just for execution)
	rprint("[verify_dpo] Initializing CK agent for verification")
	from system.ckv3.ck_main.agent import CKAgent
	ck = CKAgent(**configs)
	
	# Statistics
	total_items = 0
	total_candidate_pairs = 0
	verified_pairs = 0
	failed_verifications = 0
	
	# Resume support
	existing_samples = set()
	if args.resume and os.path.exists(args.output):
		rprint(f"[verify_dpo] Resume enabled, scanning existing output: {args.output}")
		try:
			with open(args.output, "r", encoding="utf-8") as ef:
				for line in ef:
					line = line.strip()
					if not line:
						continue
					try:
						obj = json.loads(line)
						# Create a hash of the sample to detect duplicates
						conv_str = json.dumps(obj.get("conversations", []), sort_keys=True)
						existing_samples.add(conv_str)
					except:
						continue
			rprint(f"[verify_dpo] Found {len(existing_samples)} existing samples")
		except Exception as e:
			zwarn(f"[verify_dpo] Failed to read existing output: {e}")
	
	# Process items
	write_mode = "a" if args.resume else "w"
	
	with open(args.output, write_mode, encoding="utf-8") as wf:
		for item in iter_jsonl(args.input):
			total_items += 1
			
			if args.limit > 0 and total_items > args.limit:
				break
			
			task = str(item.get("task", "")).strip()
			ground_truth = item.get("answer") or item.get("ground_truth") or item.get("expected_answer")
			item_id = item.get("id") or item.get("task_id")
			
			rprint(f"\n[verify_dpo] ===== Processing item {item_id} ({total_items}) =====")
			
			if not ground_truth:
				rprint(f"[verify_dpo] No ground truth found, skipping")
				continue
			
			# Find candidate pairs that need verification
			pairs = find_candidate_pairs_for_verification(
				item=item,
				good_min=args.good_min,
				bad_max=args.bad_max,
				phase=args.phase,
			)
			
			if not pairs:
				rprint(f"[verify_dpo] No candidate pairs found for verification")
				continue
			
			rprint(f"[verify_dpo] Found {len(pairs)} candidate pairs to verify")
			
			# Verify each pair
			for step_idx, chosen_cand, rejected_cand, llm_input in pairs:
				total_candidate_pairs += 1
				
				rprint(f"[verify_dpo] Verifying step {step_idx} pair...")
				
				# Continue execution with chosen candidate
				completed_session, final_answer = continue_from_step_with_candidate(
					item=item,
					ck=ck,
					step_idx=step_idx,
					chosen_candidate=chosen_cand,
					max_additional_steps=args.max_additional_steps,
				)
				
				if completed_session is None or final_answer is None:
					rprint(f"[verify_dpo] Verification failed: could not complete execution")
					failed_verifications += 1
					continue
				
				# Check correctness
				is_correct = check_result_correctness(final_answer, ground_truth)
				
				if not is_correct:
					rprint(f"[verify_dpo] Verification failed: incorrect answer "
					       f"(predicted={final_answer}, ground_truth={ground_truth})")
					failed_verifications += 1
					continue
				
				# Verified! Generate DPO sample for this step only
				rprint(f"[verify_dpo] ✓ Verified! Generating DPO sample for step {step_idx}")
				
				# Build user instruction
				user_instruction = flatten_llm_input_to_user_instruction(
					llm_input=llm_input,
					fallback_task=task,
					step_idx=step_idx,
				)
				
				# Build chosen and rejected texts
				chosen_text = build_answer_string(
					chosen_cand.get("thought"),
					chosen_cand.get("code")
				)
				
				rejected_text = build_answer_string(
					rejected_cand.get("thought"),
					rejected_cand.get("code")
				)
				
				# Create DPO sample
				dpo_sample = to_llamafactory_sample(
					user_instruction=user_instruction,
					chosen_text=chosen_text,
					rejected_text=rejected_text,
				)
				
				# Check for duplicates
				conv_str = json.dumps(dpo_sample.get("conversations", []), sort_keys=True)
				if conv_str in existing_samples:
					rprint(f"[verify_dpo] Duplicate sample, skipping")
					continue
				
				# Write sample
				wf.write(json.dumps(dpo_sample, ensure_ascii=False) + "\n")
				wf.flush()
				
				existing_samples.add(conv_str)
				verified_pairs += 1
				
				rprint(f"[verify_dpo] Verified pair {verified_pairs} written")
	
	# Print summary
	rprint(f"\n[verify_dpo] ===== Summary =====")
	rprint(f"[verify_dpo] Total items processed: {total_items}")
	rprint(f"[verify_dpo] Total candidate pairs found: {total_candidate_pairs}")
	rprint(f"[verify_dpo] Verified pairs: {verified_pairs}")
	rprint(f"[verify_dpo] Failed verifications: {failed_verifications}")
	if total_candidate_pairs > 0:
		success_rate = 100.0 * verified_pairs / total_candidate_pairs
		rprint(f"[verify_dpo] Success rate: {success_rate:.1f}%")
	rprint(f"[verify_dpo] Output: {args.output}")


if __name__ == "__main__":
	main()

