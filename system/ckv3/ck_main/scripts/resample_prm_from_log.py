#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from ...agents.session import AgentSession
from ...agents.utils import rprint, zwarn
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
	# Model targets
	llm_url = os.getenv("LLM_URL", "gpt:gpt-4o-mini")
	# Note: top-level model_multimodal is not supported by CKAgent/MultiStepAgent
	# Keep only top-level 'model'; sub-agents manage their own multimodal models internally.

	# Process reward config (mirror shell script variables)
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

	# Planning PRM
	enable_planning_prm = _parse_bool_env("ENABLE_PLANNING_PRM", True)
	planning_score_weight = _parse_float_env("PLANNING_SCORE_WEIGHT", 0.7)
	diversity_weight = _parse_float_env("DIVERSITY_WEIGHT", 0.3)

	# Diverse sequential sampling
	enable_diverse_sampling = _parse_bool_env("ENABLE_DIVERSE_SAMPLING", True)
	sequential_mode = _parse_bool_env("SEQUENTIAL_MODE", True)
	diversity_threshold = _parse_float_env("DIVERSITY_THRESHOLD", 0.4)
	max_sampling_attempts = _parse_int_env("MAX_SAMPLING_ATTEMPTS", 3)
	diversity_prompt_strength = os.getenv("DIVERSITY_PROMPT_STRENGTH", "medium")

	# Adaptive sampling
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


def _extract_state_from_prefix_steps(prefix_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
	if not prefix_steps:
		return {}
	try:
		last_step = prefix_steps[-1]
		plan = last_step.get("plan", {})
		state = plan.get("state", {})
		if isinstance(state, dict):
			return state
	except Exception:
		pass
	return {}


def _prepare_common_input_kwargs_for(ck: ProcessRewardCKAgent, session: AgentSession, state: Dict[str, Any]) -> Dict[str, Any]:
	# Use the agent's helper to prepare inputs exactly as in normal run
	return ck._prepare_common_input_kwargs(session, state)  # type: ignore[attr-defined]


def resample_for_step(
	ck: ProcessRewardCKAgent,
	session: AgentSession,
	state: Dict[str, Any],
	do_planning: bool,
	do_action: bool,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
	plan_prm_info: Optional[Dict[str, Any]] = None
	action_prm_info: Optional[Dict[str, Any]] = None

	# Add a placeholder current step so template preparation has a "current_step" to reference,
	# matching the normal agent loop behavior.
	session.add_step({"step_idx": session.num_of_steps()})
	input_kwargs = _prepare_common_input_kwargs_for(ck, session, state)

	# Planning PRM (no execution)
	if do_planning and ck.enable_planning_prm:
		try:
			plan_candidates = ck._generate_plan_candidates(input_kwargs, session)  # type: ignore[attr-defined]
			if len(plan_candidates) >= 2:
				scores, evals = ck._evaluate_plan_candidates(session, state, plan_candidates)  # type: ignore[attr-defined]
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
				best_idx = max(range(len(scores)), key=lambda i: scores[i]) if scores else 0
				plan_prm_info = {
					"all_candidates": safe_candidates,
					"scores": scores,
					"selected_idx": best_idx,
					"rpm_evaluations": evals,
					"num_candidates": len(plan_candidates),
					"evaluation_method": "planning_reward_model",
					"phase": "planning",
				}
		except Exception as e:
			zwarn(f"[resample] Planning PRM failed: {e}")

	# Action PRM (no execution)
	if do_action:
		try:
			# Build action kwargs like original agent does (state is json string)
			action_kwargs = input_kwargs.copy()
			try:
				import json as _json
				action_kwargs["state"] = _json.dumps(state, ensure_ascii=False, indent=2)
			except Exception:
				action_kwargs["state"] = "{}"

			action_candidates = ck._generate_action_candidates(action_kwargs, session)  # type: ignore[attr-defined]
			if len(action_candidates) >= 2:
				scores, evals = ck._evaluate_action_candidates(session, state, action_candidates)  # type: ignore[attr-defined]
				safe_candidates = []
				for cand in action_candidates:
					if isinstance(cand, dict):
						safe = {"thought": cand.get("thought", ""), "code": cand.get("code", "")}
						if "diversity_info" in cand:
							safe["diversity_info"] = cand["diversity_info"]
						safe_candidates.append(safe)
					else:
						safe_candidates.append({"thought": str(cand), "code": ""})
				best_idx = max(range(len(scores)), key=lambda i: scores[i]) if scores else 0
				action_prm_info = {
					"all_candidates": safe_candidates,
					"scores": scores,
					"selected_idx": best_idx,
					"rpm_evaluations": evals,
					"num_candidates": len(action_candidates),
					"evaluation_method": "process_reward_model",
				}
		except Exception as e:
			zwarn(f"[resample] Action PRM failed: {e}")

	# Remove the placeholder current step to avoid side effects outside this function
	try:
		if session.num_of_steps() > 0:
			session.steps.pop()
	except Exception:
		pass

	return plan_prm_info, action_prm_info


def convert_text_trajectory_to_session(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
	# Fallback: if there is a textual trajectory in feedback, we cannot reconstruct reliably here.
	# Return None to signal full re-run fallback.
	return None


def process_one_item(
	item: Dict[str, Any],
	ck: ProcessRewardCKAgent,
	phase: str,
) -> Dict[str, Any]:
	do_planning = phase in ("planning", "both")
	do_action = phase in ("action", "both")

	inst = dict(item)
	task = inst.get("task", "")

	# Determine or reconstruct session
	session_obj: Optional[Dict[str, Any]] = inst.get("session")
	if not session_obj:
		session_obj = convert_text_trajectory_to_session(inst)

	if not session_obj:
		# Fallback: do a fresh single-step PRM evaluation without execution by building an empty session
		rprint(f"[resample] No session found for id={inst.get('id')}, falling back to fresh prompt-only sampling")
		session = AgentSession.init_from_data(task=task, steps=[])
		# One-shot: treat as 1 step context (no prior steps)
		state = {}
		plan_info, action_info = resample_for_step(ck, session, state, do_planning, do_action)
		# Create synthetic one-step log
		step = {"step_idx": 0}
		if plan_info:
			step["plan"] = {"prm_info": plan_info}
		if action_info:
			step.setdefault("action", {})
			step["action"]["prm_info"] = action_info
		inst["session"] = {"steps": [step], "info": {}}
		return inst

	steps = session_obj.get("steps", [])
	if not isinstance(steps, list) or not steps:
		inst["session"] = {"steps": [], "info": session_obj.get("info", {})}
		return inst

	new_steps: List[Dict[str, Any]] = []
	for k in range(len(steps)):
		# Copy original step
		orig_step = steps[k]
		new_step = json.loads(json.dumps(orig_step, ensure_ascii=False))
		# Build prefix session
		prefix_steps = []
		for j in range(k):
			prefix_steps.append(steps[j])
		session = AgentSession.init_from_data(task=task, steps=prefix_steps)
		state = _extract_state_from_prefix_steps(prefix_steps)
		plan_info, action_info = resample_for_step(ck, session, state, do_planning, do_action)
		# Attach PRM info to current step without executing
		if plan_info:
			new_step.setdefault("plan", {})
			new_step["plan"]["prm_info"] = plan_info
		if action_info:
			new_step.setdefault("action", {})
			new_step["action"]["prm_info"] = action_info
		new_steps.append(new_step)

	inst["session"] = {
		"steps": new_steps,
		"info": session_obj.get("info", {}),
	}
	return inst


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Resample PRM candidates per step conditioned on existing trajectories and log PRM info."
	)
	parser.add_argument("--input", type=str, required=True, help="Input JSONL log (with or without session).")
	parser.add_argument("--output", type=str, required=True, help="Output JSONL with PRM info appended.")
	parser.add_argument("--phase", type=str, choices=["planning", "action", "both"], default="both", help="Which phase(s) to score.")
	parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of items.")
	parser.add_argument("--num_action_candidates", type=int, default=-1, help="Override NUM_ACTION_CANDIDATES.")
	parser.add_argument("--resume", action="store_true", help="Resume: skip items whose ids already exist in output and append new ones.")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

	# Build configs matching the shell hyperparameters
	configs = build_ck_configs_from_env()
	if args.num_action_candidates > 0:
		if "process_reward" in configs:
			configs["process_reward"]["num_action_candidates"] = int(args.num_action_candidates)

	# Ensure PRM is enabled for scoring
	if "process_reward" in configs:
		configs["process_reward"]["enable_process_reward"] = True

	# Initialize PRM-enabled CK agent
	ck = ProcessRewardCKAgent(**configs)

	processed = 0
	# Resume support: collect existing ids if requested
	existing_ids = set()
	write_mode = "w"
	if args.resume and os.path.exists(args.output):
		write_mode = "a"
		rprint(f"[resample] Resume enabled, scanning existing output: {args.output}")
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
			rprint(f"[resample] Found {len(existing_ids)} existing ids")
		except Exception as e:
			zwarn(f"[resample] Failed to read existing output for resume: {e}")

	with open(args.input, "r", encoding="utf-8") as rf, open(args.output, write_mode, encoding="utf-8") as wf:
		for line in rf:
			line = line.strip()
			if not line:
				continue
			try:
				item = json.loads(line)
			except Exception:
				continue
			item_id = item.get("id") or item.get("task_id")
			if item_id and str(item_id) in existing_ids:
				continue
			new_item = process_one_item(item, ck, args.phase)
			wf.write(json.dumps(new_item, ensure_ascii=False) + "\n")
			processed += 1
			if args.limit > 0 and processed >= args.limit:
				break
	rprint(f"[resample] Wrote {processed} items to {args.output}")


if __name__ == "__main__":
	main()


