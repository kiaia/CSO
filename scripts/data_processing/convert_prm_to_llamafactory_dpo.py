#!/usr/bin/env python3
"""
Convert PRM-resampled CK logs to LLaMA Factory DPO format.

Input:
  - JSONL with GAIA/CK session logs that include per-step PRM candidates and scores:
    item: {
      "task": str,
      "session": {
        "steps": [
          {
            "action": {
              "llm_input": [ { "role": "system"|"user"|"assistant", "content": str }, ... ],
              "prm_info": {
                "all_candidates": [ { "thought": str, "code": str, "llm_input": [...]? }, ... ],
                "scores": [ float, ... ],
                "selected_idx": int
              }
            },
            ...
          },
          ...
        ]
      }
    }

Output (JSONL or JSON array depending on --jsonl flag):
  Each DPO sample matches LLaMA Factory schema:
  {
    "conversations": [
      { "from": "human", "value": "user instruction (includes CK step context)" }
    ],
    "chosen":   { "from": "gpt", "value": "chosen answer" },
    "rejected": { "from": "gpt", "value": "rejected answer" }
  }

Selection policy:
  - For each step with >=2 candidates:
    * Let best = argmax score
    * If best_score > --good-min (default: 0.6) then create pairs with any candidate whose score < --bad-max (default: 0.5)
    * For each such low-scoring candidate, emit one DPO sample (best vs that candidate)
  - Skips empty outputs (both thought and code empty).

Context (user instruction) policy:
  - Prefer action.llm_input if present (list of chat messages).
  - Else prefer candidate[0].llm_input if present.
  - Else fallback to a compact synthesized instruction from top-level fields (task + step index).
  - The chat messages are flattened into a single text string that mirrors CK call semantics:
      [system] ...
      [user] ...
      [assistant] ...
    Joined with newlines in original order.
"""
import argparse
import json
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
	formatter = argparse.ArgumentDefaultsHelpFormatter
	parser = argparse.ArgumentParser(description="Convert PRM logs to LLaMA Factory DPO format.", formatter_class=formatter)
	parser.add_argument("--input", type=str, required=True, help="Input PRM-resampled JSONL.")
	parser.add_argument("--output", type=str, required=True, help="Output DPO file (JSONL by default).")
	parser.add_argument("--good-min", type=float, default=0.6, help="Minimum score for chosen candidate (strict >).")
	parser.add_argument("--bad-max", type=float, default=0.5, help="Maximum score for rejected candidate (strict <).")
	parser.add_argument("--phase", type=str, choices=["action", "planning", "both"], default="action", help="Which phase to use. Currently only 'action' is supported for candidate extraction.")
	parser.add_argument("--jsonl", action="store_true", help="Write JSONL (one sample per line). Defaults to JSON array when disabled.")
	return parser.parse_args()


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
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


def build_answer_string(thought: Optional[str], code: Optional[str]) -> str:
	parts: List[str] = []
	t = (thought or "").strip()
	c = (code or "").strip()
	if t:
		parts.append(f"Thought: {t}")
	if c:
		# Present code in a block-like style without backticks for safety
		parts.append("Code:\n" + c)
	return "\n".join(parts).strip()


def flatten_llm_input_to_user_instruction(llm_input: Any, fallback_task: str, step_idx: Optional[int]) -> str:
	"""
	Flatten CK llm_input (list of role/content messages) into a single 'human' message
	that mirrors the actual call context. If llm_input is unavailable, synthesize a minimal
	instruction from task and step index.
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
				# Non-dict message (fallback)
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


def extract_action_prm(step: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[float], Optional[int], Any]:
	"""
	Returns (candidates, scores, selected_idx, llm_input) from an action step.
	"""
	action = step.get("action") or {}
	prm = action.get("prm_info") or {}
	all_candidates = prm.get("all_candidates") or []
	scores = prm.get("scores") or []
	selected_idx = prm.get("selected_idx")
	llm_input = action.get("llm_input")
	return all_candidates, scores, selected_idx, llm_input


def candidate_output_nonempty(cand: Dict[str, Any]) -> bool:
	thought = (cand.get("thought") or "").strip()
	code = (cand.get("code") or "").strip()
	return bool(thought or code)


def to_llamafactory_sample(user_instruction: str, chosen_text: str, rejected_text: str) -> Dict[str, Any]:
	return {
		"conversations": [
			{"from": "human", "value": user_instruction}
		],
		"chosen": {"from": "gpt", "value": chosen_text},
		"rejected": {"from": "gpt", "value": rejected_text},
	}


def main() -> None:
	args = parse_args()
	input_path = args.input
	output_path = args.output
	good_min = float(args.good_min)
	bad_max = float(args.bad_max)
	use_jsonl = bool(args.jsonl)

	total_items = 0
	total_steps = 0
	total_pairs = 0
	emitted = 0

	samples: List[Dict[str, Any]] = []

	with (open(output_path, "w", encoding="utf-8") if use_jsonl else open(output_path, "w", encoding="utf-8")) as wf:
		for obj in iter_jsonl(input_path):
			total_items += 1
			task = str(obj.get("task") or "").strip()
			session = obj.get("session") or {}
			steps = session.get("steps") or []
			if not isinstance(steps, list) or not steps:
				continue
			for s_idx, step in enumerate(steps):
				# Only action phase supported for candidate extraction in this script
				all_candidates, scores, selected_idx, llm_input = extract_action_prm(step)
				if not isinstance(all_candidates, list) or not isinstance(scores, list):
					continue
				if len(all_candidates) < 2 or len(scores) < 2 or len(all_candidates) != len(scores):
					continue
				total_steps += 1
				# Best candidate
				try:
					best_idx = max(range(len(scores)), key=lambda i: float(scores[i]))
				except Exception:
					continue
				best_score = float(scores[best_idx])
				if not (best_score > good_min):
					# Best not confident enough for positive
					continue
				best_cand = all_candidates[best_idx] if isinstance(all_candidates[best_idx], dict) else {}
				if not candidate_output_nonempty(best_cand):
					continue
				# Derive instruction text from llm_input (prefer action-level; fallback to candidate-level; else synthesized)
				cand_llm_input = None
				if not llm_input and isinstance(best_cand, dict):
					cand_llm_input = best_cand.get("llm_input")
				user_instruction = flatten_llm_input_to_user_instruction(
					llm_input if llm_input is not None else cand_llm_input,
					fallback_task=task,
					step_idx=s_idx,
				)
				chosen_text = build_answer_string(best_cand.get("thought"), best_cand.get("code"))
				# For each low-scoring candidate create a pair
				for j, score in enumerate(scores):
					if j == best_idx:
						continue
					try:
						s = float(score)
					except Exception:
						continue
					if not (s < bad_max):
						continue
					rejected_cand = all_candidates[j] if isinstance(all_candidates[j], dict) else {}
					if not candidate_output_nonempty(rejected_cand):
						continue
					rejected_text = build_answer_string(rejected_cand.get("thought"), rejected_cand.get("code"))
					total_pairs += 1
					record = to_llamafactory_sample(user_instruction, chosen_text, rejected_text)
					if use_jsonl:
						wf.write(json.dumps(record, ensure_ascii=False) + "\n")
						emitted += 1
					else:
						samples.append(record)
		# If not JSONL, dump as a single JSON array
		if not use_jsonl:
			wf.write(json.dumps(samples, ensure_ascii=False, indent=2))
			emitted = len(samples)

	# Brief report to stderr
	print(
		f"Processed items={total_items}, steps_with_candidates={total_steps}, candidate_pairs_scanned={total_pairs}, emitted={emitted}",
		file=sys.stderr,
	)


if __name__ == "__main__":
	main()


