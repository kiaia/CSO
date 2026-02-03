import json
import os
import sys
import argparse
import re
from typing import Dict, List, Any
from tqdm import tqdm
import time

# Add the System directory to path to import ckv3 modules
# Assumes this script is in scripts/data_processing/
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, os.path.join(project_root, 'system'))

from ckv3.agents.model import LLM
from ckv3.agents.process_reward_model import ProcessRewardModel


def read_jsonl(file_path):
    """Read data from JSONL file"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data, output_path):
    """Save data to JSONL file"""
    with open(output_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def extract_action_candidate(step: Dict) -> Dict:
    """Extract action candidate from a step"""
    action = step.get('action', {})
    return {
        'thought': action.get('thought', ''),
        'code': action.get('code', '')
    }


def generate_alternative_candidates(llm: LLM, task: str, session_history: List[Dict], 
                                     current_state: Dict, original_candidate: Dict, 
                                     num_candidates: int = 2) -> List[Dict]:
    """
    Generate alternative action candidates using LLM.
    
    Args:
        llm: The language model instance
        task: The task description
        session_history: History of previous steps
        current_state: Current progress state
        original_candidate: The original action candidate
        num_candidates: Number of alternatives to generate
        
    Returns:
        List of alternative action candidates
    """
    # Build prompt for generating alternatives
    history_str = format_session_history(session_history)
    state_str = json.dumps(current_state, ensure_ascii=False, indent=2)
    
    prompt = f"""## Target Task
{task}

## Current Progress State
{state_str}

## Recent Execution History
{history_str}

## Original Action
The agent attempted the following action:
**Thought**: {original_candidate['thought']}
**Code**: 
```python
{original_candidate['code']}
```

## Task
Generate {num_candidates} ALTERNATIVE action candidates that could accomplish this step differently. Each alternative should:
1. Have a different approach/strategy than the original
2. Still be relevant to the current task and state
3. Have clear reasoning (thought) and executable code
4. Make meaningful progress toward the goal

## Output Format
For each alternative, provide:
**Thought**: [Your reasoning for this alternative approach]
**Code**: 
```python
[Your alternative code]
```

Separate each alternative with "---" marker.
"""
    
    messages = [
        {"role": "system", "content": "You are an expert assistant that generates diverse alternative approaches for agent actions."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = llm(messages)
        candidates = parse_candidates_from_response(response, num_candidates)
        return candidates
    except Exception as e:
        print(f"Error generating alternatives: {e}")
        return []


def parse_candidates_from_response(response: str, num_candidates: int) -> List[Dict]:
    """Parse multiple candidates from LLM response"""
    candidates = []
    
    # Split by "---" marker
    parts = response.split("---")
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        # Extract thought
        thought_match = search_in_text(part, r"(?:Thought|**Thought**)\s*:\s*(.+?)(?=\n.*?:|\n```|$)", 
                                       flags=re.DOTALL)
        thought = thought_match.strip() if thought_match else ""
        
        # Extract code
        code_match = search_in_text(part, r"```python\n(.+?)```", flags=re.DOTALL)
        if not code_match:
            code_match = search_in_text(part, r"```\n(.+?)```", flags=re.DOTALL)
        code = code_match.strip() if code_match else ""
        
        if thought or code:
            candidates.append({'thought': thought, 'code': code})
    
    # Ensure we have the right number
    while len(candidates) < num_candidates and candidates:
        # Duplicate last candidate if needed
        candidates.append(candidates[-1])
    
    return candidates[:num_candidates]


def search_in_text(text: str, pattern: str, flags=0) -> str:
    """Search for pattern in text and return matched group"""
    match = re.search(pattern, text, flags)
    return match.group(1).strip() if match else ""


def format_session_history(session_history: List[Dict]) -> str:
    """Format session history as string"""
    if not session_history:
        return "No previous steps."
    
    # Take last 3 steps to avoid too long context
    recent_steps = session_history[-3:] if len(session_history) > 3 else session_history
    
    formatted_steps = []
    for i, step in enumerate(recent_steps):
        step_idx = step.get('step_idx', i)
        action = step.get('action', {})
        thought = action.get('thought', 'N/A')
        code = action.get('code', 'N/A')
        observation = action.get('observation', 'N/A')
        
        step_str = f"""### Step {step_idx}
Thought: {thought}
Action: ```
{code}
```
Observation: {format_observation(observation)}"""
        formatted_steps.append(step_str)
    
    return "\n\n".join(formatted_steps)


def format_observation(observation) -> str:
    """Format observation"""
    if isinstance(observation, (list, tuple)):
        return "\n".join([f"- Result {i}: {str(obs)}" for i, obs in enumerate(observation)])
    else:
        return str(observation)


def process_trajectory_item(item: Dict, prm: ProcessRewardModel, llm: LLM) -> Dict:
    """
    Process a single trajectory item:
    1. Score original steps with PRM
    2. Generate 2 alternatives for each intermediate step
    3. Score alternatives with PRM
    """
    task = item['task']
    session = item.get('session', {})
    steps = session.get('steps', [])
    
    processed_steps = []
    
    for i, step in enumerate(tqdm(steps, desc=f"Processing task {item.get('id', 'unknown')}")):
        action = step.get('action', {})
        plan = step.get('plan', {})
        
        # Extract current step info
        current_state = plan.get('state', {})
        session_history = steps[:i] if steps else []
        original_candidate = extract_action_candidate(step)
        
        # Skip if no thought/code
        if not original_candidate['thought'] and not original_candidate['code']:
            processed_steps.append(step)
            continue
        
        # Score original action
        original_score = 0.5
        try:
            scores, evaluations = prm.evaluate_action_candidates_detailed(
                task=task,
                session_history=session_history,
                current_state=current_state,
                action_candidates=[original_candidate]
            )
            original_score = scores[0] if scores else 0.5
        except Exception as e:
            print(f"Error scoring original action: {e}")
        
        # Generate alternatives for intermediate steps
        alternative_candidates = []
        alternative_scores = []
        alternative_evaluations = []
        
        if i < len(steps) - 1:  # Only for intermediate steps (not the last one)
            try:
                alternatives = generate_alternative_candidates(
                    llm=llm,
                    task=task,
                    session_history=session_history,
                    current_state=current_state,
                    original_candidate=original_candidate,
                    num_candidates=2
                )
                
                if alternatives:
                    # Score the alternatives
                    scores, evaluations = prm.evaluate_action_candidates_detailed(
                        task=task,
                        session_history=session_history,
                        current_state=current_state,
                        action_candidates=alternatives
                    )
                    
                    alternative_candidates = alternatives
                    alternative_scores = scores
                    alternative_evaluations = evaluations
                    
            except Exception as e:
                print(f"Error generating/scoring alternatives: {e}")
        
        # Build processed step
        processed_step = {
            **step,
            'prm_original_score': original_score,
            'prm_alternatives': {
                'candidates': alternative_candidates,
                'scores': alternative_scores,
                'evaluations': alternative_evaluations
            }
        }
        
        processed_steps.append(processed_step)
    
    # Build processed item
    processed_item = {
        **item,
        'session': {
            **session,
            'steps': processed_steps
        }
    }
    
    return processed_item


def main():
    parser = argparse.ArgumentParser(description='Score trajectory steps and generate alternatives')
    parser.add_argument('--input', type=str, required=True, help='Input JSONL file path')
    parser.add_argument('--output', type=str, required=True, help='Output JSONL file path')
    parser.add_argument('--llm-endpoint', type=str, default='gpt:gpt-4.1', 
                       help='LLM endpoint for generation (default: gpt:gpt-4.1)')
    parser.add_argument('--rpm-endpoint', type=str, default='gpt:gpt-4.1',
                       help='RPM endpoint for scoring (default: gpt:gpt-4.1)')
    parser.add_argument('--max-items', type=int, default=None, 
                       help='Maximum number of items to process')
    
    args = parser.parse_args()
    
    # Initialize models
    print("Initializing LLM and PRM...")
    llm = LLM(call_target=args.llm_endpoint)
    
    # Set up environment variables if needed
    if 'AZURE_OPENAI_API_KEY' in os.environ:
        prm_config = {
            'call_target': args.rpm_endpoint,
            'enable_parallel_evaluation': False,  # Sequential for stability
            'evaluation_timeout': 60
        }
    else:
        # Use environment variables - MUST be set before running
        if 'AZURE_OPENAI_API_KEY' not in os.environ:
            raise ValueError("AZURE_OPENAI_API_KEY environment variable must be set")
        if 'AZURE_OPENAI_ENDPOINT' not in os.environ:
            raise ValueError("AZURE_OPENAI_ENDPOINT environment variable must be set")
        if 'AZURE_OPENAI_API_VERSION' not in os.environ:
            os.environ['AZURE_OPENAI_API_VERSION'] = '2025-01-01-preview'
        
        prm_config = {
            'call_target': args.rpm_endpoint,
            'enable_parallel_evaluation': False,
            'evaluation_timeout': 60
        }
    
    prm = ProcessRewardModel(**prm_config)
    
    print("Loading data...")
    data = read_jsonl(args.input)
    
    if args.max_items:
        data = data[:args.max_items]
        print(f"Processing {len(data)} items (limited from {len(read_jsonl(args.input))})")
    else:
        print(f"Processing {len(data)} items")
    
    # Process items
    processed_data = []
    for item in tqdm(data, desc="Processing trajectories"):
        try:
            processed_item = process_trajectory_item(item, prm, llm)
            processed_data.append(processed_item)
        except Exception as e:
            print(f"Error processing item: {e}")
            import traceback
            traceback.print_exc()
            # Still append the original item
            processed_data.append(item)
    
    # Save output
    print(f"Saving results to {args.output}...")
    save_jsonl(processed_data, args.output)
    print(f"Done! Processed {len(processed_data)} items.")


if __name__ == '__main__':
    main()

