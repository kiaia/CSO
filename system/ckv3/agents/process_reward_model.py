#
# Process Reward Model for evaluating action quality during agent execution

import json
import time
import multiprocessing as mp
from functools import partial
from typing import List, Dict, Any, Tuple
from .model import LLM
from .utils import rprint, zwarn, KwargsInitializable


class ProcessRewardModel(KwargsInitializable):
    """
    Process Reward Model for evaluating the quality of agent actions during execution.
    Acts as a separate evaluator that can be called via API-like interface.
    """
    
    def __init__(self, **kwargs):
        # Model configuration
        self.enable_parallel_evaluation = kwargs.pop('enable_parallel_evaluation', True)
        self.evaluation_timeout = kwargs.pop('evaluation_timeout', 30)
        self.max_retries = kwargs.pop('max_retries', 2)
        
        # Evaluation parameters
        self.score_scale = (0.0, 1.0)  # Score range
        self.default_score = 0.5  # Default score when evaluation fails
        
        # Planning PRM feature flag - if true, PRM will be used during planning stages as well
        self.enable_planning_prm = kwargs.pop('enable_planning_prm', False)
        
        # Observation truncation limit (similar to agent's obs_max_token)
        self.obs_max_token = kwargs.pop('obs_max_token', 8192)
        
        # Initialize the model with remaining kwargs
        model_kwargs = kwargs.copy()
        self.model = LLM(**model_kwargs)
        
        # Initialize base class without model parameters to avoid conflicts
        super().__init__(enable_parallel_evaluation=self.enable_parallel_evaluation,
                        evaluation_timeout=self.evaluation_timeout,
                        max_retries=self.max_retries,
                        score_scale=self.score_scale,
                        default_score=self.default_score,
                        enable_planning_prm=self.enable_planning_prm)
    
    def evaluate_action_candidates(self, task: str, session_history: List[Dict], 
                                 current_state: Dict, action_candidates: List[Dict]) -> List[float]:
        """
        Evaluate multiple action candidates and return their scores.
        
        Args:
            task: The target task description
            session_history: List of previous steps from session
            current_state: Current progress state (JSON-like dict)
            action_candidates: List of action dictionaries with 'thought' and 'code'
            
        Returns:
            List of scores (float) for each candidate, same order as input
        """
        if not action_candidates:
            return []
        
        rprint(f"[RPM] Evaluating {len(action_candidates)} action candidates")
        
        if self.enable_parallel_evaluation and len(action_candidates) > 1:
            # Parallel evaluation - following the pattern from CKAgent
            # Use smaller pool size for stability, similar to existing mrun_pool_size=5
            pool_size = min(3, len(action_candidates))  # Conservative pool size for API stability
            with mp.Pool(pool_size) as pool:
                eval_args = [(task, session_history, current_state, candidate, i) 
                           for i, candidate in enumerate(action_candidates)]
                try:
                    scores = pool.map(self._evaluate_single_candidate_wrapper, eval_args)
                except Exception as e:
                    zwarn(f"[RPM] Parallel evaluation failed: {e}, falling back to sequential")
                    # Fallback to sequential evaluation
                    scores = []
                    for i, candidate in enumerate(action_candidates):
                        score = self._evaluate_single_candidate(task, session_history, current_state, candidate, i)
                        scores.append(score)
        else:
            # Sequential evaluation
            scores = []
            for i, candidate in enumerate(action_candidates):
                score = self._evaluate_single_candidate(task, session_history, current_state, candidate, i)
                scores.append(score)
        
        rprint(f"[RPM] Evaluation scores: {scores}")
        return scores
    
    def evaluate_action_candidates_detailed(self, task: str, session_history: List[Dict], 
                                          current_state: Dict, action_candidates: List[Dict]) -> Tuple[List[float], List[Dict]]:
        """
        Evaluate multiple action candidates and return both scores and detailed evaluation outputs.
        
        Args:
            task: The target task description
            session_history: List of previous steps from session
            current_state: Current progress state (JSON-like dict)
            action_candidates: List of action dictionaries with 'thought' and 'code'
            
        Returns:
            Tuple of (scores, evaluations) where:
            - scores: List of float scores for each candidate
            - evaluations: List of dict containing detailed evaluation info (reasoning, raw output, etc.)
        """
        if not action_candidates:
            return [], []
        
        rprint(f"[RPM] Evaluating {len(action_candidates)} action candidates (detailed mode)")
        
        results = []
        if self.enable_parallel_evaluation and len(action_candidates) > 1:
            pool_size = min(3, len(action_candidates))
            with mp.Pool(pool_size) as pool:
                eval_args = [(task, session_history, current_state, candidate, i) 
                           for i, candidate in enumerate(action_candidates)]
                try:
                    results = pool.map(self._evaluate_single_candidate_detailed_wrapper, eval_args)
                except Exception as e:
                    zwarn(f"[RPM] Parallel detailed evaluation failed: {e}, falling back to sequential")
                    results = []
                    for i, candidate in enumerate(action_candidates):
                        result = self._evaluate_single_candidate_detailed(task, session_history, current_state, candidate, i)
                        results.append(result)
        else:
            # Sequential evaluation
            results = []
            for i, candidate in enumerate(action_candidates):
                result = self._evaluate_single_candidate_detailed(task, session_history, current_state, candidate, i)
                results.append(result)
        
        # Separate scores and evaluation details
        scores = [r['score'] for r in results]
        evaluations = [r['evaluation'] for r in results]
        
        rprint(f"[RPM] Detailed evaluation scores: {scores}")
        return scores, evaluations
    
    def _evaluate_single_candidate_wrapper(self, args):
        """Wrapper for parallel execution"""
        return self._evaluate_single_candidate(*args)
    
    def _evaluate_single_candidate_detailed_wrapper(self, args):
        """Wrapper for parallel execution of detailed evaluation"""
        return self._evaluate_single_candidate_detailed(*args)
    
    def _evaluate_single_candidate_detailed(self, task: str, session_history: List[Dict], 
                                          current_state: Dict, action_candidate: Dict, candidate_idx: int) -> Dict:
        """
        Evaluate a single action candidate and return detailed information.
        
        Returns:
            Dict with 'score' and 'evaluation' keys containing:
            - score: float between 0.0 and 1.0
            - evaluation: dict with reasoning, raw_output, prompt, etc.
        """
        try:
            # Debug: Log what we're receiving in the base class
            rprint(f"[ProcessRewardModel._evaluate_single_candidate_detailed] Candidate {candidate_idx}")
            rprint(f"  Type: {type(action_candidate)}")
            rprint(f"  Keys: {list(action_candidate.keys()) if isinstance(action_candidate, dict) else 'NOT A DICT'}")
            
            # Add small delay for API stability when running in parallel
            if candidate_idx > 0:
                time.sleep(0.5 * candidate_idx)
            
            # Build evaluation prompt
            evaluation_prompt = self._build_evaluation_prompt(
                task, session_history, current_state, action_candidate, candidate_idx
            )
            
            # Call model with retries
            for attempt in range(self.max_retries + 1):
                try:
                    response = self.model(evaluation_prompt)
                    score = self._parse_evaluation_response(response)
                    
                    # Validate score range
                    score = max(self.score_scale[0], min(self.score_scale[1], score))
                    
                    # Build detailed evaluation info
                    evaluation_info = {
                        'candidate_idx': candidate_idx,
                        'raw_output': response,
                        'prompt': evaluation_prompt,
                        'score': score,
                        'reasoning': self._extract_reasoning(response),
                        'attempt': attempt + 1
                    }
                    
                    return {'score': score, 'evaluation': evaluation_info}
                
                except Exception as e:
                    if attempt < self.max_retries:
                        zwarn(f"[RPM] Detailed evaluation attempt {attempt + 1} failed: {e}, retrying...")
                        time.sleep(1 + attempt * 0.5)
                    else:
                        zwarn(f"[RPM] All detailed evaluation attempts failed for candidate {candidate_idx}: {e}")
                        return {
                            'score': self.default_score,
                            'evaluation': {
                                'candidate_idx': candidate_idx,
                                'error': str(e),
                                'score': self.default_score,
                                'reasoning': 'Evaluation failed',
                                'attempt': attempt + 1
                            }
                        }
        
        except Exception as e:
            zwarn(f"[RPM] Critical detailed evaluation error for candidate {candidate_idx}: {e}")
            return {
                'score': self.default_score,
                'evaluation': {
                    'candidate_idx': candidate_idx,
                    'error': str(e),
                    'score': self.default_score,
                    'reasoning': 'Critical error',
                    'attempt': 0
                }
            }
    
    def _evaluate_single_candidate(self, task: str, session_history: List[Dict], 
                                 current_state: Dict, action_candidate: Dict, candidate_idx: int) -> float:
        """
        Evaluate a single action candidate.
        
        Args:
            task: The target task description
            session_history: List of previous steps
            current_state: Current progress state
            action_candidate: Single action dict with 'thought' and 'code'
            candidate_idx: Index of this candidate (for logging)
            
        Returns:
            Score between 0.0 and 1.0
        """
        try:
            # Add small delay for API stability when running in parallel
            if candidate_idx > 0:
                time.sleep(0.5 * candidate_idx)  # Staggered delays to avoid rate limits
            
            # Build evaluation prompt
            evaluation_prompt = self._build_evaluation_prompt(
                task, session_history, current_state, action_candidate, candidate_idx
            )
            
            # Call model with retries
            for attempt in range(self.max_retries + 1):
                try:
                    response = self.model(evaluation_prompt)
                    score = self._parse_evaluation_response(response)
                    
                    # Validate score range
                    score = max(self.score_scale[0], min(self.score_scale[1], score))
                    return score
                
                except Exception as e:
                    if attempt < self.max_retries:
                        zwarn(f"[RPM] Evaluation attempt {attempt + 1} failed: {e}, retrying...")
                        # Exponential backoff for retries
                        time.sleep(1 + attempt * 0.5)
                    else:
                        zwarn(f"[RPM] All evaluation attempts failed for candidate {candidate_idx}: {e}")
                        return self.default_score
        
        except Exception as e:
            zwarn(f"[RPM] Critical evaluation error for candidate {candidate_idx}: {e}")
            return self.default_score
    
    def _build_evaluation_prompt(self, task: str, session_history: List[Dict], 
                               current_state: Dict, action_candidate: Dict, candidate_idx: int) -> List[Dict]:
        """
        Build the evaluation prompt for the reward model.
        Uses similar format to existing agent prompts but focused on evaluation.
        """
        
        # Format session history similar to existing agent format
        history_str = self._format_session_history(session_history)
        state_str = json.dumps(current_state, ensure_ascii=False, indent=2)
        
        # Extract action components
        thought = action_candidate.get('thought', 'No thought provided')
        code = action_candidate.get('code', 'No code provided')
        
        system_prompt = self._get_rpm_system_prompt()
        user_prompt = f"""## Target Task
{task}

## Current Progress State
{state_str}

## Recent Execution History
{history_str}

## Action Candidate to Evaluate
**Thought**: {thought}
**Code**: 
```python
{code}
```

## Evaluation Request
Please evaluate this action candidate based on the rubric provided in the system prompt. 
Provide your evaluation in the exact format specified: reasoning followed by numerical score.
"""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def _format_session_history(self, session_history: List[Dict]) -> str:
        """Format session history similar to existing agent format, with truncation to avoid long inputs"""
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
            
            # Truncate thought and code if too long (use obs_max_token as unified truncation limit)
            max_thought_code_len = getattr(self, 'obs_max_token', 8192)
            if len(thought) > max_thought_code_len:
                thought = f"{thought[:max_thought_code_len]} ... (truncated)"
            if len(code) > max_thought_code_len:
                code = f"{code[:max_thought_code_len]} ... (truncated)"
            
            step_str = f"""### Step {step_idx}
Thought: {thought}
Action: ```
{code}
```
Observation: {self._format_observation(observation)}"""
            formatted_steps.append(step_str)
        
        return "\n\n".join(formatted_steps)
    
    def _format_observation(self, observation) -> str:
        """Format observation similar to existing agent format, with truncation to avoid long inputs"""
        if isinstance(observation, (list, tuple)):
            ret = "\n".join([f"- Result {i}: {str(obs)}" for i, obs in enumerate(observation)])
        else:
            ret = str(observation)
        
        # Truncate if too long (similar to agent's get_obs_str)
        max_len = getattr(self, 'obs_max_token', 8192)
        if len(ret) > max_len:
            ret = f"{ret[:max_len]} ... (observation truncated: exceeded {max_len} characters)"
        return ret
    
    def _get_rpm_system_prompt(self) -> str:
        """
        System prompt for the Process Reward Model.
        Defines evaluation rubric and scoring criteria.
        Enhanced version with stricter code quality checks and better score discrimination.
        """
        return """You are a Process Reward Model (RPM) responsible for critically evaluating the quality of agent actions during task execution. Your role is to rigorously assess whether a proposed action is likely to make meaningful progress toward completing the given task.

**IMPORTANT**: You are a STRICT evaluator. Most actions should score between 0.4-0.8. Only truly exceptional actions deserve scores above 0.85. Be critical and look for flaws.

**CRITICAL CAUTION ON DATA RELIABILITY**:
- Answers or fields retrieved from Hugging Face GAIA datasets (e.g., dataset-provided "answer"/metadata fields) are OFTEN WRONG and MUST NOT be treated as ground truth.
- Penalize any action that blindly copies, trusts, or cites GAIA dataset fields without independent verification from reliable sources or prior validated state/history.
- Actions should explicitly verify claims and cross-check sources; lack of verification is a scoring liability.

## Evaluation Rubric

Evaluate each action based on the following criteria. Be extremely rigorous in your assessment:

### 1. Code Correctness & Detail (Weight: 35%) - CRITICAL
- **1.0**: Code is flawless with perfect syntax, logic, edge case handling, and data type management
- **0.85**: Code is correct but could be more robust (missing 1-2 minor edge cases)
- **0.7**: Code is mostly correct but has 2-3 potential issues (type mismatches, off-by-one errors, missing error handling)
- **0.5**: Code has notable bugs that will likely cause partial failures
- **0.3**: Code has major logical errors or will fail in most cases
- **0.0**: Code is fundamentally broken or will definitely fail

**Critical checks for complex code**:
- Variable initialization and scope
- Loop boundaries and termination conditions
- List/array indexing (off-by-one errors are common!)
- Type compatibility (string vs int, list vs dict)
- Error handling and edge cases
- Function call signatures and return values
- Import statements and dependencies

### 2. Task Relevance (Weight: 25%)
- **1.0**: Action perfectly addresses the exact task requirement with optimal approach
- **0.75**: Action addresses the task well but approach is not optimal
- **0.5**: Action is relevant but takes an indirect or inefficient path
- **0.25**: Action has minimal relevance or addresses the wrong aspect
- **0.0**: Action is completely irrelevant or counterproductive

### 3. Logical Progression (Weight: 20%)
- **1.0**: Action perfectly builds on previous steps, avoiding redundancy and utilizing all prior results
- **0.75**: Action follows logically but may repeat some work unnecessarily
- **0.5**: Action makes sense but shows gaps in utilizing previous progress
- **0.25**: Action shows weak logical connection to previous steps
- **0.0**: Action contradicts or ignores previous progress

### 4. Information Utilization (Weight: 15%)
- **1.0**: Leverages ALL relevant information from state, history, and task description
- **0.75**: Uses most key information effectively
- **0.5**: Uses some information but misses important details from state or history
- **0.25**: Mostly ignores available information
- **0.0**: Completely fails to utilize or actively misuses available information
- NOTE: Reliance on unverified dataset-provided answers (e.g., GAIA fields) counts as misuse unless independently verified.

### 5. Thought Quality & Planning (Weight: 5%)
- **1.0**: Thought shows deep understanding with clear, detailed reasoning
- **0.75**: Thought is clear and logical
- **0.5**: Thought is vague or shows partial understanding
- **0.25**: Thought is unclear or shows misunderstanding
- **0.0**: Thought is missing or completely wrong

## Strict Scoring Guidelines

**BE CRITICAL - Most actions should score 0.4-0.8:**

- **0.9-1.0** (Exceptional - RARE): Near-perfect code with excellent thought, optimal approach, perfect logic
- **0.8-0.89** (Excellent): Very good code with minor room for improvement, strong thought and logic
- **0.7-0.79** (Good): Solid code with 1-2 fixable issues, reasonable approach
- **0.6-0.69** (Acceptable): Code works but has several issues or suboptimal approach
- **0.5-0.59** (Mediocre): Code has notable problems, weak logic or poor information use
- **0.4-0.49** (Poor): Significant code issues or wrong approach, likely to fail partially
- **0.3-0.39** (Bad): Major flaws in code or logic, will likely fail
- **0.0-0.29** (Failure): Fundamentally broken or irrelevant

## Common Code Pitfalls to Penalize

When evaluating complex code, specifically check for:

1. **Off-by-one errors** in loops and indexing (reduce score by 0.15-0.25)
2. **Type mismatches** (string concatenation with ints, etc.) (reduce by 0.1-0.2)
3. **Missing imports** for used libraries (reduce by 0.1-0.15)
4. **Variable name typos** or inconsistent naming (reduce by 0.05-0.15)
5. **Edge case failures** (empty list, None values, zero division) (reduce by 0.1-0.2)
6. **Incorrect function signatures** for tools/sub-agents (reduce by 0.2-0.3)
7. **Missing error handling** in critical sections (reduce by 0.05-0.15)
8. **Inefficient algorithms** when better approaches exist (reduce by 0.05-0.15)
9. **Redundant work** that ignores previous results (reduce by 0.1-0.2)
10. **Poor data structure choice** (reduce by 0.05-0.1)
11. **Unverified or incorrect source usage** (reduce by 0.2-0.4): blindly trusting dataset-provided fields (e.g., GAIA answers/metadata) without verification; fabricating or misattributing sources; failing to cross-check with retrieved content or authoritative references.
12. **Failure to identify the error source** when a mistake occurs (reduce by 0.1-0.2): reasoning should pinpoint whether the issue came from dataset fields, parsing, tool output, stale state, or coding logic.

## Response Format

Provide your evaluation in exactly this format:

**Reasoning**: [3-4 sentences providing specific, detailed assessment. Mention specific code issues if any, explain scoring decisions, reference the rubric criteria. If a mistake exists, explicitly identify the ERROR SOURCE (e.g., “relied on GAIA dataset field likely incorrect”, “misparsed web content”, “stale state variable”, “wrong tool signature”).]

**Score**: [Single decimal number between 0.0 and 1.0. Be conservative - most scores should be 0.5-0.75]

## Calibrated Examples

**Exceptional Action (0.92)**:
- Reasoning: Code is syntactically perfect with proper edge case handling for empty inputs and type checking. The action directly addresses the task using optimal data structures (dict instead of nested loops). It builds perfectly on previous file download results from state. Thought demonstrates deep understanding. Only trivial improvement: could add more verbose logging.
- Score: 0.92

**Good Action (0.73)**:
- Reasoning: Code is mostly correct and addresses the task well, but has a potential off-by-one error in the loop (range(len(items)) should be range(len(items)-1) for comparison). Thought is clear and logical. Uses most available information from state. Minor inefficiency in using two passes when one would suffice.
- Score: 0.73

**Acceptable Action (0.61)**:
- Reasoning: Action is relevant but code has three issues: missing import for json library, doesn't check if 'results' key exists before accessing (will crash on KeyError), and inefficient nested loop could be O(n log n) with sorting. Thought shows understanding but missed these details. Uses state but ignores helpful info from history.
- Score: 0.61

**Poor Action (0.42)**:
- Reasoning: Code has major logical flaw - tries to concatenate string with int without conversion, and the loop termination condition is wrong (will cause infinite loop or early exit). Action is somewhat relevant to task but takes inefficient path. Ignores directly relevant information available in current state. Thought is too vague.
- Score: 0.42

**Failed Action (0.18)**:
- Reasoning: Code will immediately fail - calls undefined function, has syntax error in f-string, and uses wrong variable name. Action addresses wrong aspect of the task. Completely ignores previous step results that directly provide needed data. Thought shows fundamental misunderstanding of the requirement.
- Score: 0.18
"""
    
    def _parse_evaluation_response(self, response: str) -> float:
        """
        Parse the evaluation response to extract the numerical score.
        """
        try:
            # Look for "Score: X.X" pattern
            import re
            score_match = re.search(r'(?:Score|score):\s*([0-9]*\.?[0-9]+)', response)
            if score_match:
                score = float(score_match.group(1))
                return score
            
            # Fallback: look for any decimal number between 0 and 1
            decimal_matches = re.findall(r'\b([0-1](?:\.[0-9]+)?)\b', response)
            if decimal_matches:
                # Take the last decimal found (likely the final score)
                score = float(decimal_matches[-1])
                return score
            
            # If no valid score found, try to extract from the end
            lines = response.strip().split('\n')
            for line in reversed(lines):
                numbers = re.findall(r'\b([0-1](?:\.[0-9]+)?)\b', line)
                if numbers:
                    return float(numbers[-1])
            
            zwarn(f"[RPM] Could not parse score from response: {response[:100]}...")
            return self.default_score
            
        except (ValueError, AttributeError) as e:
            zwarn(f"[RPM] Error parsing evaluation score: {e}")
            return self.default_score
    
    def _extract_reasoning(self, response: str) -> str:
        """
        Extract the reasoning text from the evaluation response.
        """
        try:
            import re
            # Look for "Reasoning:" or "**Reasoning**:" pattern
            reasoning_match = re.search(r'(?:\*\*)?Reasoning(?:\*\*)?:\s*(.+?)(?=\n\s*(?:\*\*)?Score|$)', response, re.DOTALL | re.IGNORECASE)
            if reasoning_match:
                return reasoning_match.group(1).strip()
            
            # If not found, return first paragraph or first 200 chars
            lines = response.strip().split('\n')
            for line in lines:
                if line.strip() and not line.strip().startswith('Score'):
                    return line.strip()
            
            # Fallback: return first 200 chars
            return response[:200] if response else "No reasoning extracted"
            
        except Exception as e:
            zwarn(f"[RPM] Error extracting reasoning: {e}")
            return response[:200] if response else "Error extracting reasoning"
    
    def get_call_stat(self, clear: bool = False):
        """Get model call statistics, consistent with agent interface"""
        return self.model.get_call_stat(clear=clear)
    
    def set_seed(self, seed):
        """Set random seed for reproducibility"""
        self.model.set_seed(seed)
