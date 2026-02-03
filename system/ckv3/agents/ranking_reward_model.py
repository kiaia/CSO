#
# Ranking-based Process Reward Model
# Instead of scoring each candidate individually, this model ranks all candidates together

import time
import json
from typing import List, Dict, Any, Tuple
from ..agents.utils import rprint, zwarn, KwargsInitializable
from ..agents.model import LLM


class RankingRewardModel(KwargsInitializable):
    """
    Ranking-based Process Reward Model.
    
    Instead of scoring each candidate individually (score-based PRM), 
    this model evaluates all candidates together and returns a ranking.
    
    Key differences from ProcessRewardModel:
    - Takes all candidates at once
    - Returns a ranking (best to worst) instead of individual scores
    - More efficient as context is shared
    - Better for relative comparison
    """
    
    def __init__(self, **kwargs):
        self.call_target = kwargs.pop('call_target', 'gpt:gpt-4.1')
        self.evaluation_timeout = kwargs.pop('evaluation_timeout', 60)
        self.max_retries = kwargs.pop('max_retries', 2)
        
        # Initialize the model
        self.model = LLM(call_target=self.call_target, _default_init=True)
        
        super().__init__(call_target=self.call_target,
                        evaluation_timeout=self.evaluation_timeout,
                        max_retries=self.max_retries)
        
        rprint(f"[RankingRewardModel] Initialized with model: {self.call_target}")
    
    def rank_candidates(self, task: str, session_history: List[Dict], 
                       current_state: Dict, candidates: List[Dict], 
                       is_planning: bool = False) -> Tuple[List[int], Dict]:
        """
        Rank all candidates together and return the ranking.
        
        Args:
            task: The target task description
            session_history: List of previous steps from session
            current_state: Current progress state (JSON-like dict)
            candidates: List of candidate dictionaries (with 'thought' and 'code')
            is_planning: Whether this is for planning (True) or action (False)
            
        Returns:
            Tuple of (ranking_indices, evaluation_info) where:
            - ranking_indices: List of indices in order from best to worst [0, 2, 1] means candidate 0 is best
            - evaluation_info: Dict with reasoning, raw_output, etc.
        """
        if not candidates:
            return [], {}
        
        if len(candidates) == 1:
            # Only one candidate, no need to rank
            return [0], {'reasoning': 'Only one candidate', 'raw_output': 'N/A'}
        
        rprint(f"[RankingRewardModel] Ranking {len(candidates)} {'planning' if is_planning else 'action'} candidates")
        
        try:
            # Build the ranking prompt
            prompt = self._build_ranking_prompt(
                task, session_history, current_state, candidates, is_planning
            )
            
            # Call the model with retries
            for attempt in range(self.max_retries + 1):
                try:
                    response = self.model(prompt, temperature=0.0, max_tokens=4096)
                    
                    # Parse the ranking from response
                    ranking = self._parse_ranking_response(response, len(candidates))
                    
                    evaluation_info = {
                        'raw_output': response,
                        'reasoning': self._extract_reasoning(response),
                        'prompt': prompt,
                        'attempt': attempt + 1,
                        'num_candidates': len(candidates)
                    }
                    
                    rprint(f"[RankingRewardModel] Ranking result: {ranking}")
                    return ranking, evaluation_info
                
                except Exception as e:
                    if attempt < self.max_retries:
                        zwarn(f"[RankingRewardModel] Ranking attempt {attempt + 1} failed: {e}, retrying...")
                        time.sleep(1 + attempt * 0.5)
                    else:
                        zwarn(f"[RankingRewardModel] All ranking attempts failed: {e}")
                        # Return default ranking (original order)
                        return list(range(len(candidates))), {
                            'error': str(e),
                            'reasoning': 'Ranking failed, using default order',
                            'raw_output': 'N/A'
                        }
        
        except Exception as e:
            zwarn(f"[RankingRewardModel] Critical ranking error: {e}")
            return list(range(len(candidates))), {'error': str(e)}
    
    def _build_ranking_prompt(self, task: str, session_history: List[Dict], 
                             current_state: Dict, candidates: List[Dict],
                             is_planning: bool) -> List[Dict]:
        """Build the ranking prompt with shared context and multiple candidates"""
        
        # Format session history
        history_str = self._format_session_history(session_history)
        state_str = json.dumps(current_state, ensure_ascii=False, indent=2)
        
        # Format all candidates
        candidates_str = self._format_candidates(candidates, is_planning)
        
        phase_name = "Planning" if is_planning else "Action"
        
        system_prompt = self._get_ranking_system_prompt(is_planning)
        
        user_prompt = f"""## Target Task
{task}

## Current Progress State
```json
{state_str}
```

## Recent Execution History
{history_str}

## {phase_name} Candidates to Rank

{candidates_str}

## Ranking Request
Please carefully evaluate all candidates and rank them from best to worst. Consider:
1. **Correctness**: Does the approach correctly address the task?
2. **Efficiency**: Is the solution efficient and well-structured?
3. **Completeness**: Does it cover all necessary aspects?
4. **Robustness**: Can it handle edge cases and errors?
5. **Clarity**: Is the reasoning clear and code well-organized?

Provide your evaluation in the following format:
1. **Analysis**: Brief analysis of each candidate's strengths and weaknesses
2. **Ranking**: List the candidate numbers from best to worst (e.g., "Ranking: 2, 0, 1")
3. **Justification**: Explain why the top-ranked candidate is the best choice

Your response must end with a clear ranking line like:
**Final Ranking: 2, 0, 1** (where numbers are 0-indexed candidate positions)
"""
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def _format_candidates(self, candidates: List[Dict], is_planning: bool) -> str:
        """Format all candidates for the prompt"""
        formatted = []
        
        for i, candidate in enumerate(candidates):
            # Handle both formats: thought/code and reasoning/plan
            if is_planning:
                thought = candidate.get('reasoning', candidate.get('thought', 'No reasoning provided'))
                code = candidate.get('plan', candidate.get('code', 'No plan provided'))
            else:
                thought = candidate.get('thought', 'No thought provided')
                code = candidate.get('code', 'No code provided')
            
            # Format code properly if it's a dict
            if isinstance(code, dict):
                code = json.dumps(code, ensure_ascii=False, indent=2)
            
            candidate_str = f"""### Candidate {i}

**Reasoning/Thought**:
{thought}

**Code/Plan**:
```python
{code}
```
"""
            formatted.append(candidate_str)
        
        return "\n".join(formatted)
    
    def _get_ranking_system_prompt(self, is_planning: bool) -> str:
        """Get the system prompt for ranking"""
        
        if is_planning:
            return """You are an expert Planning Evaluator responsible for ranking multiple planning candidates for agent task execution.

Your role is to compare all planning candidates and determine which one is most likely to lead to successful task completion.

**Evaluation Criteria for Planning:**

1. **Task Coverage (35%)**: Does the plan address all aspects of the task with proper sequencing?
2. **Feasibility (25%)**: Is the plan realistic and executable with available resources?
3. **Adaptability (20%)**: Does the plan account for potential issues and alternatives?
4. **Information Utilization (15%)**: Does it effectively use available context and history?
5. **Clarity & Structure (5%)**: Is the plan well-organized and easy to follow?

**Instructions:**
- Evaluate ALL candidates carefully
- Consider relative strengths and weaknesses
- Provide clear justification for your ranking
- End with explicit ranking: "**Final Ranking: X, Y, Z**" where X is best
"""
        else:
            return """You are an expert Action Evaluator responsible for ranking multiple action candidates for agent task execution.

Your role is to compare all action candidates and determine which one is most likely to successfully advance toward the task goal.

**Evaluation Criteria for Actions:**

1. **Correctness (40%)**: Does the action correctly address the current step?
2. **Progress (25%)**: Does it make meaningful progress toward the goal?
3. **Efficiency (15%)**: Is the approach efficient without unnecessary steps?
4. **Robustness (15%)**: Does it handle errors and edge cases properly?
5. **Code Quality (5%)**: Is the code clear, correct, and well-structured?

**Instructions:**
- Evaluate ALL candidates carefully
- Consider which action will best advance the task
- Provide clear justification for your ranking
- End with explicit ranking: "**Final Ranking: X, Y, Z**" where X is best
"""
    
    def _parse_ranking_response(self, response: str, num_candidates: int) -> List[int]:
        """
        Parse the ranking from the model's response.
        Looks for patterns like "Final Ranking: 2, 0, 1" or "Ranking: 2, 0, 1"
        """
        import re
        
        # Try to find ranking pattern
        patterns = [
            r'\*\*Final Ranking[:\s]+([0-9,\s]+)\*\*',
            r'Final Ranking[:\s]+([0-9,\s]+)',
            r'\*\*Ranking[:\s]+([0-9,\s]+)\*\*',
            r'Ranking[:\s]+([0-9,\s]+)',
            r'Best to worst[:\s]+([0-9,\s]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                ranking_str = match.group(1)
                # Parse the numbers
                try:
                    ranking = [int(x.strip()) for x in ranking_str.split(',')]
                    # Validate ranking
                    if len(ranking) == num_candidates and set(ranking) == set(range(num_candidates)):
                        return ranking
                except:
                    continue
        
        # If parsing fails, try to extract any sequence of numbers
        numbers = re.findall(r'\b([0-9])\b', response)
        if numbers:
            ranking = []
            for num_str in numbers:
                num = int(num_str)
                if num < num_candidates and num not in ranking:
                    ranking.append(num)
            
            if len(ranking) == num_candidates:
                return ranking
        
        # Default: return original order
        zwarn(f"[RankingRewardModel] Could not parse ranking from response, using default order")
        return list(range(num_candidates))
    
    def _extract_reasoning(self, response: str) -> str:
        """Extract the reasoning/justification from the response"""
        # Try to extract text before the final ranking
        import re
        
        # Find the final ranking line
        match = re.search(r'(\*\*)?Final Ranking', response, re.IGNORECASE)
        if match:
            # Return everything before the final ranking
            reasoning = response[:match.start()].strip()
            if reasoning:
                return reasoning
        
        # If no clear split, return first 500 chars
        return response[:500] + "..." if len(response) > 500 else response
    
    def _format_session_history(self, session_history: List[Dict]) -> str:
        """Format session history similar to ProcessRewardModel"""
        if not session_history:
            return "No previous steps."
        
        # Take last 3 steps to avoid too long context
        recent_steps = session_history[-3:] if len(session_history) > 3 else session_history
        
        formatted_steps = []
        for i, step in enumerate(recent_steps):
            step_num = len(session_history) - len(recent_steps) + i
            
            # Get plan info
            plan_info = step.get('plan', {})
            plan_thought = plan_info.get('thought', 'N/A') if isinstance(plan_info, dict) else 'N/A'
            
            # Get action info
            action_info = step.get('action', {})
            action_thought = action_info.get('thought', 'N/A') if isinstance(action_info, dict) else 'N/A'
            action_obs = action_info.get('observation', 'N/A') if isinstance(action_info, dict) else 'N/A'
            
            formatted_steps.append(f"""### Step {step_num}
**Plan**: {plan_thought[:200]}...
**Action**: {action_thought[:200]}...
**Observation**: {str(action_obs)[:200]}...""")
        
        return "\n\n".join(formatted_steps)
    
    def get_call_stat(self, clear: bool = False):
        """Get model call statistics, consistent with agent interface"""
        return self.model.get_call_stat(clear=clear)
    
    def set_seed(self, seed):
        """Set random seed for reproducibility"""
        self.model.set_seed(seed)

