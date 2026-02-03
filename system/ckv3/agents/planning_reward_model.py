# 
# Planning Reward Model for evaluating planning quality during agent execution
# Based on the Process Reward Model architecture

import json
import time
import multiprocessing as mp
from functools import partial
from typing import List, Dict, Any, Tuple
from .model import LLM
from .utils import rprint, zwarn, KwargsInitializable
from .process_reward_model import ProcessRewardModel


class PlanningRewardModel(ProcessRewardModel):
    """
    Planning Reward Model for evaluating the quality of agent planning during execution.
    Extends the ProcessRewardModel but focuses on evaluating planning steps specifically.
    """
    
    def __init__(self, **kwargs):
        # Planning specific parameters - set these before calling super().__init__
        self.planning_score_weight = kwargs.pop('planning_score_weight', 0.7)  # Weight for planning score vs diversity
        self.diversity_weight = kwargs.pop('diversity_weight', 0.3)  # Weight for diversity score
        
        # Initialize with parent class constructor
        super().__init__(**kwargs)
        
    def _build_evaluation_prompt(self, task: str, session_history: List[Dict], 
                             current_state: Dict, plan_candidate: Dict, candidate_idx: int) -> List[Dict]:
        """
        Build the evaluation prompt for the planning reward model.
        Uses similar format to existing agent prompts but focused on plan evaluation.
        """
        
        # Debug: Log what we're receiving
        rprint(f"[PlanningRewardModel] Building prompt for candidate {candidate_idx}")
        rprint(f"  Candidate keys: {list(plan_candidate.keys())}")
        rprint(f"  Has 'reasoning': {'reasoning' in plan_candidate}")
        rprint(f"  Has 'plan': {'plan' in plan_candidate}")
        
        # Format session history similar to existing agent format
        history_str = self._format_session_history(session_history)
        state_str = json.dumps(current_state, ensure_ascii=False, indent=2)
        
        # Extract action components - support both formats
        # Planning format: 'reasoning' and 'plan'
        # Agent format: 'thought' and 'code'
        # This handles both cases for robustness
        plan_raw = plan_candidate.get('plan', plan_candidate.get('code', 'No plan provided'))
        reasoning_raw = plan_candidate.get('reasoning', plan_candidate.get('thought', 'No reasoning provided'))
        
        # Debug: Log extracted values
        rprint(f"  Extracted reasoning: {reasoning_raw[:80] if isinstance(reasoning_raw, str) else reasoning_raw}...")
        rprint(f"  Extracted plan type: {type(plan_raw)}")
        
        # Format plan properly - if it's a dict or code, format it nicely
        if isinstance(plan_raw, dict):
            plan = json.dumps(plan_raw, ensure_ascii=False, indent=2)
        elif isinstance(plan_raw, str):
            plan = plan_raw
        else:
            plan = str(plan_raw)
        
        # Format reasoning properly
        if isinstance(reasoning_raw, str):
            reasoning = reasoning_raw
        else:
            reasoning = str(reasoning_raw)
        
        system_prompt = self._get_planning_rpm_system_prompt()
        user_prompt = f"""## Target Task
{task}

## Current Progress State
{state_str}

## Recent Execution History
{history_str}

## Plan Candidate to Evaluate
**Reasoning**: {reasoning}

**Plan**: 
```python
{plan}
```

## Evaluation Request
Please evaluate this plan candidate based on the rubric provided in the system prompt. 
Provide your evaluation in the exact format specified: reasoning followed by numerical score.
"""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def _get_planning_rpm_system_prompt(self) -> str:
        """
        System prompt for the Planning Reward Model.
        Defines evaluation rubric and scoring criteria specific to plans.
        """
        return """You are a Planning Reward Model (PRM) responsible for critically evaluating the quality of agent plans during task execution. Your role is to rigorously assess whether a proposed plan is likely to lead to successful completion of the given task.

**IMPORTANT**: You are a STRICT evaluator. Most plans should score between 0.4-0.8. Only truly exceptional plans deserve scores above 0.85. Be critical and look for flaws.

## Evaluation Rubric

Evaluate each plan based on the following criteria. Be extremely rigorous in your assessment:

### 1. Task Coverage & Completeness (Weight: 35%) - CRITICAL
- **1.0**: Plan covers all aspects of the task with perfect step sequence and dependency handling
- **0.85**: Plan covers all major aspects but could have minor improvements in sequencing
- **0.7**: Plan covers most aspects but misses 1-2 minor details or has suboptimal ordering
- **0.5**: Plan has notable gaps in coverage or significant sequencing issues
- **0.3**: Plan misses major aspects of the task or has serious logical flaws
- **0.0**: Plan is fundamentally inadequate or irrelevant to the task

### 2. Feasibility & Practicality (Weight: 25%)
- **1.0**: Plan is perfectly feasible with optimal resource usage and realistic timeline
- **0.75**: Plan is feasible but could be more efficient in some areas
- **0.5**: Plan is mostly feasible but has some impractical elements or inefficient steps
- **0.25**: Plan has significant feasibility issues or unrealistic expectations
- **0.0**: Plan is completely impractical or impossible to execute

### 3. Adaptability & Robustness (Weight: 20%)
- **1.0**: Plan accounts for potential issues and includes contingencies/alternatives
- **0.75**: Plan acknowledges potential issues but contingencies could be more detailed
- **0.5**: Plan shows some awareness of potential issues but limited contingency planning
- **0.25**: Plan shows minimal consideration for potential issues
- **0.0**: Plan completely ignores potential issues or failure modes

### 4. Information Utilization (Weight: 15%)
- **1.0**: Leverages ALL relevant information from state, history, and task description
- **0.75**: Uses most key information effectively
- **0.5**: Uses some information but misses important details from state or history
- **0.25**: Mostly ignores available information
- **0.0**: Completely fails to utilize or actively misuses available information

### 5. Clarity & Structure (Weight: 5%)
- **1.0**: Plan is exceptionally clear, well-structured, and easy to follow
- **0.75**: Plan is clear and logically structured
- **0.5**: Plan is somewhat clear but structure could be improved
- **0.25**: Plan is unclear or poorly structured
- **0.0**: Plan is chaotic, confusing, or lacks any meaningful structure

## Strict Scoring Guidelines

**BE CRITICAL - Most plans should score 0.4-0.8:**

- **0.9-1.0** (Exceptional - RARE): Near-perfect plan with excellent coverage, feasibility, adaptability and clarity
- **0.8-0.89** (Excellent): Very good plan with minor room for improvement
- **0.7-0.79** (Good): Solid plan with 1-2 fixable issues, reasonable approach
- **0.6-0.69** (Acceptable): Plan works but has several issues or suboptimal approaches
- **0.5-0.59** (Mediocre): Plan has notable problems, weak adaptability or poor information use
- **0.4-0.49** (Poor): Significant planning issues or wrong approach, likely to lead to partial failure
- **0.3-0.39** (Bad): Major flaws in plan, will likely fail
- **0.0-0.29** (Failure): Fundamentally broken or irrelevant plan

## Common Planning Pitfalls to Penalize

When evaluating plans, specifically check for:

1. **Missing dependencies** between steps (reduce score by 0.15-0.25)
2. **Unrealistic time estimates** for complex tasks (reduce by 0.1-0.2)
3. **Lack of specificity** in critical steps (reduce by 0.1-0.15)
4. **Poor prioritization** of important vs. trivial tasks (reduce by 0.05-0.15)
5. **Missing error handling** or contingency plans (reduce by 0.1-0.2)
6. **Resource conflicts** in parallel activities (reduce by 0.2-0.3)
7. **Ignoring constraints** from the task or environment (reduce by 0.05-0.15)
8. **Inefficient sequencing** when better approaches exist (reduce by 0.05-0.15)
9. **Redundant steps** that waste resources (reduce by 0.1-0.2)
10. **Unrealistic assumptions** about available tools/capabilities (reduce by 0.05-0.1)

## Response Format

Provide your evaluation in exactly this format:

**Reasoning**: [3-4 sentences providing specific, detailed assessment. Mention specific plan issues if any, explain scoring decisions, reference the rubric criteria]

**Score**: [Single decimal number between 0.0 and 1.0. Be conservative - most scores should be 0.5-0.75]
"""
    
    def calculate_diversity_score(self, plan_candidates: List[Dict]) -> List[float]:
        """
        Calculate diversity scores between plan candidates.
        Higher score means more diverse (different) from other plans.
        
        Args:
            plan_candidates: List of plan dictionaries with 'plan' and 'reasoning' fields
            
        Returns:
            List of diversity scores (float) for each candidate
        """
        if len(plan_candidates) <= 1:
            return [1.0]  # Only one candidate is maximally diverse
        
        # Extract plans and reasoning texts
        plans = [c.get('plan', '') for c in plan_candidates]
        reasonings = [c.get('reasoning', '') for c in plan_candidates]
        
        # Simple text similarity metric
        diversity_scores = []
        
        for i in range(len(plan_candidates)):
            # Compare current candidate with all others
            similarities = []
            for j in range(len(plan_candidates)):
                if i == j:
                    continue
                
                # Calculate simple text overlap similarity (can be replaced with embedding similarity)
                plan_sim = self._text_similarity(plans[i], plans[j])
                reasoning_sim = self._text_similarity(reasonings[i], reasonings[j])
                
                # Combined similarity with weighted components
                combined_sim = 0.7 * plan_sim + 0.3 * reasoning_sim
                similarities.append(combined_sim)
            
            # Average similarity with other candidates
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
            
            # Convert similarity to diversity (1 - similarity)
            diversity = 1.0 - avg_similarity
            diversity_scores.append(diversity)
        
        return diversity_scores
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate simple text similarity based on word overlap.
        Returns value between 0.0 (completely different) and 1.0 (identical).
        """
        if not text1 or not text2:
            return 0.0
        
        # Convert to lowercase and split into words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def evaluate_plan_candidates(self, task: str, session_history: List[Dict], 
                               current_state: Dict, plan_candidates: List[Dict]) -> List[float]:
        """
        Evaluate multiple plan candidates and return their scores.
        
        Args:
            task: The target task description
            session_history: List of previous steps from session
            current_state: Current progress state (JSON-like dict)
            plan_candidates: List of plan dictionaries with 'plan' and 'reasoning'
            
        Returns:
            List of scores (float) for each candidate, same order as input
        """
        if not plan_candidates:
            return []
        
        rprint(f"[PRM] Evaluating {len(plan_candidates)} planning candidates")
        
        # Get base evaluation scores for each plan
        base_scores = super().evaluate_action_candidates(task, session_history, current_state, plan_candidates)
        
        # Calculate diversity scores
        diversity_scores = self.calculate_diversity_score(plan_candidates)
        
        # Combine scores with weighting
        combined_scores = []
        for i in range(len(plan_candidates)):
            combined = (self.planning_score_weight * base_scores[i] + 
                      self.diversity_weight * diversity_scores[i])
            combined_scores.append(combined)
        
        rprint(f"[PRM] Planning evaluation scores: {combined_scores}")
        return combined_scores
    
    def evaluate_plan_candidates_detailed(self, task: str, session_history: List[Dict], 
                                       current_state: Dict, plan_candidates: List[Dict]) -> Tuple[List[float], List[Dict]]:
        """
        Evaluate multiple plan candidates and return both scores and detailed evaluation outputs.
        
        Args:
            task: The target task description
            session_history: List of previous steps from session
            current_state: Current progress state (JSON-like dict)
            plan_candidates: List of plan dictionaries with 'plan' and 'reasoning'
            
        Returns:
            Tuple of (scores, evaluations) where:
            - scores: List of float scores for each candidate
            - evaluations: List of dict containing detailed evaluation info (reasoning, raw output, etc.)
        """
        if not plan_candidates:
            return [], []
        
        rprint(f"[PRM] Evaluating {len(plan_candidates)} planning candidates (detailed mode)")
        
        # Debug: Log what candidates we're receiving
        rprint(f"[PRM] Planning candidates keys:")
        for i, cand in enumerate(plan_candidates):
            rprint(f"  Candidate {i}: keys={list(cand.keys())}")
        
        # Get base evaluation scores for each plan
        # Note: We pass plan_candidates (with 'reasoning' and 'plan' fields)
        # and rely on our overridden _build_evaluation_prompt to handle them
        base_scores, base_evaluations = super().evaluate_action_candidates_detailed(
            task, session_history, current_state, plan_candidates)
        
        # Calculate diversity scores
        diversity_scores = self.calculate_diversity_score(plan_candidates)
        
        # Combine scores with weighting
        combined_scores = []
        combined_evaluations = []
        
        for i in range(len(plan_candidates)):
            combined = (self.planning_score_weight * base_scores[i] + 
                      self.diversity_weight * diversity_scores[i])
            combined_scores.append(combined)
            
            # Add diversity information to evaluations
            eval_with_diversity = base_evaluations[i].copy()
            eval_with_diversity['diversity_score'] = diversity_scores[i]
            eval_with_diversity['base_score'] = base_scores[i]
            eval_with_diversity['combined_score'] = combined
            combined_evaluations.append(eval_with_diversity)
        
        rprint(f"[PRM] Planning detailed evaluation scores: {combined_scores}")
        return combined_scores, combined_evaluations
