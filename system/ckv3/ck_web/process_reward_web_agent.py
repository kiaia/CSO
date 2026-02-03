#
# Process Reward enabled Web Agent
# Similar to ProcessRewardCKAgent but for WebAgent

import time
import random
import multiprocessing as mp
from functools import partial
from typing import List, Dict, Any, Tuple, Optional, Union
import json

from ..agents.utils import zwarn, rprint
from ..agents.process_reward_model import ProcessRewardModel
from ..agents.planning_reward_model import PlanningRewardModel
from ..agents.ranking_reward_model import RankingRewardModel
from ..agents.diverse_sequential_sampling import DiverseSequentialSampler
from .agent import WebAgent


class ProcessRewardWebAgent(WebAgent):
    """
    Enhanced Web Agent with Process Reward Model integration.
    Supports greedy sampling by generating multiple action candidates per step.
    """
    
    def __init__(self, **kwargs):
        # Extract process reward configuration
        process_reward_config = kwargs.pop('process_reward', {})
        
        # Process reward specific parameters
        self.enable_process_reward = process_reward_config.get('enable_process_reward', False)
        self.num_action_candidates = process_reward_config.get('num_action_candidates', 3)
        self.enable_parallel_sampling = process_reward_config.get('enable_parallel_sampling', True)
        self.rpm_model_config = process_reward_config.get('rpm_model_config', {})
        self.sampling_temperature = process_reward_config.get('sampling_temperature', 0.8)
        self.min_score_threshold = process_reward_config.get('min_score_threshold', 0.3)
        self.prm_usage_strategy = process_reward_config.get('prm_usage_strategy', 'consistent')
        
        # PRM Mode: score-based vs ranking
        self.prm_mode = process_reward_config.get('prm_mode', 'score')  # 'score' or 'ranking'
        
        # Enable planning PRM (works with both score and ranking modes)
        self.enable_planning_prm = (process_reward_config.get('enable_planning_prm', False) or 
                                    self.rpm_model_config.get('enable_planning_prm', False))
        
        # Extract diverse sequential sampling configuration
        diverse_sampling_config = kwargs.pop('diverse_sequential_sampling', {})
        self.diverse_sequential_sampling = diverse_sampling_config
        
        # Create diverse sampling features
        self.enable_diverse_sampling = diverse_sampling_config.get('enable_diverse_sampling', False)
        self.sequential_mode = diverse_sampling_config.get('sequential_mode', False)
        self.diversity_threshold = diverse_sampling_config.get('diversity_threshold', 0.4)
        self.max_sampling_attempts = diverse_sampling_config.get('max_sampling_attempts', 3)
        self.diversity_prompt_strength = diverse_sampling_config.get('diversity_prompt_strength', 'medium')
        
        # Adaptive Sampling configuration
        self.enable_adaptive_sampling = diverse_sampling_config.get('enable_adaptive_sampling', False)
        self.adaptive_score_threshold = diverse_sampling_config.get('adaptive_score_threshold', 0.75)
        self.adaptive_min_candidates = diverse_sampling_config.get('adaptive_min_candidates', 1)
        
        # Session-level PRM state tracking
        self._session_prm_enabled = None
        
        # Initialize base agent with remaining kwargs
        # Note: kwargs now contains model, model_multimodal, web_env_kwargs, etc.
        # but NOT process_reward or diverse_sequential_sampling (they were popped)
        super().__init__(**kwargs)
        
        # Debug: Verify model configuration
        if hasattr(self, 'model') and hasattr(self.model, 'call_target'):
            rprint(f"[ProcessRewardWebAgent] model.call_target = {self.model.call_target}")
        if hasattr(self, 'model_multimodal') and hasattr(self.model_multimodal, 'call_target'):
            rprint(f"[ProcessRewardWebAgent] model_multimodal.call_target = {self.model_multimodal.call_target}")
        
        # Safety override: if model is still in manual mode, force it to the intended endpoint
        try:
            desired_model_target = None
            desired_mm_target = None
            # Prefer explicit kwargs passed for web_agent model targets
            if isinstance(kwargs.get('model'), dict):
                desired_model_target = kwargs['model'].get('call_target') or desired_model_target
            if isinstance(kwargs.get('model_multimodal'), dict):
                desired_mm_target = kwargs['model_multimodal'].get('call_target') or desired_mm_target
            # Fallback to RPM model endpoint for subagent if not provided
            if not desired_model_target:
                desired_model_target = self.rpm_model_config.get('call_target')
            if not desired_mm_target:
                desired_mm_target = desired_model_target
            
            # Apply overrides when current targets are manual
            if getattr(self, 'model', None) and getattr(self.model, 'call_target', 'manual') == 'manual' and desired_model_target:
                self.model.call_target = desired_model_target
                # refresh target type
                self.model.call_target_type = self.model.get_call_target_type()
                rprint(f"[ProcessRewardWebAgent] Corrected model.call_target -> {self.model.call_target}")
            if getattr(self, 'model_multimodal', None) and getattr(self.model_multimodal, 'call_target', 'manual') == 'manual' and desired_mm_target:
                self.model_multimodal.call_target = desired_mm_target
                self.model_multimodal.call_target_type = self.model_multimodal.get_call_target_type()
                rprint(f"[ProcessRewardWebAgent] Corrected model_multimodal.call_target -> {self.model_multimodal.call_target}")
        except Exception as e:
            zwarn(f"[ProcessRewardWebAgent] Failed to override model targets: {e}")
        
        # Initialize Process Reward Model
        if self.enable_process_reward:
            # Initialize based on PRM mode
            if self.prm_mode == 'ranking':
                # Use RankingRewardModel - evaluates all candidates together
                ranking_config = {
                    'call_target': self.rpm_model_config.get('call_target', 'claude:claude-3.7'),
                    'evaluation_timeout': self.rpm_model_config.get('evaluation_timeout', 30),
                    'max_retries': self.rpm_model_config.get('max_retries', 2),
                }
                self.process_reward_model = RankingRewardModel(**ranking_config)
                self.ranking_reward_model = self.process_reward_model  # Alias for clarity
                
                planning_note = " with Planning PRM" if self.enable_planning_prm else ""
                rprint(f"[ProcessRewardWebAgent] Initialized RankingRewardModel (ranking mode{planning_note})")
            else:
                # Use score-based PRM (existing behavior)
                # Base RPM config - only include parameters that ProcessRewardModel accepts
                rpm_config = {
                    'call_target': self.rpm_model_config.get('call_target', 'claude:claude-3.7'),
                    'enable_parallel_evaluation': self.rpm_model_config.get('enable_parallel_evaluation', True),
                    'evaluation_timeout': self.rpm_model_config.get('evaluation_timeout', 30),
                    'max_retries': self.rpm_model_config.get('max_retries', 2),
                }
                
                # Check if planning PRM is enabled
                if self.enable_planning_prm:
                    # For PlanningRewardModel, add planning-specific parameters
                    planning_score_weight = self.rpm_model_config.get('planning_score_weight', 0.7)
                    diversity_weight = self.rpm_model_config.get('diversity_weight', 0.3)
                    
                    rpm_config.update({
                        'enable_planning_prm': self.enable_planning_prm,
                        'planning_score_weight': planning_score_weight,
                        'diversity_weight': diversity_weight
                    })
                    
                    self.process_reward_model = PlanningRewardModel(**rpm_config)
                    rprint(f"[ProcessRewardWebAgent] Initialized PlanningRewardModel (score mode)")
                else:
                    # For ProcessRewardModel, don't include planning-specific parameters
                    # They will cause errors as ProcessRewardModel doesn't recognize them
                    self.process_reward_model = ProcessRewardModel(**rpm_config)
                    rprint(f"[ProcessRewardWebAgent] Initialized ProcessRewardModel (score mode)")
        else:
            self.process_reward_model = None
        
        # Initialize Diverse Sequential Sampler if enabled
        if self.enable_diverse_sampling:
            diverse_sampler_config = {
                'enable_diverse_sampling': self.enable_diverse_sampling,
                'sequential_mode': self.sequential_mode,
                'diversity_threshold': self.diversity_threshold,
                'max_sampling_attempts': self.max_sampling_attempts,
                'diversity_prompt_strength': self.diversity_prompt_strength,
            }
            self.diverse_sampler = DiverseSequentialSampler(**diverse_sampler_config)
            rprint(f"[ProcessRewardWebAgent] Initialized DiverseSequentialSampler")
        else:
            self.diverse_sampler = None
    
    def step(self, session, state):
        """Override step to add PRM support for planning and action phases"""
        if not self.enable_process_reward:
            # If PRM disabled, use original step
            return super().step(session, state)
        
        # Session-level PRM decision (similar to ProcessRewardCKAgent)
        if self.prm_usage_strategy == 'consistent':
            if self._session_prm_enabled is None:
                self._session_prm_enabled = random.random() < 0.5
            use_prm_this_step = self._session_prm_enabled
        elif self.prm_usage_strategy == 'always':
            use_prm_this_step = True
        else:  # selective
            use_prm_this_step = random.random() < 0.5
        
        if not use_prm_this_step:
            return super().step(session, state)
        
        # Get input kwargs
        _input_kwargs, _extra_kwargs = self.step_prepare(session, state)
        _current_step = session.get_current_step()
        
        # Planning phase with PRM
        has_plan_template = "plan" in self.templates
        if has_plan_template and isinstance(self.process_reward_model, PlanningRewardModel) and \
           self.process_reward_model.enable_planning_prm:
            rprint(f"[ProcessRewardWebAgent] Planning phase with PRM")
            
            # Use adaptive sampling if enabled
            if self.enable_adaptive_sampling:
                plan_candidates, plan_scores, plan_evaluations = self._generate_and_evaluate_adaptively(
                    session, state, _input_kwargs, is_planning=True)
            else:
                plan_candidates = self._generate_plan_candidates(_input_kwargs, session)
                plan_scores, plan_evaluations = self._evaluate_plan_candidates(session, state, plan_candidates)
            
            if len(plan_candidates) > 1:
                best_plan, best_score, best_idx = self._select_best_candidate(plan_candidates, plan_scores)
                plan_res = best_plan
                
                # Store PRM info
                safe_candidates = []
                for cand in plan_candidates:
                    if isinstance(cand, dict):
                        safe_cand = {
                            'thought': cand.get('thought', ''),
                            'code': cand.get('code', ''),
                        }
                        if 'diversity_info' in cand:
                            safe_cand['diversity_info'] = cand['diversity_info']
                        safe_candidates.append(safe_cand)
                    else:
                        safe_candidates.append({'thought': str(cand), 'code': ''})
                
                plan_res['prm_info'] = {
                    'all_candidates': safe_candidates,
                    'scores': plan_scores,
                    'selected_idx': best_idx,
                    'rpm_evaluations': plan_evaluations,
                    'num_candidates': len(plan_candidates),
                    'evaluation_method': 'planning_reward_model',
                    'phase': 'planning',
                }
                
                rprint(f"[ProcessRewardWebAgent] Selected planning candidate {best_idx} with score {best_score:.3f}")
            else:
                plan_messages = self.templates["plan"].format(**_input_kwargs)
                plan_response = self.step_call(messages=plan_messages, session=session)
                plan_res = self._parse_output(plan_response)
        else:
            # Normal planning
            plan_messages = self.templates["plan"].format(**_input_kwargs)
            plan_response = self.step_call(messages=plan_messages, session=session)
            plan_res = self._parse_output(plan_response)
        
        # Execute plan code
        if isinstance(plan_res, dict) and "code" in plan_res and plan_res["code"]:
            try:
                exec(plan_res["code"], {"state": state})
            except Exception as e:
                plan_res["obs"] = f"Error in planning: {e}"
                zwarn(f"[ProcessRewardWebAgent] Error executing plan code: {e}")
        
        if isinstance(plan_res, dict):
            plan_res["state"] = state.copy()
        
        _current_step["plan"] = plan_res
        
        # Action phase with PRM
        return self._execute_action_with_prm(session, state, _input_kwargs, _extra_kwargs)
    
    def _execute_action_with_prm(self, session, state, _input_kwargs, _extra_kwargs):
        """Execute action with PRM evaluation"""
        try:
            # Use adaptive sampling if enabled
            if self.enable_adaptive_sampling:
                action_candidates, scores, rpm_evaluations = self._generate_and_evaluate_adaptively(
                    session, state, _input_kwargs, is_planning=False)
            else:
                action_candidates = self._generate_action_candidates(_input_kwargs, session)
                
                if len(action_candidates) <= 1:
                    # Fall back to normal execution
                    action_messages = self.templates["action"].format(**_input_kwargs)
                    action_response = self.step_call(messages=action_messages, session=session)
                    action_res = self._parse_output(action_response)
                    step_res = self.step_action(action_res, _input_kwargs, **_extra_kwargs)
                    action_res["observation"] = step_res
                    session.get_current_step()["action"] = action_res
                    return session.get_current_step()
                
                scores, rpm_evaluations = self._evaluate_action_candidates(session, state, action_candidates)
            
            # Select best candidate
            best_candidate, best_score, best_idx = self._select_best_candidate(action_candidates, scores)
            
            # Create action_res with best candidate
            action_res = best_candidate.copy()
            
            # Store PRM info
            safe_candidates = []
            for cand in action_candidates:
                if isinstance(cand, dict):
                    safe_cand = {
                        'thought': cand.get('thought', ''),
                        'code': cand.get('code', ''),
                    }
                    if 'diversity_info' in cand:
                        safe_cand['diversity_info'] = cand['diversity_info']
                    safe_candidates.append(safe_cand)
                else:
                    safe_candidates.append({'thought': str(cand), 'code': ''})
            
            action_res['prm_info'] = {
                'all_candidates': safe_candidates,
                'scores': scores,
                'selected_idx': best_idx,
                'rpm_evaluations': rpm_evaluations,
                'num_candidates': len(action_candidates),
                'evaluation_method': 'process_reward_model',
            }
            
            # Execute the selected action
            step_res = self.step_action(action_res, _input_kwargs, **_extra_kwargs)
            action_res["observation"] = step_res
            
            _current_step = session.get_current_step()
            _current_step["action"] = action_res
            
            rprint(f"[ProcessRewardWebAgent] Selected action candidate {best_idx} with score {best_score:.3f}")
            
            return _current_step
            
        except Exception as e:
            zwarn(f"[ProcessRewardWebAgent] Error in action execution with PRM: {e}")
            # Fall back to normal execution
            return super().step(session, state)
    
    # Copy PRM-related methods from ProcessRewardCKAgent
    def _generate_plan_candidates(self, input_kwargs, session) -> List[Dict]:
        """Generate multiple planning candidates"""
        if self.enable_diverse_sampling and self.diverse_sampler and self.sequential_mode:
            return self._generate_diverse_plan_candidates(input_kwargs, session)
        
        candidates = []
        for i in range(self.num_action_candidates):
            try:
                original_seed = self.get_seed()
                candidate_seed = (original_seed + i + 100) if original_seed else (i + 100)
                
                try:
                    self.set_seed(candidate_seed)
                    plan_messages = self.templates["plan"].format(**input_kwargs)
                    plan_response = self.step_call(messages=plan_messages, session=session)
                    plan_res = self._parse_output(plan_response)
                    candidates.append(plan_res)
                finally:
                    self.set_seed(original_seed)
            except Exception as e:
                zwarn(f"[ProcessRewardWebAgent] Failed to generate planning candidate {i}: {e}")
        
        return [c for c in candidates if c is not None]
    
    def _generate_diverse_plan_candidates(self, input_kwargs, session) -> List[Dict]:
        """Generate planning candidates using diverse sequential sampling"""
        try:
            self.diverse_sampler.reset_candidates()
            plan_messages = self.templates["plan"].format(**input_kwargs)
            
            raw_responses = self.diverse_sampler.generate_candidates(
                model=self.model,
                prompt=plan_messages,
                num_candidates=self.num_action_candidates,
                temperature=self.sampling_temperature,
                max_tokens=2000
            )
            
            candidates = []
            for i, response in enumerate(raw_responses):
                try:
                    plan_res = self._parse_output(response)
                    plan_res['diversity_info'] = {
                        'candidate_idx': i,
                        'diverse_sampling': True,
                        'diversity_threshold': self.diversity_threshold,
                    }
                    candidates.append(plan_res)
                except Exception as e:
                    zwarn(f"[ProcessRewardWebAgent] Failed to parse diverse planning candidate {i}: {e}")
            
            return candidates
        except Exception as e:
            zwarn(f"[ProcessRewardWebAgent] Diverse planning generation failed: {e}")
            return self._generate_plan_candidates(input_kwargs, session)
    
    def _generate_action_candidates(self, action_input_kwargs, session) -> List[Dict]:
        """Generate multiple action candidates"""
        if self.enable_diverse_sampling and self.diverse_sampler and self.sequential_mode:
            return self._generate_diverse_action_candidates(action_input_kwargs, session)
        
        candidates = []
        for i in range(self.num_action_candidates):
            try:
                original_seed = self.get_seed()
                candidate_seed = (original_seed + i) if original_seed else i
                
                try:
                    self.set_seed(candidate_seed)
                    action_messages = self.templates["action"].format(**action_input_kwargs)
                    action_response = self.step_call(messages=action_messages, session=session)
                    action_res = self._parse_output(action_response)
                    candidates.append(action_res)
                finally:
                    self.set_seed(original_seed)
            except Exception as e:
                zwarn(f"[ProcessRewardWebAgent] Failed to generate action candidate {i}: {e}")
        
        return [c for c in candidates if c is not None]
    
    def _generate_diverse_action_candidates(self, action_input_kwargs, session) -> List[Dict]:
        """Generate action candidates using diverse sequential sampling"""
        try:
            self.diverse_sampler.reset_candidates()
            action_messages = self.templates["action"].format(**action_input_kwargs)
            
            raw_responses = self.diverse_sampler.generate_candidates(
                model=self.model,
                prompt=action_messages,
                num_candidates=self.num_action_candidates,
                temperature=self.sampling_temperature,
                max_tokens=2000
            )
            
            candidates = []
            for i, response in enumerate(raw_responses):
                try:
                    action_res = self._parse_output(response)
                    action_res['diversity_info'] = {
                        'candidate_idx': i,
                        'diverse_sampling': True,
                        'diversity_threshold': self.diversity_threshold,
                    }
                    candidates.append(action_res)
                except Exception as e:
                    zwarn(f"[ProcessRewardWebAgent] Failed to parse diverse action candidate {i}: {e}")
            
            return candidates
        except Exception as e:
            zwarn(f"[ProcessRewardWebAgent] Diverse action generation failed: {e}")
            return self._generate_action_candidates(action_input_kwargs, session)
    
    def _generate_and_evaluate_adaptively(self, session, current_state, 
                                          input_kwargs, is_planning=False) -> Tuple[List[Dict], List[float], List[Dict]]:
        """Generate and evaluate candidates adaptively - stop early if we find a good one"""
        rprint(f"[ProcessRewardWebAgent] Using adaptive sampling (threshold={self.adaptive_score_threshold})")
        
        all_candidates = []
        all_scores = []
        all_evaluations = []
        best_score = -1.0
        
        if self.enable_diverse_sampling and hasattr(self, 'diverse_sampler'):
            self.diverse_sampler.reset_candidates()
        
        for i in range(self.num_action_candidates):
            # Generate one candidate
            if is_planning:
                if self.enable_diverse_sampling and i > 0:
                    plan_messages = self.templates["plan"].format(**input_kwargs)
                    raw_responses = self.diverse_sampler.generate_candidates(
                        model=self.model,
                        prompt=plan_messages,
                        num_candidates=1,
                        temperature=self.sampling_temperature,
                        max_tokens=2000
                    )
                    if raw_responses:
                        candidate = self._parse_output(raw_responses[0])
                        candidate['diversity_info'] = {
                            'candidate_idx': i,
                            'diverse_sampling': True,
                            'diversity_threshold': self.diversity_threshold,
                        }
                    else:
                        continue
                else:
                    # Generate first candidate normally
                    original_seed = self.get_seed()
                    candidate_seed = (original_seed + i + 100) if original_seed else (i + 100)
                    try:
                        self.set_seed(candidate_seed)
                        plan_messages = self.templates["plan"].format(**input_kwargs)
                        plan_response = self.step_call(messages=plan_messages, session=session)
                        candidate = self._parse_output(plan_response)
                    finally:
                        self.set_seed(original_seed)
            else:
                # Generate action candidate
                if self.enable_diverse_sampling and i > 0:
                    action_messages = self.templates["action"].format(**input_kwargs)
                    raw_responses = self.diverse_sampler.generate_candidates(
                        model=self.model,
                        prompt=action_messages,
                        num_candidates=1,
                        temperature=self.sampling_temperature,
                        max_tokens=2000
                    )
                    if raw_responses:
                        candidate = self._parse_output(raw_responses[0])
                        candidate['diversity_info'] = {
                            'candidate_idx': i,
                            'diverse_sampling': True,
                            'diversity_threshold': self.diversity_threshold,
                        }
                    else:
                        continue
                else:
                    original_seed = self.get_seed()
                    candidate_seed = (original_seed + i) if original_seed else i
                    try:
                        self.set_seed(candidate_seed)
                        action_messages = self.templates["action"].format(**input_kwargs)
                        action_response = self.step_call(messages=action_messages, session=session)
                        candidate = self._parse_output(action_response)
                    finally:
                        self.set_seed(original_seed)
            
            if candidate is None:
                continue
            
            all_candidates.append(candidate)
            
            # Evaluate this candidate immediately
            if is_planning:
                scores, evals = self._evaluate_plan_candidates(session, current_state, [candidate])
            else:
                scores, evals = self._evaluate_action_candidates(session, current_state, [candidate])
            
            if scores:
                score = scores[0]
                all_scores.append(score)
                all_evaluations.append(evals[0] if evals else {})
                
                if score > best_score:
                    best_score = score
                
                # Check if we should stop early
                if (i + 1 >= self.adaptive_min_candidates and 
                    score >= self.adaptive_score_threshold):
                    rprint(f"[ProcessRewardWebAgent] Adaptive sampling: Found good candidate "
                          f"(score={score:.3f}), stopping after {i+1} candidates")
                    break
            else:
                all_scores.append(0.5)
                all_evaluations.append({})
        
        return all_candidates, all_scores, all_evaluations
    
    def _evaluate_plan_candidates(self, session, current_state, candidates) -> Tuple[List[float], List[Dict]]:
        """Evaluate planning candidates using PRM (score-based or ranking)"""
        try:
            session_history = session.steps[:-1] if session.steps else []
            
            # Use Ranking PRM if in ranking mode
            if self.prm_mode == 'ranking':
                return self._rank_plan_candidates(session.task, session_history, current_state, candidates)
            
            # Otherwise, use score-based evaluation
            # Convert candidates format if needed
            converted_candidates = []
            for candidate in candidates:
                converted = {}
                if 'thought' in candidate:
                    converted['reasoning'] = candidate['thought']
                if 'code' in candidate:
                    converted['plan'] = candidate['code']
                for key in ['diversity_info', 'diversity_enabled']:
                    if key in candidate:
                        converted[key] = candidate[key]
                converted_candidates.append(converted)
            
            if hasattr(self.process_reward_model, 'evaluate_plan_candidates_detailed'):
                scores, evaluations = self.process_reward_model.evaluate_plan_candidates_detailed(
                    task=session.task,
                    session_history=session_history,
                    current_state=current_state,
                    plan_candidates=converted_candidates
                )
            else:
                scores, evaluations = self.process_reward_model.evaluate_action_candidates_detailed(
                    task=session.task,
                    session_history=session_history,
                    current_state=current_state,
                    action_candidates=converted_candidates
                )
            
            return scores, evaluations
        except Exception as e:
            zwarn(f"[ProcessRewardWebAgent] Planning evaluation failed: {e}")
            return [0.5] * len(candidates), [{"error": str(e)} for _ in candidates]
    
    def _evaluate_action_candidates(self, session, current_state, candidates) -> Tuple[List[float], List[Dict]]:
        """Evaluate action candidates using PRM (score-based or ranking)"""
        try:
            session_history = session.steps[:-1] if session.steps else []
            
            # Use Ranking PRM if in ranking mode
            if self.prm_mode == 'ranking':
                return self._rank_action_candidates(session.task, session_history, current_state, candidates)
            
            # Otherwise, use score-based evaluation
            scores, evaluations = self.process_reward_model.evaluate_action_candidates_detailed(
                task=session.task,
                session_history=session_history,
                current_state=current_state,
                action_candidates=candidates
            )
            
            return scores, evaluations
        except Exception as e:
            zwarn(f"[ProcessRewardWebAgent] Action evaluation failed: {e}")
            return [0.5] * len(candidates), [{"error": str(e)} for _ in candidates]
    
    def _rank_plan_candidates(self, task: str, session_history: List[Dict], 
                              current_state: Dict, candidates: List[Dict]) -> Tuple[List[float], List[Dict]]:
        """Use Ranking PRM to rank planning candidates"""
        try:
            rprint(f"[ProcessRewardWebAgent] Using Ranking PRM for {len(candidates)} planning candidates")
            
            # Call RankingRewardModel to get ranking
            ranked_indices, evaluation_info = self.ranking_reward_model.rank_candidates(
                task=task,
                session_history=session_history,
                current_state=current_state,
                candidates=candidates,
                is_planning=True
            )
            
            # Convert ranking to scores
            scores = self._ranking_to_scores(ranked_indices)
            
            # Create evaluation info for each candidate
            evaluations = []
            for i, candidate_idx in enumerate(ranked_indices):
                eval_dict = {
                    'rank': i + 1,
                    'score': scores[candidate_idx],
                    'ranking_method': 'ranking_prm',
                    'evaluation_info': evaluation_info
                }
                evaluations.append(eval_dict)
            
            rprint(f"[ProcessRewardWebAgent] Ranking PRM planning scores: {scores}")
            return scores, evaluations
            
        except Exception as e:
            zwarn(f"[ProcessRewardWebAgent] Ranking PRM planning evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return [0.5] * len(candidates), [{"error": str(e)} for _ in candidates]
    
    def _rank_action_candidates(self, task: str, session_history: List[Dict], 
                                current_state: Dict, candidates: List[Dict]) -> Tuple[List[float], List[Dict]]:
        """Use Ranking PRM to rank action candidates"""
        try:
            rprint(f"[ProcessRewardWebAgent] Using Ranking PRM for {len(candidates)} action candidates")
            
            # Call RankingRewardModel to get ranking
            ranked_indices, evaluation_info = self.ranking_reward_model.rank_candidates(
                task=task,
                session_history=session_history,
                current_state=current_state,
                candidates=candidates,
                is_planning=False
            )
            
            # Convert ranking to scores
            scores = self._ranking_to_scores(ranked_indices)
            
            # Create evaluation info for each candidate
            evaluations = []
            for i, candidate_idx in enumerate(ranked_indices):
                eval_dict = {
                    'rank': i + 1,
                    'score': scores[candidate_idx],
                    'ranking_method': 'ranking_prm',
                    'evaluation_info': evaluation_info
                }
                evaluations.append(eval_dict)
            
            rprint(f"[ProcessRewardWebAgent] Ranking PRM action scores: {scores}")
            return scores, evaluations
            
        except Exception as e:
            zwarn(f"[ProcessRewardWebAgent] Ranking PRM action evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return [0.5] * len(candidates), [{"error": str(e)} for _ in candidates]
    
    def _ranking_to_scores(self, ranked_indices: List[int]) -> List[float]:
        """Convert ranking indices to scores"""
        num_candidates = len(ranked_indices)
        scores = [0.0] * num_candidates
        
        for rank, candidate_idx in enumerate(ranked_indices):
            # Best candidate (rank 0) gets score 1.0
            # Worst candidate gets score 0.1
            # Linear interpolation in between
            if num_candidates == 1:
                score = 1.0
            else:
                score = 1.0 - (rank * 0.9 / (num_candidates - 1))
            scores[candidate_idx] = score
        
        return scores
    
    def _select_best_candidate(self, candidates: List[Dict], scores: List[float]) -> Tuple[Dict, float, int]:
        """Select the best candidate based on scores"""
        if not candidates or not scores:
            return {}, 0.0, -1
        
        best_idx = scores.index(max(scores))
        best_candidate = candidates[best_idx]
        best_score = scores[best_idx]
        
        return best_candidate, best_score, best_idx

    # Ensure PRM info is surfaced to the top-level log via captured stdout
    def __call__(self, task: str, **kwargs):  # allow *args styled calling
        result = super().__call__(task, **kwargs)
        try:
            if self.enable_process_reward and hasattr(result, 'session') and result.session:
                sess = result.session
                last_step = sess.get_current_step() if hasattr(sess, 'get_current_step') else None
                prm_payload = {}
                if isinstance(last_step, dict):
                    plan_dict = last_step.get('plan', {})
                    action_dict = last_step.get('action', {})
                    if isinstance(plan_dict, dict) and 'prm_info' in plan_dict:
                        prm_payload['plan_prm_info'] = plan_dict['prm_info']
                    if isinstance(action_dict, dict) and 'prm_info' in action_dict:
                        prm_payload['action_prm_info'] = action_dict['prm_info']
                if prm_payload:
                    print(json.dumps({'__subagent_prm__': prm_payload}, ensure_ascii=False))
        except Exception as e:
            zwarn(f"[ProcessRewardWebAgent] Failed to emit PRM info to stdout: {e}")
        return result

