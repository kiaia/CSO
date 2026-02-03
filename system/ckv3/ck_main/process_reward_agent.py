#
# Process Reward enabled CK Agent with greedy sampling

import time
import random
import multiprocessing as mp
import copy
from functools import partial
from typing import List, Dict, Any, Tuple, Optional, Union
import json

from ..agents.agent import MultiStepAgent, register_template, AgentResult
from ..agents.utils import zwarn, rprint, GET_ENV_VAR
from ..agents.process_reward_model import ProcessRewardModel
from ..agents.planning_reward_model import PlanningRewardModel
from ..agents.ranking_reward_model import RankingRewardModel
from ..agents.diverse_sequential_sampling import DiverseSequentialSampler
from .agent import CKAgent
from ..ck_web.process_reward_web_agent import ProcessRewardWebAgent


class ProcessRewardCKAgent(CKAgent):
    """
    Enhanced CK Agent with Process Reward Model integration.
    Supports greedy sampling by generating multiple action candidates per step.
    """
    
    @staticmethod
    def _parse_bool(value, default=None):
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            norm = value.strip().lower()
            if norm in {"1", "true", "t", "yes", "y", "on"}:
                return True
            if norm in {"0", "false", "f", "no", "n", "off"}:
                return False
        return default

    def __init__(self, **kwargs):
        # Extract process reward configuration
        process_reward_config = kwargs.pop('process_reward', {})
        
        # Process reward specific parameters
        self.enable_process_reward = process_reward_config.get('enable_process_reward', False)
        self.num_action_candidates = process_reward_config.get('num_action_candidates', 3)
        self.enable_parallel_sampling = process_reward_config.get('enable_parallel_sampling', True)
        self.rpm_model_config = process_reward_config.get('rpm_model_config', {})
        # Note: sampling_temperature is no longer used - we keep original model config
        # and only vary seeds for diversity (modern LLMs provide sufficient diversity even at temp=0)
        self.sampling_temperature = process_reward_config.get('sampling_temperature', 0.8)  # Kept for backward compatibility
        self.min_score_threshold = process_reward_config.get('min_score_threshold', 0.3)
        
        # Session-level consistency strategy
        self.prm_usage_strategy = process_reward_config.get('prm_usage_strategy', 'consistent')
        # consistent: session-level decision, all steps use same strategy
        # selective: current step-level decision (original behavior)
        # always: force PRM for all steps
        
        # PRM Mode: score-based vs ranking
        self.prm_mode = process_reward_config.get('prm_mode', 'score')  # 'score' or 'ranking'
        # score: Each candidate evaluated independently and scored (existing behavior)
        # ranking: All candidates evaluated together and ranked (new feature)
        
        # Enable planning PRM (works with both score and ranking modes)
        # Check both top-level and rpm_model_config for backward compatibility
        self.enable_planning_prm = (process_reward_config.get('enable_planning_prm', False) or 
                                    self.rpm_model_config.get('enable_planning_prm', False))

        # Default sub-agent PRM flag (resolved later with env/config fallbacks)
        self.enable_subagent_prm = False
        
        # Extract diverse sequential sampling configuration
        diverse_sampling_config = kwargs.pop('diverse_sequential_sampling', {})
        
        # Store the config as an attribute to fix the AssertionError
        self.diverse_sequential_sampling = diverse_sampling_config
        
        # Create diverse sampling features
        self.enable_diverse_sampling = diverse_sampling_config.get('enable_diverse_sampling', False)
        self.sequential_mode = diverse_sampling_config.get('sequential_mode', False)
        self.diversity_threshold = diverse_sampling_config.get('diversity_threshold', 0.4)
        self.max_sampling_attempts = diverse_sampling_config.get('max_sampling_attempts', 3)
        self.diversity_prompt_strength = diverse_sampling_config.get('diversity_prompt_strength', 'medium')
        
        # Adaptive Sampling configuration - stop early if we get a good candidate
        self.enable_adaptive_sampling = diverse_sampling_config.get('enable_adaptive_sampling', False)
        self.adaptive_score_threshold = diverse_sampling_config.get('adaptive_score_threshold', 0.75)  # Stop if score >= this
        self.adaptive_min_candidates = diverse_sampling_config.get('adaptive_min_candidates', 1)  # Always generate at least this many
        
        # Mutual exclusion check: Ranking PRM and Adaptive Sampling cannot coexist
        if self.prm_mode == 'ranking' and self.enable_adaptive_sampling:
            zwarn("[ProcessRewardCKAgent] WARNING: Ranking PRM and Adaptive Sampling are mutually exclusive!")
            zwarn("[ProcessRewardCKAgent] Disabling Adaptive Sampling to use Ranking PRM")
            self.enable_adaptive_sampling = False
        
        # Session-level PRM state tracking
        self._session_prm_enabled = None  # Will be set per session
        
        # Setup web agent with PRM support if enabled
        # Note: This must be done before super().__init__() because CKAgent.__init__() 
        # creates self.web_agent, and we want to override that
        web_agent_kwargs = kwargs.get('web_agent')
        self._web_agent_prm_config = None
        if self.enable_process_reward:
            # Resolve enable_subagent_prm with multiple fallbacks
            raw_subagent_flag = process_reward_config.get('enable_subagent_prm', None)
            resolved_subagent_flag = self._parse_bool(raw_subagent_flag, default=None)

            if resolved_subagent_flag is None:
                env_subagent_flag = GET_ENV_VAR("ENABLE_SUBAGENT_PRM")
                resolved_subagent_flag = self._parse_bool(env_subagent_flag, default=None)

            if resolved_subagent_flag is None and isinstance(web_agent_kwargs, dict):
                # If the caller explicitly configured PRM settings on the web agent, assume they want it enabled
                if ('process_reward' in web_agent_kwargs) or ('diverse_sequential_sampling' in web_agent_kwargs):
                    resolved_subagent_flag = True

            self.enable_subagent_prm = self._parse_bool(resolved_subagent_flag, default=False)

            if self.enable_subagent_prm:
                # Extract web_agent config from kwargs if provided
                if 'web_agent' in kwargs:
                    web_agent_kwargs = kwargs.pop('web_agent')  # Remove from kwargs to prevent standard WebAgent init

                # Create ProcessRewardWebAgent with the same configurations
                web_agent_config = {
                    'process_reward': process_reward_config,
                    'diverse_sequential_sampling': diverse_sampling_config,
                }
                # Copy web_agent specific config if provided
                if web_agent_kwargs is not None:
                    web_agent_config.update(web_agent_kwargs)

                # Ensure the PRM web agent inherits the primary model targets when not explicitly set
                top_level_model_cfg = kwargs.get('model')
                if 'model' not in web_agent_config:
                    if isinstance(top_level_model_cfg, dict):
                        web_agent_config['model'] = top_level_model_cfg.copy()
                    elif hasattr(top_level_model_cfg, 'call_target'):
                        web_agent_config['model'] = {'call_target': top_level_model_cfg.call_target}

                top_level_mm_model_cfg = kwargs.get('model_multimodal')
                if 'model_multimodal' not in web_agent_config:
                    if isinstance(top_level_mm_model_cfg, dict):
                        web_agent_config['model_multimodal'] = top_level_mm_model_cfg.copy()
                    elif hasattr(top_level_mm_model_cfg, 'call_target'):
                        web_agent_config['model_multimodal'] = {'call_target': top_level_mm_model_cfg.call_target}

                # Preserve config for post-init assignment to avoid being overwritten by CKAgent.__init__
                self._web_agent_prm_config = copy.deepcopy(web_agent_config)
            else:
                rprint("[ProcessRewardCKAgent] Sub-agent PRM disabled; using standard WebAgent")
        
        # Initialize base agent
        super().__init__(**kwargs)

        # Replace the default web agent with PRM-enabled version after base initialization
        if self.enable_subagent_prm and self._web_agent_prm_config is not None:
            self.web_agent = ProcessRewardWebAgent(**self._web_agent_prm_config)
            # Refresh ACTIVE_FUNCTIONS to point to the new sub-agent instance
            if hasattr(self, 'ACTIVE_FUNCTIONS') and isinstance(self.ACTIVE_FUNCTIONS, dict):
                self.ACTIVE_FUNCTIONS['web_agent'] = self.web_agent
            rprint("[ProcessRewardCKAgent] Initialized ProcessRewardWebAgent for sub-agent")
        
        # Initialize Process Reward Model
        if self.enable_process_reward:
            # Extract only supported parameters for basic RPM models
            rpm_config = {
                'call_target': self.rpm_model_config.get('call_target', 'claude:claude-3.7'),
                'enable_parallel_evaluation': self.rpm_model_config.get('enable_parallel_evaluation', True),
                'evaluation_timeout': self.rpm_model_config.get('evaluation_timeout', 30),
                'max_retries': self.rpm_model_config.get('max_retries', 2),
            }
            
            # Initialize based on PRM mode
            if self.prm_mode == 'ranking':
                # Use RankingRewardModel - evaluates all candidates together
                ranking_config = {
                    'call_target': rpm_config['call_target'],
                    'evaluation_timeout': rpm_config['evaluation_timeout'],
                    'max_retries': rpm_config.get('max_retries', 2),
                }
                self.process_reward_model = RankingRewardModel(**ranking_config)
                self.ranking_reward_model = self.process_reward_model  # Alias for clarity
                
                planning_note = " with Planning PRM" if self.enable_planning_prm else ""
                rprint(f"[ProcessRewardCKAgent] Initialized RankingRewardModel (ranking mode{planning_note})")
            else:
                # Use score-based PRM (existing behavior)
                # Check if planning PRM is enabled (respect merged flag from both locations)
                enable_planning_prm = self.enable_planning_prm
                if enable_planning_prm:
                    # Use PlanningRewardModel which extends ProcessRewardModel
                    planning_score_weight = self.rpm_model_config.get('planning_score_weight', 0.7)
                    diversity_weight = self.rpm_model_config.get('diversity_weight', 0.3)
                    
                    # Create config for PlanningRewardModel with specific parameters
                    planning_config = rpm_config.copy()
                    planning_config.update({
                        'enable_planning_prm': enable_planning_prm,
                        'planning_score_weight': planning_score_weight,
                        'diversity_weight': diversity_weight
                    })
                    
                    self.process_reward_model = PlanningRewardModel(**planning_config)
                    rprint(f"[ProcessRewardCKAgent] Initialized PlanningRewardModel with weights: "
                          f"planning={planning_score_weight}, diversity={diversity_weight}")
                else:
                    # Use standard ProcessRewardModel
                    self.process_reward_model = ProcessRewardModel(**rpm_config)
                    rprint(f"[ProcessRewardCKAgent] Initialized ProcessRewardModel (score-based mode)")
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
                'call_target': self.rpm_model_config.get('call_target', 'claude:claude-3.7'),
            }
            self.diverse_sampler = DiverseSequentialSampler(**diverse_sampler_config)
        else:
            self.diverse_sampler = None
        
        rprint(f"[ProcessRewardCKAgent] Initialized with process_reward={self.enable_process_reward}, "
               f"candidates={self.num_action_candidates}, strategy={self.prm_usage_strategy}, "
               f"diverse_sampling={self.enable_diverse_sampling}, sequential_mode={self.sequential_mode}")
    
    def step_prepare(self, session, state):
        """Override step_prepare to track current session and state"""
        # Store session and state for later use in step_action
        self._current_session = session
        self._current_state = state
        
        # Initialize session-level PRM strategy if not already set
        if self._session_prm_enabled is None and self.prm_usage_strategy == 'consistent':
            self._session_prm_enabled = self._should_enable_prm_for_session(session)
            rprint(f"[ProcessRewardCKAgent] Session PRM enabled: {self._session_prm_enabled} "
                   f"(strategy: {self.prm_usage_strategy})")
        
        return super().step_prepare(session, state)
    
    def step(self, session, state):
        """
        Override step method to add PRM support for planning phase
        """
        if not self.enable_process_reward or self.process_reward_model is None:
            # If PRM is disabled, use the original step method
            return super().step(session, state)
        
        _input_kwargs, _extra_kwargs = self.step_prepare(session, state)
        _current_step = session.get_current_step()
        
        # Planning phase
        has_plan_template = "plan" in self.templates
        if has_plan_template and self.enable_planning_prm:
            # Use PRM for planning phase (works with both score and ranking modes)
            prm_mode_str = "Ranking PRM" if self.prm_mode == 'ranking' else "Score-based PRM"
            rprint(f"[ProcessRewardCKAgent] Planning phase with {prm_mode_str}")
            
            # Use adaptive sampling if enabled
            if self.enable_adaptive_sampling:
                plan_candidates, plan_scores, plan_evaluations = self._generate_and_evaluate_adaptively(
                    session, state, _input_kwargs, is_planning=True)
            else:
                # Standard generation then evaluation
                plan_candidates = self._generate_plan_candidates(_input_kwargs, session)
                
                # Debug: Log the raw plan candidates before evaluation
                rprint(f"[ProcessRewardCKAgent] Raw planning candidates before evaluation:")
                for i, cand in enumerate(plan_candidates):
                    rprint(f"  Candidate {i}: type={type(cand)}")
                    if isinstance(cand, dict):
                        rprint(f"    Keys: {list(cand.keys())}")
                        rprint(f"    thought: {cand.get('thought', 'N/A')[:100] if cand.get('thought') else 'N/A'}")
                        rprint(f"    code: {cand.get('code', 'N/A')[:100] if cand.get('code') else 'N/A'}")
                
                # Evaluate plan candidates using PlanningRewardModel
                plan_scores, plan_evaluations = self._evaluate_plan_candidates(session, state, plan_candidates)
            
            if len(plan_candidates) > 1:
                
                # Select best plan candidate
                best_plan, best_score, best_idx = self._select_best_candidate(plan_candidates, plan_scores)
                
                # Store plan result and PRM info
                plan_res = best_plan
                
                # Create safe candidates for storage (avoid circular references)
                safe_candidates = []
                for cand in plan_candidates:
                    if isinstance(cand, dict):
                        # Only store essential fields to avoid circular references
                        safe_cand = {
                            'thought': cand.get('thought', ''),
                            'code': cand.get('code', ''),
                        }
                        # Add diversity info if present
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
                
                # Create plan_messages and plan_response for evaluator
                plan_messages = self.templates["plan"].format(**_input_kwargs)
                
                # Extract plan_response safely without circular references
                if isinstance(best_plan, dict):
                    if "raw_response" in best_plan:
                        plan_response = best_plan["raw_response"]
                    else:
                        # Create a safe representation without circular references
                        safe_plan = {}
                        for key in ["thought", "code", "reasoning"]:
                            if key in best_plan:
                                safe_plan[key] = best_plan[key]
                        plan_response = json.dumps(safe_plan, ensure_ascii=False)
                else:
                    plan_response = str(best_plan)
                
                # Store LLM input/output for evaluator
                plan_res["llm_input"] = plan_messages
                plan_res["llm_output"] = plan_response
                
                rprint(f"[ProcessRewardCKAgent] Selected planning candidate {best_idx} with score {best_score:.3f}")
            else:
                # Not enough candidates, use normal planning
                plan_messages = self.templates["plan"].format(**_input_kwargs)
                plan_response = self.step_call(messages=plan_messages, session=session)
                plan_res = self._parse_output(plan_response)
                
                # Store LLM input/output for evaluator
                plan_res["llm_input"] = plan_messages
                plan_res["llm_output"] = plan_response
        else:
            # Normal planning without PRM
            plan_messages = self.templates["plan"].format(**_input_kwargs)
            plan_response = self.step_call(messages=plan_messages, session=session)
            plan_res = self._parse_output(plan_response)
            
            # Store LLM input/output for evaluator
            plan_res["llm_input"] = plan_messages
            plan_res["llm_output"] = plan_response
            
        # Process plan result - state update like in the original agent
        if isinstance(plan_res, dict) and "code" in plan_res and plan_res["code"]:
            try:
                exec(plan_res["code"], {"state": state})
            except Exception as e:
                plan_res["obs"] = f"Error in planning: {e}"
                zwarn(f"[ProcessRewardCKAgent] Error executing plan code: {e}")
        
        # Store state in plan_res (following the original agent pattern)
        if isinstance(plan_res, dict):
            plan_res["state"] = state.copy()  # Make a copy like the original agent
        
        # Update step with planning result
        _current_step["plan"] = plan_res
        
        # Continue with action phase using original PRM
        return self._execute_action_with_prm(session, state, _input_kwargs, _extra_kwargs)
    
    def _execute_action_with_prm(self, session, state, _input_kwargs, _extra_kwargs):
        """Execute the action phase with PRM"""
        try:
            # Generate and evaluate action
            action_messages = self.templates["action"].format(**_input_kwargs)
            action_response = self.step_call(messages=action_messages, session=session)
            
            # Make sure action_res is a dictionary
            if isinstance(action_response, str):
                action_res = self._parse_output(action_response)
            else:
                action_res = action_response
                
            if not isinstance(action_res, dict):
                # Convert to a dict if still not a dict
                action_res = {"code": str(action_res), "thought": "Auto-generated thought"}
                
            # Store LLM input/output for evaluator
            action_res["llm_input"] = action_messages
            action_res["llm_output"] = action_response
            
            # Execute action with PRM if enabled
            # Note: step_action returns the observation result, not the action_res itself
            step_res = self.step_action(action_res, _input_kwargs, **_extra_kwargs)
            
            # Store observation in action_res (following the original agent pattern)
            action_res["observation"] = step_res
            
            # Store action result in current step
            _current_step = session.get_current_step()
            _current_step["action"] = action_res
            
            # Check for termination
            _should_terminate = self._check_action_termination(action_res)
            if _should_terminate:
                _current_step["terminated"] = True
                return _current_step
            
            # Return the result
            return _current_step
            
        except Exception as e:
            zwarn(f"[ProcessRewardCKAgent] Error in action execution: {e}")
            # Create a fallback action result
            _current_step = session.get_current_step()
            
            # Generate dummy messages for evaluator
            try:
                action_messages = self.templates["action"].format(**_input_kwargs)
            except:
                action_messages = "Error generating action messages"
            
            _current_step["action"] = {
                "thought": f"Error in action execution: {e}",
                "code": "# Error occurred",
                "error": str(e),
                # Required fields for evaluator
                "llm_input": action_messages,
                "llm_output": f"Error: {str(e)}"
            }
            return _current_step
        
    def _check_action_termination(self, action_res):
        """Check if the action signals termination"""
        if not action_res:
            return False
            
        # Handle if action_res is a string
        if isinstance(action_res, str):
            # Check if the string contains termination signals
            action_res_lower = action_res.lower()
            return "stop(" in action_res_lower or "terminate(" in action_res_lower
            
        # If action_res is a dictionary
        try:
            # Check for explicit termination
            if "terminated" in action_res and action_res["terminated"]:
                return True
                
            # Check for termination via code
            if "code" in action_res:
                code = action_res["code"]
                if isinstance(code, str) and ("stop(" in code.lower() or "terminate(" in code.lower()):
                    return True
                    
            return False
            
        except Exception as e:
            # If any error occurs, don't terminate
            zwarn(f"[ProcessRewardCKAgent] Error in termination check: {e}, action_res type: {type(action_res)}")
            return False
    
    def step_action(self, action_res, action_input_kwargs, **kwargs):
        """
        Override step_action to implement greedy sampling with process reward evaluation.
        Uses session-level consistency strategy to ensure all steps in a session use the same approach.
        """
        rprint(f"[ProcessRewardCKAgent] step_action called, enable_process_reward={self.enable_process_reward}")
        
        if not self.enable_process_reward or self.process_reward_model is None:
            rprint("[ProcessRewardCKAgent] Process reward disabled or model not available, using original step_action")
            return super().step_action(action_res, action_input_kwargs, **kwargs)
        
        # Extract session and state from the active execution context
        session = getattr(self, '_current_session', None)
        current_state = getattr(self, '_current_state', {})
        
        rprint(f"[ProcessRewardCKAgent] Session available: {session is not None}")
        rprint(f"[ProcessRewardCKAgent] Action code: {action_res.get('code', 'N/A')[:100]}...")
        
        if session is None:
            zwarn("[ProcessRewardCKAgent] No session available for process reward evaluation")
            return super().step_action(action_res, action_input_kwargs, **kwargs)
        
        # Apply session-level consistency strategy
        should_use_prm = self._should_use_prm_for_step(action_res, session)
        
        if not should_use_prm:
            strategy_reason = self._get_prm_skip_reason(action_res, session)
            rprint(f"[ProcessRewardCKAgent] Skipping PRM: {strategy_reason}")
            return super().step_action(action_res, action_input_kwargs, **kwargs)
        
        # Execute with PRM
        return self._execute_with_prm(action_res, action_input_kwargs, session, current_state, **kwargs)
    
    def _extract_current_state(self, action_input_kwargs) -> Dict:
        """Extract current state from action input kwargs"""
        state_str = action_input_kwargs.get('state', '{}')
        try:
            return json.loads(state_str) if isinstance(state_str, str) else state_str
        except:
            return {}
    
    def _should_use_rpm_evaluation(self, action_res) -> bool:
        """
        Determine if we should use RPM evaluation for this action.
        Skip for simple operations or when action contains certain patterns.
        """
        code = action_res.get('code', '')
        
        # Skip RPM for simple operations
        simple_patterns = [
            'print(',  # Simple print statements
            'stop(',   # Stop operations
            'len(',    # Simple length checks
        ]
        
        # Use RPM for complex operations
        complex_patterns = [
            'web_agent(',
            'file_agent(',
            'ask_llm(',
            'simple_web_search(',
        ]
        
        # If it's a simple operation, skip RPM
        if any(pattern in code for pattern in simple_patterns) and \
           not any(pattern in code for pattern in complex_patterns):
            return False
        
        return True
    
    def _should_enable_prm_for_session(self, session) -> bool:
        """
        Determine if PRM should be enabled for the entire session based on task complexity.
        This ensures consistency across all steps in the session.
        """
        task = getattr(session, 'task', '')
        if not task:
            return True  # Default to enabled if no task info
        
        # Keywords indicating complex tasks that benefit from PRM
        complex_keywords = [
            'search', 'analyze', 'compare', 'calculate', 'download', 
            'browse', 'extract', 'process', 'multiple', 'steps', 'file',
            'web', 'data', 'information', 'find', 'get', 'retrieve',
            'complex', 'difficult', 'research', 'investigation'
        ]
        
        task_lower = task.lower()
        complexity_score = sum(1 for keyword in complex_keywords if keyword in task_lower)
        
        # Task length also indicates complexity
        length_score = 1 if len(task) > 100 else 0
        
        # Enable PRM if task has complexity indicators
        total_score = complexity_score + length_score
        enable_prm = total_score >= 2
        
        rprint(f"[ProcessRewardCKAgent] Task complexity analysis: "
               f"keywords={complexity_score}, length={length_score}, "
               f"total={total_score}, enable_prm={enable_prm}")
        
        return enable_prm
    
    def _should_use_prm_for_step(self, action_res, session) -> bool:
        """
        Determine if PRM should be used for this specific step based on the session-level strategy.
        """
        # Strategy 1: Always use PRM for all steps
        if self.prm_usage_strategy == 'always':
            return True
        
        # Strategy 2: Session-level consistency
        elif self.prm_usage_strategy == 'consistent':
            # Use session-level decision
            return getattr(self, '_session_prm_enabled', True)
        
        # Strategy 3: Original selective behavior (step-level decisions)
        elif self.prm_usage_strategy == 'selective':
            # Apply original logic
            if self.num_action_candidates <= 1:
                return False
            return self._should_use_rpm_evaluation(action_res)
        
        # Default: use session-level decision
        return getattr(self, '_session_prm_enabled', True)
    
    def _get_prm_skip_reason(self, action_res, session) -> str:
        """
        Get the reason why PRM was skipped for logging purposes.
        """
        if self.prm_usage_strategy == 'always':
            return "strategy=always but conditions not met"
        elif self.prm_usage_strategy == 'consistent':
            if not getattr(self, '_session_prm_enabled', True):
                return "session-level decision: task not complex enough"
            else:
                return "session-level decision: insufficient candidates"
        elif self.prm_usage_strategy == 'selective':
            if self.num_action_candidates <= 1:
                return "insufficient candidates"
            elif not self._should_use_rpm_evaluation(action_res):
                return "simple operation detected"
        return "unknown reason"
    
    def _execute_with_prm(self, action_res, action_input_kwargs, session, current_state, **kwargs):
        """
        Execute the action with PRM evaluation.
        Stores all candidates, scores, and RPM evaluation details for visualization.
        """
        # Use adaptive sampling if enabled
        if self.enable_adaptive_sampling:
            action_candidates, scores, rpm_evaluations = self._generate_and_evaluate_adaptively(
                session, current_state, action_input_kwargs, is_planning=False)
        else:
            # Standard generation then evaluation
            # Generate multiple action candidates
            action_candidates = self._generate_action_candidates(action_input_kwargs, session)
            
            if len(action_candidates) <= 1:
                # If we couldn't generate multiple candidates, fall back to original
                rprint("[ProcessRewardCKAgent] Insufficient candidates generated, using original action")
                return super().step_action(action_res, action_input_kwargs, **kwargs)
            
            # Debug: Log the raw action candidates before evaluation
            rprint(f"[ProcessRewardCKAgent] Raw action candidates before evaluation:")
            for i, cand in enumerate(action_candidates):
                rprint(f"  Candidate {i}: type={type(cand)}")
                if isinstance(cand, dict):
                    rprint(f"    Keys: {list(cand.keys())}")
                    rprint(f"    thought: {cand.get('thought', 'N/A')[:100] if cand.get('thought') else 'N/A'}")
                    rprint(f"    code: {cand.get('code', 'N/A')[:100] if cand.get('code') else 'N/A'}")
            
            # Evaluate candidates using RPM and get detailed evaluation info
            # Important: Make sure we're using the action evaluation method, not planning
            scores, rpm_evaluations = self._evaluate_action_candidates(session, current_state, action_candidates)
        
        if len(action_candidates) <= 1:
            # If we couldn't generate multiple candidates (even with adaptive), fall back
            rprint("[ProcessRewardCKAgent] Insufficient candidates generated, using original action")
            return super().step_action(action_res, action_input_kwargs, **kwargs)
        
        # Select best candidate
        best_candidate, best_score, best_idx = self._select_best_candidate(action_candidates, scores)
        
        rprint(f"[ProcessRewardCKAgent] Selected candidate {best_idx} with score {best_score:.3f}")
        
        # CRITICAL: Update action_res with best_candidate's content
        # The base class will save action_res to session, so we need to replace its thought/code
        # with the selected candidate's thought/code (while preserving other fields like llm_input/output)
        action_res['thought'] = best_candidate.get('thought', action_res.get('thought', ''))
        action_res['code'] = best_candidate.get('code', action_res.get('code', ''))
        
        # Create safe candidates for storage (avoid circular references)
        safe_candidates = []
        for cand in action_candidates:
            if isinstance(cand, dict):
                # Only store essential fields to avoid circular references
                safe_cand = {
                    'thought': cand.get('thought', ''),
                    'code': cand.get('code', ''),
                }
                # Add diversity info if present
                if 'diversity_info' in cand:
                    safe_cand['diversity_info'] = cand['diversity_info']
                safe_candidates.append(safe_cand)
            else:
                safe_candidates.append({'thought': str(cand), 'code': ''})
        
        # Store PRM information in action_res for output serialization
        action_res['prm_info'] = {
            'all_candidates': safe_candidates,  # All generated candidates (safe version)
            'scores': scores,  # RPM scores for each candidate
            'selected_idx': best_idx,  # Index of selected candidate
            'rpm_evaluations': rpm_evaluations,  # Detailed RPM evaluation outputs
            'num_candidates': len(action_candidates),
            'evaluation_method': 'process_reward_model',
            # Add diversity information if enabled
            'diversity_enabled': self.enable_diverse_sampling and self.diverse_sampler is not None,
            'sequential_mode': self.sequential_mode if hasattr(self, 'sequential_mode') else False,
            'diversity_threshold': self.diversity_threshold if hasattr(self, 'diversity_threshold') else 0.0
        }
        
        # Execute the best candidate
        # Note: action_res now contains best_candidate's thought/code, so both execution and logging are consistent
        return super().step_action(best_candidate, action_input_kwargs, **kwargs)
    
    def _generate_action_candidates(self, action_input_kwargs, session) -> List[Dict]:
        """
        Generate multiple action candidates using different sampling strategies.
        If diverse_sampler is enabled, use it for sequential diverse generation.
        """
        # If diverse sampling is enabled and we have a sampler, use it
        if self.enable_diverse_sampling and self.diverse_sampler and self.sequential_mode:
            rprint(f"[ProcessRewardCKAgent] Using diverse sequential sampler for {self.num_action_candidates} candidates")
            return self._generate_diverse_candidates(action_input_kwargs, session)
        
        # Otherwise use the original sampling approach
        candidates = []
        
        if self.enable_parallel_sampling and self.num_action_candidates > 1:
            # Parallel generation - following existing patterns for stability
            pool_size = min(3, self.num_action_candidates)  # Conservative pool size
            with mp.Pool(pool_size) as pool:
                args_list = [(action_input_kwargs, i) for i in range(self.num_action_candidates)]
                try:
                    candidates = pool.map(self._generate_single_candidate_wrapper, args_list)
                except Exception as e:
                    zwarn(f"[ProcessRewardCKAgent] Parallel generation failed: {e}, falling back to sequential")
                    # Fallback to sequential generation
                    candidates = []
                    for i in range(self.num_action_candidates):
                        candidate = self._generate_single_candidate(action_input_kwargs, i)
                        candidates.append(candidate)
        else:
            # Sequential generation
            candidates = []
            for i in range(self.num_action_candidates):
                candidate = self._generate_single_candidate(action_input_kwargs, i)
                candidates.append(candidate)
        
        # Filter out None candidates
        valid_candidates = [c for c in candidates if c is not None]
        
        rprint(f"[ProcessRewardCKAgent] Generated {len(valid_candidates)} valid candidates "
               f"out of {self.num_action_candidates} attempts")
        
        return valid_candidates
        
    def _generate_diverse_candidates(self, action_input_kwargs, session) -> List[Dict]:
        """
        Generate multiple action candidates using diverse sequential sampling.
        This ensures each candidate is sufficiently different from the others.
        """
        try:
            rprint("[ProcessRewardCKAgent] Using diverse sequential sampling")
            
            # Reset diverse sampler's candidate history
            self.diverse_sampler.reset_candidates()
            
            # Generate action template messages
            action_messages = self.templates["action"].format(**action_input_kwargs)
            
            # Use diverse_sampler to generate multiple candidates
            raw_responses = self.diverse_sampler.generate_candidates(
                model=self.model,  # Use the agent's model for generation
                prompt=action_messages,
                num_candidates=self.num_action_candidates,
                temperature=self.sampling_temperature,
                max_tokens=2000  # 使用固定值而不是尝试访问不存在的属性
                # Note: Don't pass session here as it's not a valid parameter for LLM API
            )
            
            # Parse responses into action candidates
            candidates = []
            for i, response in enumerate(raw_responses):
                try:
                    action_res = self._parse_output(response)
                    # Store diversity info in candidate
                    action_res['diversity_info'] = {
                        'candidate_idx': i,
                        'diverse_sampling': True,
                        'diversity_threshold': self.diversity_threshold,
                    }
                    candidates.append(action_res)
                except Exception as e:
                    zwarn(f"[ProcessRewardCKAgent] Failed to parse diverse candidate {i}: {e}")
            
            # Filter out None candidates
            valid_candidates = [c for c in candidates if c is not None]
            
            rprint(f"[ProcessRewardCKAgent] Generated {len(valid_candidates)} valid diverse candidates "
                  f"out of {self.num_action_candidates} attempts")
            
            # Include diversity info in PRM data
            if valid_candidates:
                # Add diversity flag to the first candidate so it's visible in UI
                valid_candidates[0]['diversity_enabled'] = True
            
            return valid_candidates
            
        except Exception as e:
            zwarn(f"[ProcessRewardCKAgent] Diverse candidate generation failed: {e}, falling back to standard generation")
            # Fallback to standard generation
            return self._generate_standard_candidates(action_input_kwargs, session)
            
    def _generate_standard_candidates(self, action_input_kwargs, session) -> List[Dict]:
        """Fallback method for standard candidate generation when diverse sampling fails"""
        candidates = []
        for i in range(self.num_action_candidates):
            candidate = self._generate_single_candidate(action_input_kwargs, i)
            candidates.append(candidate)
            
        # Filter out None candidates
        valid_candidates = [c for c in candidates if c is not None]
        return valid_candidates
    
    def _generate_and_evaluate_adaptively(self, session, current_state, 
                                          input_kwargs, is_planning=False) -> Tuple[List[Dict], List[float], List[Dict]]:
        """
        Generate and evaluate candidates adaptively - stop early if we find a good one.
        This saves computation by not generating unnecessary candidates.
        
        Args:
            session: Current session
            current_state: Current progress state
            input_kwargs: Input parameters for generation
            is_planning: Whether this is for planning (True) or action (False)
        
        Returns:
            Tuple of (candidates, scores, evaluations)
        """
        rprint(f"[ProcessRewardCKAgent] Using adaptive sampling (threshold={self.adaptive_score_threshold}, "
               f"min_candidates={self.adaptive_min_candidates})")
        
        all_candidates = []
        all_scores = []
        all_evaluations = []
        best_score = -1.0
        
        # Reset diverse sampler if using diverse sampling
        if self.enable_diverse_sampling and hasattr(self, 'diverse_sampler'):
            self.diverse_sampler.reset_candidates()
        
        for i in range(self.num_action_candidates):
            # Generate one candidate
            rprint(f"[ProcessRewardCKAgent] Generating adaptive candidate {i+1}/{self.num_action_candidates}")
            
            if is_planning:
                # Generate planning candidate
                if self.enable_diverse_sampling and i > 0:
                    # For subsequent candidates, use diverse sampling
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
                    candidate = self._generate_single_plan_candidate(input_kwargs, i, session)
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
                    candidate = self._generate_single_candidate(input_kwargs, i)
            
            if candidate is None:
                continue
            
            all_candidates.append(candidate)
            
            # Evaluate this candidate immediately
            if is_planning:
                # Use planning evaluation
                scores, evals = self._evaluate_plan_candidates(session, current_state, [candidate])
            else:
                # Use action evaluation
                scores, evals = self._evaluate_action_candidates(session, current_state, [candidate])
            
            if scores:
                score = scores[0]
                all_scores.append(score)
                all_evaluations.append(evals[0] if evals else {})
                
                rprint(f"[ProcessRewardCKAgent] Adaptive candidate {i+1} score: {score:.3f}")
                
                if score > best_score:
                    best_score = score
                
                # Check if we should stop early
                if (i + 1 >= self.adaptive_min_candidates and 
                    score >= self.adaptive_score_threshold):
                    rprint(f"[ProcessRewardCKAgent] Adaptive sampling: Found good candidate "
                          f"(score={score:.3f} >= {self.adaptive_score_threshold}), "
                          f"stopping after {i+1} candidates (saved {self.num_action_candidates - i - 1} generations)")
                    break
            else:
                # No score returned, use default
                all_scores.append(0.5)
                all_evaluations.append({})
        
        rprint(f"[ProcessRewardCKAgent] Adaptive sampling completed: {len(all_candidates)} candidates, "
               f"best_score={best_score:.3f}")
        
        return all_candidates, all_scores, all_evaluations
    
    def _generate_single_plan_candidate(self, input_kwargs, candidate_idx, session):
        """Generate a single planning candidate"""
        try:
            original_seed = self.get_seed()
            candidate_seed = (original_seed + candidate_idx + 100) if original_seed else (candidate_idx + 100)
            
            try:
                self.set_seed(candidate_seed)
                plan_messages = self.templates["plan"].format(**input_kwargs)
                plan_response = self.step_call(messages=plan_messages, session=session)
                plan_res = self._parse_output(plan_response)
                return plan_res
            finally:
                self.set_seed(original_seed)
        except Exception as e:
            zwarn(f"[ProcessRewardCKAgent] Failed to generate planning candidate {candidate_idx}: {e}")
            return None
    
    def _generate_single_candidate_wrapper(self, args):
        """Wrapper for parallel candidate generation"""
        return self._generate_single_candidate(*args)
    
    def _generate_single_candidate(self, action_input_kwargs, candidate_idx) -> Dict:
        """
        Generate a single action candidate with varied sampling.
        Uses only seed variation to maintain consistency with normal generation config.
        Modern LLMs provide sufficient diversity even with temperature=0.
        """
        try:
            # Vary only the seed for diversity, keep temperature and other configs unchanged
            original_seed = self.get_seed()
            candidate_seed = (original_seed + candidate_idx) if original_seed else candidate_idx
            
            try:
                # Set different seed for each candidate
                self.set_seed(candidate_seed)
                
                # Generate action using the template system with original generation config
                action_messages = self.templates["action"].format(**action_input_kwargs)
                # Use current session for consistency
                current_session = getattr(self, '_current_session', None)
                action_response = self.step_call(messages=action_messages, session=current_session)
                action_res = self._parse_output(action_response)
                
                return action_res
                
            finally:
                # Restore original seed
                self.set_seed(original_seed)
                    
        except Exception as e:
            zwarn(f"[ProcessRewardCKAgent] Failed to generate candidate {candidate_idx}: {e}")
            return None
    
    def _evaluate_action_candidates(self, session, current_state, candidates) -> Tuple[List[float], List[Dict]]:
        """
        Evaluate action candidates using either score-based or ranking-based PRM.
        """
        try:
            session_history = session.steps[:-1] if session.steps else []
            
            # Debug: Log what we're about to evaluate
            rprint(f"[ProcessRewardCKAgent] Evaluating {len(candidates)} action candidates")
            for i, cand in enumerate(candidates):
                if isinstance(cand, dict):
                    rprint(f"  Action candidate {i}: thought={cand.get('thought', 'N/A')[:80]}..., code={cand.get('code', 'N/A')[:80]}...")
                else:
                    rprint(f"  Action candidate {i}: {type(cand)}")
            
            # Check PRM mode
            if self.prm_mode == 'ranking':
                # Use Ranking PRM - evaluate all candidates together
                return self._rank_action_candidates(session, current_state, candidates, session_history)
            else:
                # Use Score-based PRM (existing behavior)
                scores, evaluations = self.process_reward_model.evaluate_action_candidates_detailed(
                    task=session.task,
                    session_history=session_history,
                    current_state=current_state,
                    action_candidates=candidates
                )
                
                # Add evaluation phase marker
                for eval_info in evaluations:
                    if isinstance(eval_info, dict):
                        eval_info['phase'] = 'action'
                
                rprint(f"[ProcessRewardCKAgent] Action evaluation completed, scores: {scores}")
                return scores, evaluations
            
        except Exception as e:
            zwarn(f"[ProcessRewardCKAgent] Action evaluation failed: {e}")
            import traceback
            zwarn(f"Traceback: {traceback.format_exc()}")
            return [0.5] * len(candidates), [{"error": str(e), "phase": "action"} for _ in candidates]
    
    def _rank_action_candidates(self, session, current_state, candidates, session_history) -> Tuple[List[float], List[Dict]]:
        """Use Ranking PRM to rank action candidates and convert to scores"""
        try:
            rprint(f"[ProcessRewardCKAgent] Using Ranking PRM for {len(candidates)} action candidates")
            
            # Call ranking model
            ranking_indices, eval_info = self.ranking_reward_model.rank_candidates(
                task=session.task,
                session_history=session_history,
                current_state=current_state,
                candidates=candidates,
                is_planning=False
            )
            
            # Convert ranking to scores
            scores = self._ranking_to_scores(ranking_indices, len(candidates))
            
            # Create evaluation info for each candidate
            evaluations = []
            for i in range(len(candidates)):
                rank_position = ranking_indices.index(i) if i in ranking_indices else len(candidates)
                eval_dict = {
                    'candidate_idx': i,
                    'rank_position': rank_position,
                    'score': scores[i],
                    'ranking_method': 'ranking_prm',
                    'raw_ranking': ranking_indices,
                    'phase': 'action'
                }
                eval_dict.update(eval_info)
                evaluations.append(eval_dict)
            
            rprint(f"[ProcessRewardCKAgent] Ranking PRM scores: {scores}")
            return scores, evaluations
            
        except Exception as e:
            zwarn(f"[ProcessRewardCKAgent] Ranking PRM failed: {e}")
            return [0.5] * len(candidates), [{"error": str(e), "phase": "action"} for _ in candidates]
    
    def _ranking_to_scores(self, ranking_indices: List[int], num_candidates: int) -> List[float]:
        """
        Convert ranking indices to scores.
        Best candidate (rank 0) gets score 1.0, worst gets score approaching 0.0
        
        Args:
            ranking_indices: List like [2, 0, 1] meaning candidate 2 is best, 0 is second, 1 is worst
            num_candidates: Total number of candidates
            
        Returns:
            List of scores where scores[i] is the score for candidate i
        """
        scores = [0.0] * num_candidates
        
        for rank_position, candidate_idx in enumerate(ranking_indices):
            # Linear scoring: best=1.0, worst=0.1
            # This ensures best candidate has highest score for selection
            score = 1.0 - (rank_position / max(num_candidates - 1, 1)) * 0.9
            scores[candidate_idx] = score
        
        return scores
    
    def _select_best_candidate(self, candidates, scores) -> Tuple[Dict, float, int]:
        """
        Select the best candidate based on RPM scores.
        Returns: (best_candidate, best_score, best_idx)
        """
        if not candidates or not scores:
            raise ValueError("No candidates or scores provided")
        
        if len(candidates) != len(scores):
            zwarn(f"[ProcessRewardCKAgent] Mismatch between candidates ({len(candidates)}) "
                  f"and scores ({len(scores)})")
            # Truncate to minimum length
            min_len = min(len(candidates), len(scores))
            candidates = candidates[:min_len]
            scores = scores[:min_len]
        
        # Find the best candidate
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best_candidate = candidates[best_idx]
        best_score = scores[best_idx]
        
        # Log selection reasoning
        score_summary = [f"C{i}: {s:.3f}" for i, s in enumerate(scores)]
        rprint(f"[ProcessRewardCKAgent] Candidate scores: {', '.join(score_summary)} -> Selected C{best_idx}")
        
        # Check if best score meets threshold
        if best_score < self.min_score_threshold:
            zwarn(f"[ProcessRewardCKAgent] Best score {best_score:.3f} below threshold {self.min_score_threshold}")
        
        return best_candidate, best_score, best_idx
    
    def _generate_plan_candidates(self, input_kwargs, session) -> List[Dict]:
        """
        Generate multiple planning candidates.
        Similar to _generate_action_candidates but for planning phase.
        """
        # If diverse sampling is enabled and we have a sampler, use it
        if self.enable_diverse_sampling and self.diverse_sampler and self.sequential_mode:
            rprint(f"[ProcessRewardCKAgent] Using diverse sequential sampler for {self.num_action_candidates} planning candidates")
            return self._generate_diverse_plan_candidates(input_kwargs, session)
        
        candidates = []
        # Sequential generation for planning
        for i in range(self.num_action_candidates):
            try:
                # Vary only the seed for diversity
                original_seed = self.get_seed()
                candidate_seed = (original_seed + i + 100) if original_seed else (i + 100)  # Different range from action seeds
                
                try:
                    # Set different seed for each candidate
                    self.set_seed(candidate_seed)
                    
                    # Generate planning using template
                    plan_messages = self.templates["plan"].format(**input_kwargs)
                    plan_response = self.step_call(messages=plan_messages, session=session)
                    plan_res = self._parse_output(plan_response)
                    
                    # Debug: Log what we generated
                    rprint(f"[ProcessRewardCKAgent] Generated planning candidate {i}:")
                    rprint(f"  Response length: {len(plan_response) if plan_response else 0}")
                    rprint(f"  Parsed thought: {plan_res.get('thought', 'N/A')[:100] if plan_res.get('thought') else 'EMPTY'}")
                    rprint(f"  Parsed code: {plan_res.get('code', 'N/A')[:100] if plan_res.get('code') else 'EMPTY'}")
                    
                    candidates.append(plan_res)
                    
                finally:
                    # Restore original seed
                    self.set_seed(original_seed)
                    
            except Exception as e:
                zwarn(f"[ProcessRewardCKAgent] Failed to generate planning candidate {i}: {e}")
        
        # Filter out None candidates
        valid_candidates = [c for c in candidates if c is not None]
        
        rprint(f"[ProcessRewardCKAgent] Generated {len(valid_candidates)} valid planning candidates "
               f"out of {self.num_action_candidates} attempts")
        
        return valid_candidates
    
    def _generate_diverse_plan_candidates(self, input_kwargs, session) -> List[Dict]:
        """
        Generate planning candidates using diverse sequential sampling.
        """
        try:
            rprint("[ProcessRewardCKAgent] Using diverse sequential sampling for planning")
            
            # Reset diverse sampler's candidate history
            self.diverse_sampler.reset_candidates()
            
            # Generate plan template messages
            plan_messages = self.templates["plan"].format(**input_kwargs)
            
            # Use diverse_sampler to generate multiple candidates
            raw_responses = self.diverse_sampler.generate_candidates(
                model=self.model,
                prompt=plan_messages,
                num_candidates=self.num_action_candidates,
                temperature=self.sampling_temperature,
                max_tokens=2000  # 使用固定值而不是尝试访问不存在的属性
                # Note: Don't pass session here as it's not a valid parameter for LLM API
            )
            
            # Parse responses into planning candidates
            candidates = []
            for i, response in enumerate(raw_responses):
                try:
                    plan_res = self._parse_output(response)
                    
                    # Debug: Log what we generated
                    rprint(f"[ProcessRewardCKAgent] Parsed diverse planning candidate {i}:")
                    rprint(f"  Response length: {len(response) if response else 0}")
                    rprint(f"  Parsed thought: {plan_res.get('thought', 'N/A')[:100] if plan_res.get('thought') else 'EMPTY'}")
                    rprint(f"  Parsed code: {plan_res.get('code', 'N/A')[:100] if plan_res.get('code') else 'EMPTY'}")
                    
                    # Store diversity info in candidate
                    plan_res['diversity_info'] = {
                        'candidate_idx': i,
                        'diverse_sampling': True,
                        'diversity_threshold': self.diversity_threshold,
                    }
                    candidates.append(plan_res)
                except Exception as e:
                    zwarn(f"[ProcessRewardCKAgent] Failed to parse diverse planning candidate {i}: {e}")
            
            # Filter out None candidates
            valid_candidates = [c for c in candidates if c is not None]
            
            rprint(f"[ProcessRewardCKAgent] Generated {len(valid_candidates)} valid diverse planning candidates "
                  f"out of {self.num_action_candidates} attempts")
            
            # Include diversity info in PRM data
            if valid_candidates:
                # Add diversity flag to the first candidate so it's visible in UI
                valid_candidates[0]['diversity_enabled'] = True
            
            return valid_candidates
            
        except Exception as e:
            zwarn(f"[ProcessRewardCKAgent] Diverse planning candidate generation failed: {e}, falling back to standard")
            return self._generate_standard_plan_candidates(input_kwargs, session)
    
    def _generate_standard_plan_candidates(self, input_kwargs, session) -> List[Dict]:
        """Standard plan candidate generation fallback"""
        candidates = []
        for i in range(self.num_action_candidates):
            try:
                plan_messages = self.templates["plan"].format(**input_kwargs)
                plan_response = self.step_call(messages=plan_messages, session=session)
                plan_res = self._parse_output(plan_response)
                candidates.append(plan_res)
            except Exception as e:
                zwarn(f"[ProcessRewardCKAgent] Failed to generate standard planning candidate {i}: {e}")
        
        return [c for c in candidates if c is not None]
    
    def _evaluate_plan_candidates(self, session, current_state, candidates) -> Tuple[List[float], List[Dict]]:
        """
        Evaluate planning candidates using either score-based or ranking-based PRM.
        """
        try:
            session_history = session.steps[:-1] if session.steps else []
            
            # Check PRM mode
            if self.prm_mode == 'ranking':
                # Use Ranking PRM - evaluate all candidates together
                return self._rank_plan_candidates(session, current_state, candidates, session_history)
            else:
                # Use Score-based PRM (existing behavior)
                # Convert candidates from agent format (thought/code) to planning format (reasoning/plan)
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
                
                # Log the converted candidates for debugging
                rprint(f"[ProcessRewardCKAgent] Converted {len(converted_candidates)} planning candidates")
                for i, conv in enumerate(converted_candidates):
                    rprint(f"  Candidate {i}: reasoning={conv.get('reasoning', 'N/A')[:50]}..., plan={conv.get('plan', 'N/A')[:50]}...")
                
                # Use specific method for planning evaluation if available
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
            zwarn(f"[ProcessRewardCKAgent] Planning evaluation failed: {e}")
            import traceback
            zwarn(f"Traceback: {traceback.format_exc()}")
            return [0.5] * len(candidates), [{"error": str(e)} for _ in candidates]
    
    def _rank_plan_candidates(self, session, current_state, candidates, session_history) -> Tuple[List[float], List[Dict]]:
        """Use Ranking PRM to rank planning candidates and convert to scores"""
        try:
            rprint(f"[ProcessRewardCKAgent] Using Ranking PRM for {len(candidates)} planning candidates")
            
            # Call ranking model
            ranking_indices, eval_info = self.ranking_reward_model.rank_candidates(
                task=session.task,
                session_history=session_history,
                current_state=current_state,
                candidates=candidates,
                is_planning=True
            )
            
            # Convert ranking to scores (best=1.0, worst=0.0)
            scores = self._ranking_to_scores(ranking_indices, len(candidates))
            
            # Create evaluation info for each candidate
            evaluations = []
            for i in range(len(candidates)):
                rank_position = ranking_indices.index(i) if i in ranking_indices else len(candidates)
                eval_dict = {
                    'candidate_idx': i,
                    'rank_position': rank_position,
                    'score': scores[i],
                    'ranking_method': 'ranking_prm',
                    'raw_ranking': ranking_indices,
                }
                eval_dict.update(eval_info)
                evaluations.append(eval_dict)
            
            rprint(f"[ProcessRewardCKAgent] Ranking PRM scores: {scores}")
            return scores, evaluations
            
        except Exception as e:
            zwarn(f"[ProcessRewardCKAgent] Ranking PRM failed: {e}")
            return [0.5] * len(candidates), [{"error": str(e)} for _ in candidates]
    
    def get_call_stat(self, clear: bool):
        """Get call statistics including RPM statistics"""
        stats = super().get_call_stat(clear)
        
        if self.process_reward_model:
            stats["process_reward_model"] = self.process_reward_model.get_call_stat(clear)
        
        return stats
    
    def set_seed(self, seed):
        """Set seed for both main agent and RPM"""
        super().set_seed(seed)
        if self.process_reward_model:
            self.process_reward_model.set_seed(seed)
    
    def reset_session_state(self):
        """
        Reset session-level state. Call this between different sessions/tasks.
        """
        self._session_prm_enabled = None
        self._current_session = None
        self._current_state = None
        rprint("[ProcessRewardCKAgent] Session state reset")
