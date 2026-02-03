#!/usr/bin/env python3
"""
Test script for Process Reward Model integration
"""

import os
import sys
import json

from pathlib import Path

# Add the project path to sys.path (portable; avoids hardcoded local absolute paths)
_gaia_path = os.getenv("COGNITIVE_KERNEL_GAIA_PATH", "").strip()
if _gaia_path:
    sys.path.insert(0, _gaia_path)
else:
    # Repo layout expected:
    #   AgentRPM/
    #     CSO/system/ckv3/test_process_reward.py  (this file)
    #     cognitive_kernel_GAIA/
    repo_root = Path(__file__).resolve().parents[3]  # .../AgentRPM
    candidate = repo_root / "cognitive_kernel_GAIA"
    if candidate.exists():
        sys.path.insert(0, str(candidate))
    else:
        raise RuntimeError(
            "Cannot locate 'cognitive_kernel_GAIA'. Set COGNITIVE_KERNEL_GAIA_PATH or place "
            "'cognitive_kernel_GAIA' next to 'CSO' in the repo root."
        )

from ckv3.agents.process_reward_model import ProcessRewardModel
from ckv3.ck_main.process_reward_agent import ProcessRewardCKAgent


def test_process_reward_model():
    """Test the ProcessRewardModel independently"""
    print("=" * 50)
    print("Testing ProcessRewardModel")
    print("=" * 50)
    
    # Initialize RPM
    rpm_config = {
        'call_target': 'claude:claude-3.7',
        'enable_parallel_evaluation': False,  # Disable for testing
        'evaluation_timeout': 30,
    }
    
    try:
        rpm = ProcessRewardModel(**rpm_config)
        print("✓ ProcessRewardModel initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize ProcessRewardModel: {e}")
        return False
    
    # Test evaluation
    task = "Find the current price of a Tesla Model 3"
    session_history = [
        {
            'step_idx': 0,
            'action': {
                'thought': 'I should search for Tesla Model 3 pricing information',
                'code': 'result = simple_web_search("Tesla Model 3 price 2024")\nprint(result)',
                'observation': 'Found search results with various Tesla pricing websites'
            }
        }
    ]
    current_state = {
        "completed_list": ["Searched for Tesla Model 3 pricing information"],
        "todo_list": ["Visit Tesla website to get official pricing", "Check multiple sources for comparison"],
        "experience": [],
        "information": ["Tesla Model 3 has different variants with different prices"]
    }
    
    action_candidates = [
        {
            'thought': 'I should visit the official Tesla website to get accurate pricing',
            'code': 'result = web_agent("Visit https://tesla.com and find the current price of Tesla Model 3")\nprint(result)'
        },
        {
            'thought': 'I should search for Tesla Model 3 reviews that might contain pricing',
            'code': 'result = simple_web_search("Tesla Model 3 review price comparison")\nprint(result)'
        },
        {
            'thought': 'I should stop and provide a general answer',
            'code': 'stop("Around $40,000", "Tesla Model 3 pricing varies by configuration")'
        }
    ]
    
    try:
        scores = rpm.evaluate_action_candidates(task, session_history, current_state, action_candidates)
        print(f"✓ RPM evaluation completed. Scores: {scores}")
        
        if len(scores) == len(action_candidates):
            print("✓ Correct number of scores returned")
        else:
            print(f"✗ Expected {len(action_candidates)} scores, got {len(scores)}")
            return False
            
        # Check score ranges
        for i, score in enumerate(scores):
            if 0.0 <= score <= 1.0:
                print(f"✓ Score {i}: {score:.3f} (valid range)")
            else:
                print(f"✗ Score {i}: {score:.3f} (invalid range)")
                return False
                
    except Exception as e:
        print(f"✗ RPM evaluation failed: {e}")
        return False
    
    return True


def test_process_reward_agent():
    """Test the ProcessRewardCKAgent configuration"""
    print("\n" + "=" * 50)
    print("Testing ProcessRewardCKAgent")
    print("=" * 50)
    
    # Test configuration
    config = {
        "model": {"call_target": "claude:claude-3.7"},
        "process_reward": {
            "enable_process_reward": True,
            "num_action_candidates": 2,
            "enable_parallel_sampling": False,  # Disable for testing
            "sampling_temperature": 0.8,  # No longer used - only for backward compatibility
            "min_score_threshold": 0.3,
            "rpm_model_config": {
                "call_target": "claude:claude-3.7",
                "enable_parallel_evaluation": False,
                "evaluation_timeout": 30,
            }
        }
    }
    
    try:
        agent = ProcessRewardCKAgent(**config)
        print("✓ ProcessRewardCKAgent initialized successfully")
        print(f"  - Process reward enabled: {agent.enable_process_reward}")
        print(f"  - Number of candidates: {agent.num_action_candidates}")
        print(f"  - Sampling temperature (deprecated): {agent.sampling_temperature}")
        
        if agent.process_reward_model is not None:
            print("✓ Process Reward Model properly initialized")
        else:
            print("✗ Process Reward Model not initialized")
            return False
            
    except Exception as e:
        print(f"✗ Failed to initialize ProcessRewardCKAgent: {e}")
        return False
    
    return True


def main():
    """Main test function"""
    print("Process Reward Model Integration Test")
    print("====================================")
    
    # Run tests
    rpm_test_passed = test_process_reward_model()
    agent_test_passed = test_process_reward_agent()
    
    print("\n" + "=" * 50)
    print("Test Results Summary")
    print("=" * 50)
    print(f"ProcessRewardModel test: {'PASSED' if rpm_test_passed else 'FAILED'}")
    print(f"ProcessRewardCKAgent test: {'PASSED' if agent_test_passed else 'FAILED'}")
    
    if rpm_test_passed and agent_test_passed:
        print("\n🎉 All tests passed! Process Reward Model integration is ready.")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the issues above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
