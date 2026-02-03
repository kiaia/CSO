# 
# Diverse Sequential Sampling implementation
# For generating diverse candidates in sequential manner

import json
import torch
import time
from typing import List, Dict, Any, Tuple, Optional, Union
from .model import LLM
from .utils import rprint, zwarn


class DiverseSequentialSampler:
    """
    Implements diverse sequential sampling for generating multiple diverse candidates.
    When generating candidates, each new candidate is required to be different from previously generated ones.
    """
    
    def __init__(self, **kwargs):
        # Sampling configuration
        self.enable_diverse_sampling = kwargs.pop('enable_diverse_sampling', False)
        self.sequential_mode = kwargs.pop('sequential_mode', False)  # If True, generates sequentially
        self.diversity_threshold = kwargs.pop('diversity_threshold', 0.4)  # Minimum required diversity
        self.max_sampling_attempts = kwargs.pop('max_sampling_attempts', 5)  # Maximum attempts to generate diverse samples
        self.diversity_prompt_strength = kwargs.pop('diversity_prompt_strength', "medium")  # How strongly to encourage diversity
        
        # Model for similarity checking (can use same as generation model)
        self.use_external_similarity = kwargs.pop('use_external_similarity', False)
        if self.use_external_similarity:
            similarity_kwargs = kwargs.copy()
            self.similarity_model = LLM(**similarity_kwargs)
        else:
            self.similarity_model = None
        
        # Keeping track of previously generated candidates
        self.previous_candidates = []
    
    def reset_candidates(self):
        """Reset the history of previously generated candidates"""
        self.previous_candidates = []
    
    def generate_candidates(self, model: LLM, prompt: List[Dict], num_candidates: int = 3, 
                          temperature: float = 0.8, max_tokens: int = 2000, **kwargs) -> List[str]:
        """
        Generate multiple diverse candidates sequentially
        
        Args:
            model: LLM instance to use for generation
            prompt: The prompt to use for generation (list of message dicts)
            num_candidates: Number of candidates to generate
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters to pass to the model
            
        Returns:
            List of candidate responses as strings
        """
        if not self.enable_diverse_sampling or not self.sequential_mode:
            # If diverse sequential sampling is disabled, use standard parallel sampling
            return self._generate_standard(model, prompt, num_candidates, temperature, max_tokens, **kwargs)
        
        # Sequential diverse sampling
        candidates = []
        
        for i in range(num_candidates):
            # Create diversity-enhanced prompt for subsequent candidates
            current_prompt = self._enhance_prompt_for_diversity(prompt, candidates, i)
            
            # Generate candidate
            for attempt in range(self.max_sampling_attempts):
                try:
                    # Add small delay between API calls if needed
                    if i > 0 and attempt > 0:
                        time.sleep(0.5)
                        
                    # Generate response
                    response = model(current_prompt, temperature=temperature, max_tokens=max_tokens, **kwargs)
                    
                    # Check if it's diverse enough from previous candidates
                    if i == 0 or self._is_sufficiently_diverse(response, candidates):
                        candidates.append(response)
                        self.previous_candidates.append(response)
                        break
                    else:
                        # Not diverse enough, strengthen diversity prompt and retry
                        rprint(f"[DSS] Candidate {i+1} not diverse enough (attempt {attempt+1}), retrying...")
                        current_prompt = self._strengthen_diversity_prompt(current_prompt, candidates, attempt)
                        
                except Exception as e:
                    zwarn(f"[DSS] Error generating candidate {i+1} (attempt {attempt+1}): {e}")
                    if attempt == self.max_sampling_attempts - 1:
                        # Last attempt failed, use a dummy placeholder
                        candidates.append(f"Error generating candidate {i+1}: {str(e)}")
        
        return candidates
    
    def _generate_standard(self, model: LLM, prompt: List[Dict], num_candidates: int = 3, 
                         temperature: float = 0.8, max_tokens: int = 2000, **kwargs) -> List[str]:
        """
        Generate candidates using standard approach (no sequential diversity)
        """
        responses = []
        
        # Generate each candidate independently
        for i in range(num_candidates):
            try:
                if i > 0:
                    time.sleep(0.5)  # Small delay between API calls
                
                # Adjust temperature slightly for diversity without explicit enforcement
                adj_temperature = temperature * (1.0 + i * 0.1)
                adj_temperature = min(adj_temperature, 1.0)  # Cap temperature
                
                response = model(prompt, temperature=adj_temperature, max_tokens=max_tokens, **kwargs)
                responses.append(response)
                
            except Exception as e:
                zwarn(f"[DSS] Error generating candidate {i+1}: {e}")
                responses.append(f"Error generating candidate {i+1}: {str(e)}")
        
        return responses
    
    def _enhance_prompt_for_diversity(self, prompt: List[Dict], previous_responses: List[str], 
                                   candidate_idx: int) -> List[Dict]:
        """
        Enhance the prompt to encourage diversity based on previous responses
        """
        if candidate_idx == 0 or not previous_responses:
            # No enhancement needed for first candidate
            return prompt
        
        # Clone the original prompt
        enhanced_prompt = prompt.copy()
        
        # Strength levels for diversity prompting
        strength_levels = {
            "low": "Please generate a response that is somewhat different from previous ones.",
            "medium": "Please generate a significantly different response from previous ones, with different approach or reasoning.",
            "high": "Generate a completely different solution from previous ones. Use a fundamentally different approach, strategy, or perspective.",
            "very_high": "IMPORTANT: You MUST generate a completely different solution. Previous solutions were: "
        }
        
        # Get diversity instruction based on configured strength
        diversity_instruction = strength_levels.get(self.diversity_prompt_strength, strength_levels["medium"])
        
        # For higher strength levels, include snippets of previous responses
        if self.diversity_prompt_strength == "very_high" and previous_responses:
            # Include brief summaries of previous responses
            prev_examples = []
            for i, resp in enumerate(previous_responses):
                # Extract just the first part of each response (to save context)
                summary = resp[:100] + "..." if len(resp) > 100 else resp
                prev_examples.append(f"Response {i+1}: {summary}")
            
            diversity_instruction += "\n\n" + "\n".join(prev_examples)
            diversity_instruction += "\n\nYour response MUST be substantially different in approach and content."
        
        # Find the user message and add the diversity instruction
        for i, msg in enumerate(enhanced_prompt):
            if msg["role"] == "user":
                # Add diversity instruction to the end of the user message
                content = msg.get("content")
                # content can be a str or a list (multimodal). Handle both.
                if isinstance(content, str):
                    new_content = content + "\n\n" + diversity_instruction
                elif isinstance(content, list):
                    # Append as a text block to avoid list+str concatenation errors
                    new_content = content + [{"type": "text", "text": "\n\n" + diversity_instruction}]
                else:
                    # Fallback: cast to str and append
                    new_content = f"{content}\n\n{diversity_instruction}"
                enhanced_prompt[i] = {
                    "role": "user",
                    "content": new_content
                }
                break
        
        return enhanced_prompt
    
    def _strengthen_diversity_prompt(self, prompt: List[Dict], previous_responses: List[str], 
                                  attempt: int) -> List[Dict]:
        """
        Strengthen the diversity prompt after failed attempts
        """
        # Determine strength based on attempt number
        strengths = ["medium", "high", "very_high", "very_high"]
        strength = strengths[min(attempt, len(strengths) - 1)]
        
        # Override the current strength setting for this attempt
        old_strength = self.diversity_prompt_strength
        self.diversity_prompt_strength = strength
        
        # Create enhanced prompt with the stronger diversity instruction
        enhanced_prompt = self._enhance_prompt_for_diversity(prompt, previous_responses, 1)
        
        # Restore original strength setting
        self.diversity_prompt_strength = old_strength
        
        return enhanced_prompt
    
    def _is_sufficiently_diverse(self, new_response: str, previous_responses: List[str]) -> bool:
        """
        Check if the new response is sufficiently diverse from previous ones
        Returns True if diverse enough, False otherwise
        """
        if not previous_responses:
            return True
        
        # Calculate similarity with each previous response
        similarities = []
        for prev_response in previous_responses:
            similarity = self._calculate_similarity(new_response, prev_response)
            similarities.append(similarity)
        
        # Get highest similarity score (worst case)
        max_similarity = max(similarities)
        
        # Check if it exceeds our threshold
        is_diverse = max_similarity < (1.0 - self.diversity_threshold)
        return is_diverse
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two text strings
        Returns value between 0.0 (completely different) and 1.0 (identical)
        """
        if self.use_external_similarity and self.similarity_model:
            return self._calculate_model_similarity(text1, text2)
        else:
            return self._calculate_text_similarity(text1, text2)
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate simple text similarity based on word overlap
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
    
    def _calculate_model_similarity(self, text1: str, text2: str) -> float:
        """
        Use the similarity model to calculate semantic similarity
        """
        try:
            prompt = [
                {"role": "system", "content": "You are a helpful AI assistant that evaluates text similarity."},
                {"role": "user", "content": f"""Please evaluate the similarity between these two texts on a scale from 0.0 (completely different) to 1.0 (identical).
                
Text 1:
{text1[:500]}...

Text 2:
{text2[:500]}...

Return only a number between 0.0 and 1.0 as your answer.
"""}
            ]
            
            response = self.similarity_model(prompt)
            
            # Extract float value from response
            import re
            float_match = re.search(r'([0-9]*\.?[0-9]+)', response)
            if float_match:
                similarity = float(float_match.group(1))
                # Ensure value is in valid range
                similarity = max(0.0, min(similarity, 1.0))
                return similarity
            
            return 0.5  # Default value if parsing fails
            
        except Exception as e:
            zwarn(f"[DSS] Error calculating model similarity: {e}")
            # Fallback to text similarity
            return self._calculate_text_similarity(text1, text2)
