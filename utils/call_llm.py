"""
LLM Integration Module
Handles communication with Gemini API
"""

import os
import json
import time
from typing import Dict, Any, Optional
import google.generativeai as genai


class LLMClient:
    """Client for interacting with Gemini API"""
    
    def __init__(self, api_key: str = None, model: str = None):
        """Initialize LLM client"""
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model or "gemini-2.0-flash"
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not provided")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(self.model)
        
        # Rate limiting - increased to avoid API rate limits
        self.last_call_time = 0
        self.min_interval = 10.0  # Minimum seconds between calls (increased from 5.0 to handle rate limits better)
    
    def call_llm(self, prompt: str, max_retries: int = 3) -> str:
        """Call the LLM with rate limiting and retry logic"""
        # Rate limiting
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        
        if time_since_last_call < self.min_interval:
            time.sleep(self.min_interval - time_since_last_call)
        
        self.last_call_time = time.time()
        
        # Retry logic
        for attempt in range(max_retries):
            try:
                response = self.client.generate_content(prompt)
                
                # Check if response is valid
                if not response or not hasattr(response, 'text'):
                    if attempt < max_retries - 1:
                        print(f"    WARNING: Invalid response structure (attempt {attempt + 1}/{max_retries})")
                        time.sleep(2 ** attempt)
                        continue
                    return ""
                
                response_text = response.text
                
                # Check if response is empty
                if not response_text or not response_text.strip():
                    if attempt < max_retries - 1:
                        print(f"    WARNING: Empty response from LLM (attempt {attempt + 1}/{max_retries})")
                        time.sleep(2 ** attempt)
                        continue
                    return ""
                
                return response_text
                
            except Exception as e:
                error_str = str(e).lower()
                # Check for rate limit errors
                if 'rate limit' in error_str or '429' in error_str or 'quota' in error_str or 'resource_exhausted' in error_str:
                    wait_time = 30 * (attempt + 1)  # Much longer wait for rate limits (30s, 60s, 90s)
                    if attempt < max_retries - 1:
                        print(f"    ⚠️  RATE LIMIT HIT (attempt {attempt + 1}/{max_retries}), waiting {wait_time}s...")
                        print(f"    Error details: {str(e)[:200]}")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"    ❌ ERROR: Rate limit persists after {max_retries} attempts")
                        print(f"    Final error: {str(e)[:200]}")
                        return ""
                
                if attempt < max_retries - 1:
                    print(f"    WARNING: LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    # Return empty string instead of raising to allow graceful degradation
                    print(f"    ERROR: LLM call failed after {max_retries} attempts: {e}")
                    return ""
    
    def test_connection(self) -> bool:
        """Test LLM connection"""
        try:
            response = self.call_llm("Hello, this is a test.")
            return bool(response)
        except Exception:
            return False


# Global LLM client instance
_llm_client = None


def init_llm_from_config(config: Dict[str, Any]) -> None:
    """Initialize LLM client from config (api_key, model). Call before processing."""
    global _llm_client
    api_key = config.get('api_key') or os.getenv("GEMINI_API_KEY")
    model = config.get('model', "gemini-2.0-flash")
    _llm_client = LLMClient(api_key=api_key, model=model)


def get_llm_client(config: Dict[str, Any] = None) -> LLMClient:
    """Get global LLM client instance. Optionally init from config if provided."""
    global _llm_client
    if config is not None and _llm_client is None:
        init_llm_from_config(config)
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


def call_llm(prompt: str, max_retries: int = 3, config: Dict[str, Any] = None) -> str:
    """Convenience function to call LLM"""
    client = get_llm_client(config)
    return client.call_llm(prompt, max_retries)


def test_llm_connection() -> bool:
    """Test LLM connection"""
    try:
        client = get_llm_client()
        return client.test_connection()
    except Exception:
        return False


if __name__ == "__main__":
    """Test LLM connection"""
    print("Testing LLM connection...")
    
    if test_llm_connection():
        print("LLM connection successful")
        
        # Test a simple call
        try:
            response = call_llm("What is 2+2?")
            print(f"LLM response: {response[:100]}...")
        except Exception as e:
            print(f"ERROR: LLM call failed: {e}")
    else:
        print("ERROR: LLM connection failed")
        print("Please check your GEMINI_API_KEY environment variable")
