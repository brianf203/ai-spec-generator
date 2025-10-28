"""
LLM Integration Module
Handles communication with Gemini API using PocketFlow patterns
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
        
        # Rate limiting
        self.last_call_time = 0
        self.min_interval = 2.0  # Minimum seconds between calls
    
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
                return response.text
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"    WARNING: LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    raise e
    
    def test_connection(self) -> bool:
        """Test LLM connection"""
        try:
            response = self.call_llm("Hello, this is a test.")
            return bool(response)
        except Exception:
            return False


# Global LLM client instance
_llm_client = None


def get_llm_client() -> LLMClient:
    """Get global LLM client instance"""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


def call_llm(prompt: str, max_retries: int = 3) -> str:
    """Convenience function to call LLM"""
    client = get_llm_client()
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
