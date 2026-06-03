"""
LLM integration — Anthropic Claude API (Messages).
Default model: claude-sonnet-4-6
"""

import os
import time
from typing import Any, Dict, Optional

from anthropic import Anthropic
from anthropic import APIStatusError, APIError


class LLMClient:
    """Client for Anthropic Claude (Messages API)."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model or "claude-sonnet-4-6"
        self.max_tokens = int(os.getenv("CLAUDE_MAX_TOKENS", "32768"))
        self.temperature = float(os.getenv("CLAUDE_TEMPERATURE", "0.2"))

        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not provided")

        self.client = Anthropic(api_key=self.api_key)
        self.last_call_time = 0.0
        self.min_interval = float(os.getenv("CLAUDE_MIN_INTERVAL", "1.0"))

    def call_llm(
        self,
        prompt: str,
        max_retries: int = 3,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Call Claude with rate limiting and retries.

        ``max_tokens`` defaults to ``self.max_tokens``. Very large values can make
        the Anthropic Python SDK require streaming; for pings use a small override.
        """
        current_time = time.time()
        elapsed = current_time - self.last_call_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call_time = time.time()

        use_tokens = self.max_tokens if max_tokens is None else max_tokens
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": use_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        for attempt in range(max_retries):
            try:
                # Non-streaming `create()` rejects very large max_tokens on current SDK
                # ("Streaming is required for operations that may take longer than 10 minutes").
                with self.client.messages.stream(**kwargs) as stream:
                    text = stream.get_final_text().strip()
                if not text:
                    if attempt < max_retries - 1:
                        print(f"    WARNING: Empty response from LLM (attempt {attempt + 1}/{max_retries})")
                        time.sleep(2 ** attempt)
                        continue
                    return ""
                return text

            except APIStatusError as e:
                err = str(e).lower()
                status = getattr(e, "status_code", None)
                if status == 429 or "rate" in err or "overloaded" in err:
                    wait_time = 30 * (attempt + 1)
                    if attempt < max_retries - 1:
                        print(f"    WARNING: Rate limit / overload (attempt {attempt + 1}/{max_retries}), waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    print(f"    ERROR: Rate limit persists after {max_retries} attempts")
                    return ""
                if attempt < max_retries - 1:
                    print(f"    WARNING: LLM API error (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(2 ** attempt)
                    continue
                print(f"    ERROR: LLM call failed after {max_retries} attempts: {e}")
                return ""

            except APIError as e:
                if attempt < max_retries - 1:
                    print(f"    WARNING: LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(2 ** attempt)
                    continue
                print(f"    ERROR: LLM call failed after {max_retries} attempts: {e}")
                return ""

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"    WARNING: LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(2 ** attempt)
                    continue
                print(f"    ERROR: LLM call failed after {max_retries} attempts: {e}")
                return ""

        return ""

    def test_connection(self) -> bool:
        try:
            # Small max_tokens: SDK may require streaming for huge non-streaming budgets.
            response = self.call_llm("Reply with exactly: OK", max_tokens=256)
            return bool(response)
        except Exception:
            return False


_llm_client: Optional[LLMClient] = None


def init_llm_from_config(config: Dict[str, Any]) -> None:
    """Initialize LLM client from config (api_key, model). Call before processing."""
    global _llm_client
    api_key = config.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
    model = config.get("model", "claude-sonnet-4-6")
    _llm_client = LLMClient(api_key=api_key, model=model)


def get_llm_client(config: Dict[str, Any] = None) -> LLMClient:
    global _llm_client
    if config is not None and _llm_client is None:
        init_llm_from_config(config)
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


def call_llm(
    prompt: str,
    max_retries: int = 3,
    config: Dict[str, Any] = None,
    system: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> str:
    client = get_llm_client(config)
    return client.call_llm(prompt, max_retries, system=system, max_tokens=max_tokens)


def test_llm_connection() -> bool:
    try:
        client = get_llm_client()
        return client.test_connection()
    except Exception:
        return False


if __name__ == "__main__":
    print("Testing Claude API connection...")
    if test_llm_connection():
        print("LLM connection successful")
        try:
            r = call_llm("What is 2+2? One word.")
            print(f"Sample: {r[:120]}...")
        except Exception as e:
            print(f"ERROR: {e}")
    else:
        print("ERROR: LLM connection failed")
        print("Set ANTHROPIC_API_KEY")
