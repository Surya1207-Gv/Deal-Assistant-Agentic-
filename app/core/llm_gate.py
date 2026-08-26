from __future__ import annotations

"""
Single gate through which every LLM call in the application passes.

WHY THIS EXISTS
---------------
Two independent call sites (QueryResolver's operation classifier and the Planner) each
walked their own list of candidate models and each retried. When the provider is down or
the quota is exhausted, that meant ~9 failing network calls per user turn, several seconds
of latency, and — worse — no shared knowledge that the LLM was unavailable, so every turn
paid the same cost again.

The gate does three things:

  * Records provider health process-wide. After a failure the LLM is considered unavailable
    for COOLDOWN_SECONDS and callers skip it entirely instead of re-discovering the outage.
  * Owns the candidate-model chain so both call sites agree on it.
  * Makes "is the LLM usable right now?" a question the deterministic paths can ask cheaply.

FINANCIAL SAFETY INVARIANT
--------------------------
Nothing here (and nothing downstream of it) may produce a monetary quantity. The LLM is
used exclusively for INTERPRETATION — which operation the user is asking for, which tools
to plan. Every price, discount, reward, cap and ranking is computed by the deterministic
engine from dataset records. Degrading this gate to "unavailable" must therefore never
change an answer's arithmetic; it only removes an interpretation aid, which the structural
classifier in QueryResolver compensates for.
"""

import os
import time
from typing import Any, Callable, List, Optional

CANDIDATE_MODELS: List[str] = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
    "gemini-flash-latest",
]

# How long to stop attempting calls after a failure. Long enough that a quota-exhausted or
# offline run does not repeatedly pay the timeout, short enough that a transient blip
# self-heals within a normal session.
COOLDOWN_SECONDS = float(os.environ.get("LLM_COOLDOWN_SECONDS", "120"))


class LLMGate:
    _unavailable_until: float = 0.0
    _last_error: Optional[str] = None
    calls_attempted: int = 0
    calls_skipped: int = 0

    @classmethod
    def api_key(cls) -> Optional[str]:
        key = os.environ.get("GEMINI_API_KEY")
        if not key or len(key.strip()) < 10:
            return None
        return key.strip()

    @classmethod
    def is_available(cls) -> bool:
        if cls.api_key() is None:
            return False
        return time.time() >= cls._unavailable_until

    @classmethod
    def last_error(cls) -> Optional[str]:
        if cls.api_key() is None:
            return "GEMINI_API_KEY environment variable missing or invalid."
        return cls._last_error

    @classmethod
    def mark_failed(cls, error: Any) -> None:
        cls._last_error = str(error)
        cls._unavailable_until = time.time() + COOLDOWN_SECONDS

    @classmethod
    def mark_succeeded(cls) -> None:
        cls._last_error = None
        cls._unavailable_until = 0.0

    @classmethod
    def reset(cls) -> None:
        cls._unavailable_until = 0.0
        cls._last_error = None

    @classmethod
    def generate(cls, prompt: str) -> Optional[str]:
        """
        Run one text generation across the candidate-model chain, or return None when the
        LLM is unavailable. Callers MUST treat None as "no interpretation aid this turn"
        and fall back to their deterministic path — never as a reason to abstain or guess.
        """
        if not cls.is_available():
            cls.calls_skipped += 1
            return None

        key = cls.api_key()
        try:
            from google import genai
            client = genai.Client(api_key=key)
        except Exception as exc:  # provider SDK missing / misconfigured
            cls.mark_failed(exc)
            return None

        # Walk the whole chain: models carry independent quotas, so a failure on one says
        # nothing about the next. The provider is only marked unhealthy when every model
        # fails, and the cooldown then stops the chain being re-walked each turn.
        last_exc: Optional[Exception] = None
        for model_name in CANDIDATE_MODELS:
            try:
                cls.calls_attempted += 1
                resp = client.models.generate_content(model=model_name, contents=prompt)
                cls.mark_succeeded()
                return resp.text or ""
            except Exception as exc:
                last_exc = exc
                continue

        cls.mark_failed(last_exc)
        return None
