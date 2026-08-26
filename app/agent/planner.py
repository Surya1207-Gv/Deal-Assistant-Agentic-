from __future__ import annotations
import os
import json
import hashlib
import time
import sys
import re
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Any
from dotenv import load_dotenv

from app.core.llm_gate import LLMGate

warnings.filterwarnings("ignore", message=".*automatic function calling.*")
warnings.filterwarnings("ignore", category=UserWarning, module="google.genai")

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_FILE = PROJECT_ROOT / "eval" / ".planner_cache.json"

class Planner:
    """
    LLM-driven Planner with candidate model fallback chain,
    loud exception reporting on fallback, intent reasoning,
    on-disk query caching (LLM plans ONLY), and exponential backoff retry logic.
    """

    CANDIDATE_MODELS = [
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash",
        "gemini-3.7-flash",
        "gemini-flash-latest"
    ]

    llm_calls: int = 0
    cache_hits: int = 0

    @classmethod
    def _load_cache(cls) -> Dict[str, Any]:
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    @classmethod
    def _save_cache(cls, cache: Dict[str, Any]):
        try:
            CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2)
        except Exception:
            pass

    @classmethod
    def is_llm_available(cls) -> bool:
        # Provider health is tracked process-wide by LLMGate, so a quota-exhausted or
        # offline provider is discovered ONCE rather than re-probed on every turn by every
        # call site. Planning then degrades to the deterministic tool order below.
        return LLMGate.is_available()

    @classmethod
    def plan_tools(cls, query: str, context: Dict[str, Any]) -> Tuple[List[str], str]:
        query_clean = (query or "").strip().lower()
        if not query_clean:
            return ["search_deals", "best_card"], "deterministic-fallback"

        # Check Cache first (LLM entries ONLY)
        q_hash = hashlib.md5(query_clean.encode("utf-8")).hexdigest()
        cache = cls._load_cache()
        if q_hash in cache and cache[q_hash].get("planner_mode") == "llm":
            cls.cache_hits += 1
            entry = cache[q_hash]
            return entry["planned_tools"], "llm"

        if not cls.is_llm_available():
            err_msg = LLMGate.last_error() or "LLM provider unavailable."
            if os.environ.get("STRICT_LLM_DEMO") == "true":
                raise RuntimeError(f"DEMO ABORTED: LLM unavailable — {err_msg}")
            print(f"WARNING: LLM unavailable, using fixed tool order. REASON: {err_msg}", file=sys.stderr)
            return cls._deterministic_plan(query_clean, context), "deterministic-fallback"

        try:
            prompt = (
                f"You are the Deal Assistant Planner.\n"
                f"Given the user request, return a JSON object with key 'planned_tools' containing\n"
                f"an ordered list subset of available tools: ['search_deals', 'compare_prices', 'best_card', 'get_reward_rules', 'watch_price'].\n\n"
                f"Tool Capabilities:\n"
                f"- 'compare_prices': Finds prices across merchants for any named product (e.g. Sony headphones, MacBook, Nike shoes, Swiggy meal, iPhone, electronics, groceries).\n"
                f"- 'search_deals': Finds merchant coupon discounts and promotional offers.\n"
                f"- 'best_card': Evaluates card cashback, reward rates, and determines the best card.\n"
                f"- 'get_reward_rules': Checks card terms, category caps, headroom, and reward policy limits.\n"
                f"- 'watch_price': Sets price drop alerts when user specifies a target trigger price.\n\n"
                f"Intent Planning Rules:\n"
                f"1. DEAL_DISCOVERY / DEAL_EXPLANATION (e.g. 'What deals are available at BigBasket?', 'What discounts does Amazon have?', 'Show me grocery offers', 'What does deal_032 offer?'): plan ['search_deals'] ONLY.\n"
                f"2. PRICE_LOOKUP (e.g. 'Compare prices for Sony WH-1000XM5', 'How much is iPhone 15 on Flipkart?'): plan ['compare_prices'] ONLY.\n"
                f"3. CARD_INFO / CARD_REWARD (e.g. 'Reward cap on HDFC Millennia groceries', 'Which card is best for ₹6,000 groceries?'): plan ['get_reward_rules'] or ['best_card', 'get_reward_rules'].\n"
                f"4. PRICE_WATCH (e.g. 'Track Sony WH-1000XM5 at ₹25000', 'Alert if price drops'): plan ['watch_price'] ONLY.\n"
                f"5. PRODUCT_OPTIMIZATION (e.g. 'Cheapest way to buy iPhone 15', 'Buy groceries worth ₹4000'): plan ['compare_prices', 'search_deals', 'best_card'].\n"
                f"6. MEMORY_UPDATE (e.g. 'I prefer HDFC Millennia', 'My budget is ₹3,300'): plan [].\n\n"
                f"User Request: {query}\n"
                f"Return JSON ONLY."
            )

            # One call through the shared gate: it owns the candidate-model chain and marks
            # the provider unhealthy on failure, so an outage costs one attempt per cooldown
            # window rather than one attempt per model per turn.
            cls.llm_calls += 1
            text = LLMGate.generate(prompt) or ""
            json_match = re.search(r"\{.*\}", text, re.DOTALL) if text else None
            if json_match:
                data = json.loads(json_match.group(0))
                tools = data.get("planned_tools", [])
                if isinstance(tools, list):
                    resolved = context.get("resolved_query") if context else None
                    if resolved:
                        op_allowed = {
                            "LIST": ["search_deals"],
                            "LOOKUP": ["compare_prices", "get_reward_rules", "search_deals"],
                            "AGGREGATE": ["search_deals", "compare_prices"],
                            "COMPUTE": ["best_card", "get_reward_rules", "search_deals"],
                            "OPTIMIZE": ["compare_prices", "search_deals", "best_card", "get_reward_rules"],
                            "COMPARE": ["compare_prices", "search_deals", "best_card", "get_reward_rules"],
                            "ELIGIBILITY": ["get_reward_rules", "search_deals"],
                            "WATCH": ["watch_price"],
                            "CLARIFY": [],
                            "EXPLAIN": [],
                            "STATE": [],
                            "ABSTAIN": []
                        }
                        allowed = op_allowed.get(resolved.operation.value, ["compare_prices", "search_deals", "best_card"])
                        tools = [t for t in tools if t in allowed]
                    cache[q_hash] = {"planned_tools": tools, "planner_mode": "llm"}
                    cls._save_cache(cache)
                    return tools, "llm"

            last_exception = LLMGate.last_error()
            if os.environ.get("STRICT_LLM_DEMO") == "true":
                raise RuntimeError(f"DEMO ABORTED: LLM unavailable — {last_exception}")
            print(f"WARNING: LLM unavailable, using fixed tool order. REASON: {last_exception}", file=sys.stderr)
        except Exception as e:
            if os.environ.get("STRICT_LLM_DEMO") == "true":
                raise RuntimeError(f"DEMO ABORTED: LLM unavailable — {e}")
            print(f"WARNING: LLM unavailable, using fixed tool order. REASON: {e}", file=sys.stderr)

        plan = cls._deterministic_plan(query_clean, context)
        return plan, "deterministic-fallback"

    # Default tool set per resolved operation. Consulted BEFORE the keyword rules below,
    # because the resolver has already interpreted the turn against the conversation
    # thread — re-guessing intent from the raw text here would let the planner disagree
    # with the operation actually being executed.
    OPERATION_PLANS = {
        "WATCH": ["watch_price"],
        "LIST": ["search_deals"],
        "AGGREGATE": ["search_deals"],
        "LOOKUP": ["compare_prices", "get_reward_rules"],
        "ELIGIBILITY": ["get_reward_rules", "search_deals"],
        "STATE": [],
        "CLARIFY": [],
        "EXPLAIN": [],
        "ABSTAIN": [],
    }

    @classmethod
    def _deterministic_plan(cls, q_lower: str, context: Dict[str, Any] = None) -> List[str]:
        resolved = (context or {}).get("resolved_query")
        if resolved is not None:
            planned = cls.OPERATION_PLANS.get(resolved.operation.value)
            if planned is not None:
                return list(planned)

        # Price Watch
        if any(w in q_lower for w in ["watch", "alert", "drops below", "hits", "track", "ping me"]):
            return ["watch_price"]

        # Memory / State Updates
        if any(w in q_lower for w in ["prefer", "rather use", "my budget is", "make that", "scratch that", "instead"]):
            if not any(w in q_lower for w in ["buy", "cheapest", "price", "order"]):
                return []

        # Price Lookup
        if any(w in q_lower for w in ["compare prices", "how much is", "price on"]) and not any(w in q_lower for w in ["buy", "cheapest", "deal", "discount", "offer"]):
            return ["compare_prices"]

        # Card Info / Eligibility
        if any(w in q_lower for w in ["reward cap", "category cap", "caps on", "base cashback rate", "reward rules", "can i use", "is it eligible", "does sbi cover", "does hdfc cover"]):
            return ["get_reward_rules"]

        # Deal Discovery / Deal Lookup
        if any(w in q_lower for w in ["what deals", "what grocery deals", "what discounts", "all deals", "available at", "offers at", "does blinkit have", "what does deal_", "details for deal_"]) and not any(w in q_lower for w in ["cheapest way", "buy", "purchase", "pay least"]):
            return ["search_deals"]

        # Product Optimization & Computation. The full chain the brief describes: find the
        # prices, find the offers, rank the cards, then read the winning card's reward rules
        # — the last of which also lets the answer be cross-checked against the card record.
        return ["compare_prices", "search_deals", "best_card", "get_reward_rules"]
