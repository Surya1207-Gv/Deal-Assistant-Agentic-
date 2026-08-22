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
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key or len(api_key.strip()) < 10:
            return False
        return True

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
            err_msg = "GEMINI_API_KEY environment variable missing or invalid."
            if os.environ.get("STRICT_LLM_DEMO") == "true":
                raise RuntimeError(f"DEMO ABORTED: LLM unavailable — {err_msg}")
            print(f"WARNING: LLM unavailable, using fixed tool order. REASON: {err_msg}", file=sys.stderr)
            return cls._deterministic_plan(query_clean), "deterministic-fallback"

        try:
            from google import genai
            api_key = os.environ.get("GEMINI_API_KEY")
            client = genai.Client(api_key=api_key)

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

            last_exception = None

            # Retry loop across candidate models
            for model_name in cls.CANDIDATE_MODELS:
                for retry in range(2):
                    try:
                        cls.llm_calls += 1
                        response = client.models.generate_content(
                            model=model_name,
                            contents=prompt,
                        )
                        text = response.text or ""
                        json_match = re.search(r"\{.*\}", text, re.DOTALL)
                        if json_match:
                            data = json.loads(json_match.group(0))
                            tools = data.get("planned_tools", [])
                            if isinstance(tools, list):
                                res_tools, res_mode = tools, "llm"
                                cache[q_hash] = {"planned_tools": res_tools, "planner_mode": "llm"}
                                cls._save_cache(cache)
                                return res_tools, res_mode
                        break
                    except Exception as e:
                        last_exception = e
                        break

            if os.environ.get("STRICT_LLM_DEMO") == "true":
                raise RuntimeError(f"DEMO ABORTED: LLM unavailable — {last_exception}")
            print(f"WARNING: LLM unavailable, using fixed tool order. REASON: {last_exception}", file=sys.stderr)
        except Exception as e:
            if os.environ.get("STRICT_LLM_DEMO") == "true":
                raise RuntimeError(f"DEMO ABORTED: LLM unavailable — {e}")
            print(f"WARNING: LLM unavailable, using fixed tool order. REASON: {e}", file=sys.stderr)

        plan = cls._deterministic_plan(query_clean)
        return plan, "deterministic-fallback"

    @classmethod
    def _deterministic_plan(cls, q_lower: str) -> List[str]:
        # Memory Updates
        if ("prefer" in q_lower or "use" in q_lower or "actually" in q_lower) and any(card in q_lower for card in ["hdfc millennia", "sbi cashback", "axis ace", "amex", "icici", "regalia", "millennia", "sbi", "axis"]):
            if not any(w in q_lower for w in ["buy", "cheapest", "price", "order", "what is"]):
                return []
        if "budget" in q_lower and ("is" in q_lower or "my budget" in q_lower or "make that" in q_lower or "₹" in q_lower or "rs" in q_lower):
            if not any(w in q_lower for w in ["buy", "cheapest", "price", "order", "what is"]):
                return []

        # Price Watch
        if "watch" in q_lower or "alert" in q_lower or "drops below" in q_lower or "track" in q_lower:
            return ["watch_price"]

        # Price Lookup
        if any(w in q_lower for w in ["compare prices for", "compare price for", "how much is", "price on"]) and not any(w in q_lower for w in ["buy", "cheapest", "deal", "discount", "offer"]):
            return ["compare_prices"]

        # Card Info
        if any(w in q_lower for w in ["reward cap", "category cap", "caps on", "base cashback rate", "reward rules", "does sbi cover", "does hdfc cover"]):
            return ["get_reward_rules"]

        # Deal Discovery / Deal Explanation
        if any(w in q_lower for w in ["what deals are available", "what grocery deals", "what discounts does", "show me", "does blinkit have", "any deal for", "deals that work with", "largest discount at", "what does", "deal details", "discount does", "is there a deal", "offers on", "offers at", "deals at", "deals on"]) and not any(w in q_lower for w in ["cheapest way", "buy", "purchase"]):
            return ["search_deals"]

        # Product Optimization
        return ["compare_prices", "search_deals", "best_card"]
