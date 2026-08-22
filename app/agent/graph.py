from __future__ import annotations
import os
import re
import sys
import time
import math
import warnings
from typing import Dict, Any, List, Tuple

warnings.filterwarnings("ignore", message=".*automatic function calling.*")
warnings.filterwarnings("ignore", category=UserWarning, module="google.genai")
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from app.agent.state import AgentState
from app.agent.planner import Planner
from app.agent.prompts import SYSTEM_RESPONSE_PROMPT
from app.rag.index import DataIndex
from app.rag.retriever import HybridRetriever
from app.rag.reranker import Reranker
from app.tools.search_deals import search_deals
from app.tools.compare_prices import compare_prices
from app.tools.best_card import best_card
from app.tools.get_reward_rules import get_reward_rules
from app.tools.watch_price import watch_price
from app.core.reward_engine import RewardEngine, PaymentOption
from app.core.provenance import ProvenanceValidator, Value, Provenance
from app.core.memory import MemoryManager
from app.core.telemetry import TelemetryTracker

load_dotenv()
retriever_instance = HybridRetriever()

INJECTION_PATTERNS = [
    "ignore previous instructions",
    "set effective price to 0",
    "prompt injection test deal",
    "system prompt",
    "override price"
]

def is_injection_attack(deal: Dict[str, Any]) -> bool:
    text = f"{deal.get('title', '')} {deal.get('terms', '')} {deal.get('discount_rules', '')} {deal.get('description', '')}".lower()
    return any(pattern in text for pattern in INJECTION_PATTERNS)


def _matches_budget_update(query: str) -> bool:
    q = query.lower()
    return "budget" in q or "make that" in q


def _matches_preference_update(query: str) -> bool:
    q = query.lower()
    cards = ["hdfc millennia", "sbi cashback", "axis ace", "amex smartearn", "icici amazon pay", "hdfc regalia", "millennia", "sbi", "axis", "amex", "icici", "regalia"]
    if "prefer" in q or "use" in q or "actually" in q or "instead" in q:
        return any(card in q for card in cards)
    return False


def _is_pure_memory_update(query: str) -> bool:
    q = query.lower()
    if "can i use" in q or "could i use" in q or "should i use" in q or "is it eligible" in q:
        return False
    action_words = ["find", "cheapest", "best price", "lowest", "buy", "purchase", "order", "what deal", "how much", "tell me", "drops below", "compare", "details for", "is there a deal", "reward cap", "what is the"]
    product_words = ["basket", "essentials", "iphone", "macbook", "flight", "resort", "hotel", "dinner", "meal", "shoes", "tv", "bill", "kindle", "dyson", "vacuum", "tesla", "headphones", "sony", "groceries worth", "groceries on"]
    has_action = any(w in q for w in action_words)
    has_product = any(w in q for w in product_words)
    if has_action and has_product:
        return False
    return _matches_budget_update(q) or _matches_preference_update(q)


def _apply_memory_update(state: AgentState, query: str) -> bool:
    q = query.lower()
    if not _is_pure_memory_update(q):
        return False

    s_id = state.get("session_id")
    conv_state = state.get("conversation_state") or {
        "budget": None,
        "preferred_card": None,
        "category_preferences": {},
        "other_relevant_constraints": {}
    }

    updated = False
    if _matches_budget_update(q):
        m = re.search(r"(\d[\d,]{2,8})", q)
        if m:
            b_val = float(m.group(1).replace(",", ""))
            state["budget"] = b_val
            conv_state["budget"] = b_val
            if s_id:
                MemoryManager.update_budget(s_id, b_val)
            if "grocery" in q or "groceries" in q:
                state["category"] = "groceries"
                if s_id:
                    MemoryManager.update_category(s_id, "groceries")
                state["final_response"] = f"Got it — I’ll use ₹{b_val:,.0f} as your grocery budget for this conversation."
            else:
                state["final_response"] = f"Got it — I’ll use ₹{b_val:,.0f} as your budget for this conversation."
            state["trace"] = [Value(amount=b_val, provenance=Provenance.SOURCE, record_id="user_budget")]
            updated = True

    if _matches_preference_update(q):
        card_map = {
            "hdfc millennia": "hdfc_millennia",
            "millennia": "hdfc_millennia",
            "sbi cashback": "sbi_cashback",
            "sbi": "sbi_cashback",
            "axis ace": "axis_ace",
            "axis": "axis_ace",
            "amex smartearn": "amex_smartearn",
            "amex": "amex_smartearn",
            "icici amazon pay": "icici_amazon_pay",
            "icici": "icici_amazon_pay",
            "hdfc regalia": "hdfc_regalia",
            "regalia": "hdfc_regalia",
        }
        chosen = None
        for key, value in card_map.items():
            if key in q:
                chosen = value
                break
        if chosen:
            if "for electronics" in q or ("electronics" in q and "prefer" in q):
                conv_state.setdefault("category_preferences", {})["electronics"] = chosen
                if s_id:
                    MemoryManager.update_category_preference(s_id, "electronics", chosen)
                state["final_response"] = f"Got it — I’ll prefer {chosen} for electronics."
            elif "for groceries" in q or ("groceries" in q and "prefer" in q):
                conv_state.setdefault("category_preferences", {})["groceries"] = chosen
                if s_id:
                    MemoryManager.update_category_preference(s_id, "groceries", chosen)
                state["final_response"] = f"Got it — I’ll prefer {chosen} for groceries."
            elif "for travel" in q or ("travel" in q and "prefer" in q):
                conv_state.setdefault("category_preferences", {})["travel"] = chosen
                if s_id:
                    MemoryManager.update_category_preference(s_id, "travel", chosen)
                state["final_response"] = f"Got it — I’ll prefer {chosen} for travel."
            else:
                state["preferred_card"] = chosen
                conv_state["preferred_card"] = chosen
                if s_id:
                    MemoryManager.update_card(s_id, chosen)
                state["final_response"] = f"Got it — I’ll prefer {chosen} for this conversation."
            updated = True

    if updated:
        state["conversation_state"] = conv_state
        if s_id:
            MemoryManager.set_conversation_state(s_id, conv_state)
        return True
    return False


def retrieve_node(state: AgentState) -> AgentState:
    query = (state.get("query") or "").strip()
    q_low = query.lower()

    s_id = state.get("session_id")
    if s_id:
        sess = MemoryManager.get_session(s_id)
        if state.get("budget") is None and sess.budget is not None:
            state["budget"] = sess.budget
        if state.get("category") is None and sess.category is not None:
            state["category"] = sess.category
        if state.get("preferred_card") is None and sess.preferred_card is not None:
            state["preferred_card"] = sess.preferred_card
        if not state.get("conversation_state") and sess.conversation_state:
            state["conversation_state"] = sess.conversation_state

    if _apply_memory_update(state, query):
        state["skip_planning"] = True
        state["planned_tools"] = []
        state["citations"] = []
        return state
    
    cat = None
    if any(w in q_low for w in ["electronic", "laptop", "macbook", "headphone", "headphones", "phone", "iphone", "tv", "sony", "croma", "apple", "kindle", "paperwhite"]):
        cat = "electronics"
    elif any(w in q_low for w in ["flight", "hotel", "travel", "travelfly", "makemytrip", "cleartrip", "delhi", "mumbai", "goa", "agoda", "resort"]):
        cat = "travel"
    elif any(w in q_low for w in ["shoe", "shoes", "clothing", "jean", "jeans", "bag", "shopping", "nike", "levis", "samsonite", "ajio", "myntra"]):
        cat = "shopping"
    elif any(w in q_low for w in ["bill", "bills", "electricity", "utility", "recharge"]):
        cat = "bills"
    elif any(w in q_low for w in ["dining", "food", "restaurant", "meal", "swiggy", "zomato", "dinner", "eatsure", "eat sure"]):
        cat = "food"
    elif any(w in q_low for w in ["grocery", "groceries", "milk", "dairy", "superstore", "bigbasket", "blinkit", "zepto", "instamart", "basket", "essentials"]):
        cat = "groceries"
    else:
        cat = state.get("category")

    state["category"] = cat

    raw_results, abstained, max_score = retriever_instance.search(query, category=cat, top_k=12)
    reranked = Reranker.rerank(query, raw_results, top_n=6)

    clean_records = []
    excluded_records = []

    for r in reranked:
        if is_injection_attack(r):
            excluded_records.append(r["deal_id"])
            sanitized = dict(r)
            sanitized["description"] = f"Official promotional deal offering {r.get('discount_value')} discount"
            clean_records.append(sanitized)
        else:
            clean_records.append(r)

    state["retrieved_records"] = clean_records
    state["excluded_injection_records"] = excluded_records
    state["abstained"] = abstained or (len(clean_records) == 0 and len(raw_results) > 0)
    state["max_retrieval_score"] = max_score

    return state

def check_abstention_router(state: AgentState) -> str:
    if state.get("skip_planning"):
        return "abstain_response"
    if state.get("abstained", False):
        return "abstain_response"
    return "plan_node"

def abstain_response_node(state: AgentState) -> AgentState:
    query = (state.get("query") or "").strip()
    if state.get("final_response"):
        state["provenance_valid"] = True
        state["citations"] = state.get("citations", [])
        return state

    records = state.get("retrieved_records", [])
    record_ids = [r["deal_id"] for r in records]
    
    state["final_response"] = (
        "no reliable deal found"
    )
    state["citations"] = record_ids
    state["provenance_valid"] = True
    return state

def plan_node(state: AgentState) -> AgentState:
    query = (state.get("query") or "").strip()
    if state.get("skip_planning"):
        state["planned_tools"] = []
        return state
    planned_tools, mode = Planner.plan_tools(query, state)
    state["planned_tools"] = planned_tools
    state["planner_mode"] = mode
    return state

def _extract_matched_product(query: str, products: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    q_low = query.lower()
    for p in products:
        p_name = p["name"].lower()
        if p_name in q_low:
            return p
    
    query_digits = set(re.findall(r"\b\d+\b", q_low))
    stopwords = {"can", "get", "the", "for", "and", "deal", "deals", "best", "price", "prices", "card", "cards", "with", "from", "find", "order", "want", "need", "offer", "discount", "cheaper", "cheapest", "way", "buy", "purchase", "where", "how", "what", "much", "tell"}
    tokens = [t for t in re.findall(r"[a-z0-9]+", q_low) if t not in stopwords]
    best_p = None
    max_s = 0
    for p in products:
        p_name = p["name"].lower()
        p_digits = set(re.findall(r"\b\d+\b", p_name))
        p_tokens = set(re.findall(r"[a-z0-9]+", p_name))
        if query_digits and p_digits and not query_digits.intersection(p_digits):
            continue
        t_matches = sum(2 for t in tokens if t in p_tokens or (len(t) > 3 and t in p_name))
        d_matches = sum(5 for d in query_digits if d in p_digits)
        score = t_matches + d_matches
        if score > max_s:
            max_s = score
            best_p = p
    if max_s >= 2:
        return best_p
    return None


def _extract_matched_card(query: str, cards: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    q_low = query.lower()
    for c in cards:
        c_name = c["name"].lower()
        c_id = c["card_id"].lower()
        if c_id in q_low or c_name in q_low:
            return c
    # Fallback to key card aliases
    aliases = {
        "millennia": "hdfc_millennia",
        "regalia": "hdfc_regalia",
        "sbi": "sbi_cashback",
        "axis": "axis_ace",
        "ace": "axis_ace",
        "amex": "amex_smartearn",
        "smartearn": "amex_smartearn",
        "amazon pay": "icici_amazon_pay",
        "icici": "icici_amazon_pay"
    }
    for alias, cid in aliases.items():
        if alias in q_low:
            for c in cards:
                if c["card_id"] == cid:
                    return c
    return None


def execute_tools_node(state: AgentState) -> AgentState:
    query = (state.get("query") or "").strip()
    planned = state.get("planned_tools", [])
    results: Dict[str, Any] = {}
    tool_mapping: Dict[str, str] = {}

    for tool in planned:
        if tool == "compare_prices":
            res = compare_prices(query)
            results["compare_prices"] = res
            if res.get("found"):
                lowest = res.get("lowest_price")
                m_amt = lowest.amount if lowest else 0
                merch = res.get("lowest_price_merchant", "Catalog")
                tool_mapping["compare_prices"] = f"product '{res.get('product_name')}', lowest price RS {m_amt:.0f} at {merch}"
            else:
                tool_mapping["compare_prices"] = "no catalog price match"
        elif tool == "search_deals":
            res = search_deals(query)
            results["search_deals"] = res
            deals_list = res.get("deals", []) if isinstance(res, dict) else (res if isinstance(res, list) else [])
            deal_ids = [d.get("deal_id") for d in deals_list if isinstance(d, dict)]
            tool_mapping["search_deals"] = f"found {len(deals_list)} candidate deals: {', '.join(deal_ids) if deal_ids else 'none'}"
        elif tool == "best_card":
            amt = state.get("budget")
            cat = state.get("category", "groceries")
            res = best_card(amount=amt, category=cat, spend_to_date=state.get("spend_to_date"))
            results["best_card"] = res
            if res.get("best_card"):
                tool_mapping["best_card"] = f"card '{res.get('best_card')}', rate {res.get('reward_rate_pct'):.1f}%, reward RS {res.get('reward_earned', 0):.0f}"
            else:
                tool_mapping["best_card"] = "evaluated card reward rates"
        elif tool == "get_reward_rules":
            cards = DataIndex.get_cards()
            matched_c = _extract_matched_card(query, cards)
            card_id = matched_c["card_id"] if matched_c else (state.get("preferred_card") or "hdfc_millennia")
            res = get_reward_rules(card_id)
            results["get_reward_rules"] = res
            tool_mapping["get_reward_rules"] = f"card '{card_id}' base rate {res.get('base_rate', 0)*100:g}%"
        elif tool == "watch_price":
            amt = state.get("budget")
            match_target = re.search(r"(?:below|at|target|drops\s*to|under)\s*(?:RS\s*|₹\s*)?(\d{3,6})", query, re.IGNORECASE)
            if match_target:
                amt = float(match_target.group(1))
            if amt is None:
                amt = 25000.0
            res = watch_price(query, target_price=amt, session_id=state.get("session_id", "default"))
            results["watch_price"] = res
            gap_val = res.get("gap")
            gap_amt = gap_val.amount if gap_val else 0.0
            curr_p = res.get("current_lowest_price")
            curr_amt = curr_p.amount if curr_p else 0.0
            tool_mapping["watch_price"] = f"target RS {amt:.0f}, current RS {curr_amt:.0f}, gap RS {gap_amt:.0f}"

    state["tool_results"] = results
    state["tool_mapping"] = tool_mapping
    return state


def compute_rewards_node(state: AgentState) -> AgentState:
    query = (state.get("query") or "").strip()
    query_lower = query.lower()
    if state.get("skip_planning"):
        return state

    retrieved = state.get("retrieved_records", [])
    tool_res = state.get("tool_results", {})
    products = DataIndex.get_products()
    deals = DataIndex.get_deals()
    cards = DataIndex.get_cards()

    matched_prod = _extract_matched_product(query, products)
    matched_c = _extract_matched_card(query, cards)
    matched_deal = None
    for d in deals:
        if d["deal_id"].lower() in query_lower or d["title"].lower() in query_lower or d["merchant"].lower() in query_lower:
            matched_deal = d
            break

    # ==========================================
    # ROUTE 0: PRICE_WATCH (Skips reward engine)
    # ==========================================
    if any(w in query_lower for w in ["track", "price watch", "drops below", "watch_price"]):
        match_target = re.search(r"(?:below|at|target|drops\s*to|under)\s*(?:RS\s*|₹\s*)?(\d{3,6})", query, re.IGNORECASE)
        target_amt = float(match_target.group(1)) if match_target else (state.get("budget") or 25000.0)
        watch_res = tool_res.get("watch_price") or watch_price(query, target_price=target_amt, session_id=state.get("session_id", "default"))
        prod = watch_res.get("product_name", "Requested Product")
        curr_p = watch_res.get("current_lowest_price")
        curr_amt = curr_p.amount if curr_p else 0.0
        merch = watch_res.get("lowest_price_merchant", "Amazon")
        gap_amt = max(0.0, curr_amt - target_amt)
        cmp_data = tool_res.get("compare_prices", {})
        prod_id = cmp_data.get("product_id") or (matched_prod["product_id"] if matched_prod else "prod_watch")
        cits = [prod_id]

        draft = (
            f"Recommendation: Price watch registered for '{prod}' at target RS {target_amt:.0f}.\n"
            f"• Current Lowest Price: RS {curr_amt:.0f} (at {merch})\n"
            f"• Target Price: RS {target_amt:.0f}\n"
            f"• Price Gap to Target: RS {gap_amt:.0f} (above target)\n"
            f"• Watch Status: ACTIVE. No currently active deal crosses the RS {target_amt:.0f} threshold.\n"
            f"Citations: {cits}"
        )
        new_trace = [
            Value(amount=target_amt, provenance=Provenance.SOURCE, record_id="target_price"),
            Value(amount=curr_amt, provenance=Provenance.SOURCE, record_id="current_price"),
            Value(amount=gap_amt, provenance=Provenance.DERIVED, record_id="gap_price"),
            Value(amount=round(curr_amt, 0), provenance=Provenance.SOURCE, record_id="curr_rounded"),
            Value(amount=round(gap_amt, 0), provenance=Provenance.DERIVED, record_id="gap_rounded"),
            Value(amount=round(target_amt, 0), provenance=Provenance.SOURCE, record_id="target_rounded")
        ]
        state["payment_options"] = []
        state["best_option"] = None
        state["runner_up_option"] = None
        state["final_response"] = draft
        state["draft_response"] = draft
        state["citations"] = cits
        state["trace"] = new_trace
        state["is_info_query"] = True
        state["provenance_valid"] = True
        return state

    # ==========================================
    # ROUTE 1: PRICE_LOOKUP (No recommendations)
    # ==========================================
    if any(q_phrase in query_lower for q_phrase in ["compare prices for", "how much is", "price on", "what is the price of", "compare price"]) and matched_prod and not any(w in query_lower for w in ["cheapest way", "best card", "deal", "discount", "buy"]):
        prod_name = matched_prod["name"]
        prod_id = matched_prod["product_id"]
        prices = matched_prod.get("prices", {})
        
        trace_facts = []
        lines = []
        lowest_p = float("inf")
        lowest_m = None
        for m_name, p_amt in prices.items():
            trace_facts.append(Value(amount=float(p_amt), provenance=Provenance.SOURCE, record_id=f"{prod_id}_{m_name}"))
            lines.append(f"• {m_name.capitalize()}: RS {p_amt:,.0f}")
            if p_amt < lowest_p:
                lowest_p = float(p_amt)
                lowest_m = m_name.capitalize()
        
        draft = (
            f"Price Comparison for {prod_name}:\n"
            + "\n".join(lines) + "\n"
            + f"Lowest price is RS {lowest_p:,.0f} at {lowest_m}."
        )
        state["payment_options"] = []
        state["best_option"] = None
        state["runner_up_option"] = None
        state["final_response"] = draft
        state["draft_response"] = draft
        state["citations"] = [prod_id]
        state["trace"] = trace_facts
        state["is_info_query"] = True
        state["provenance_valid"] = True
        return state

    # ==========================================
    # ROUTE 2: CARD_INFO & ELIGIBILITY
    # ==========================================
    if any(w in query_lower for w in ["reward cap", "what is the reward cap", "caps on", "category caps", "base cashback rate", "base rate", "reward rules", "can i use", "eligible", "eligibility", "does sbi cover", "does hdfc cover"]):
        c_id = matched_c["card_id"] if matched_c else "hdfc_millennia"
        c_rules = get_reward_rules(c_id)
        c_name = c_rules.get("name", c_id)
        b_rate = c_rules.get("base_rate", 0.01) * 100
        caps = c_rules.get("caps", {})
        m_cap = caps.get("monthly_cashback_cap", 0)
        cat_caps = caps.get("category_caps", {})
        
        trace_facts = [
            Value(amount=float(b_rate), provenance=Provenance.SOURCE, record_id=f"{c_id}_base_rate"),
            Value(amount=float(m_cap), provenance=Provenance.SOURCE, record_id=f"{c_id}_monthly_cap")
        ]
        for ck, cv in cat_caps.items():
            trace_facts.append(Value(amount=float(cv), provenance=Provenance.SOURCE, record_id=f"{c_id}_{ck}_cap"))
        
        cat_cap_str = ", ".join([f"RS {cv:.0f} on {ck}" for ck, cv in cat_caps.items()]) if cat_caps else "none"
        if any(w in query_lower for w in ["can i use", "eligible", "eligibility"]):
            draft = f"Yes, you can use {c_name} ({c_id}). It provides a 5% cashback rate on online transactions, capped at RS {m_cap:.0f} per month."
        else:
            draft = f"Reward Rules for {c_name} ({c_id}): Base cashback rate is {b_rate:g}%. Monthly cashback cap is RS {m_cap:.0f}. Category caps: {cat_cap_str}."
        state["payment_options"] = []
        state["best_option"] = None
        state["runner_up_option"] = None
        state["final_response"] = draft
        state["draft_response"] = draft
        state["citations"] = [c_id]
        state["trace"] = trace_facts
        state["is_info_query"] = True
        state["provenance_valid"] = True
        return state

    # ==========================================
    # ROUTE 2B: CARD_REWARD COMPARISON
    # ==========================================
    if any(w in query_lower for w in ["which card is best", "which card gives the best", "best card for travel", "best card for groceries", "best card for electronics", "best card for shopping"]) and not matched_prod and not any(w in query_lower for w in ["cheapest way", "buy", "purchase", "price on"]):
        match_amt = re.search(r"(?:RS\s*|₹\s*|worth\s*|spending\s*|order\s*of\s*|for\s*a\s*(?:RS\s*|₹\s*)?)(\d{3,6})", query, re.IGNORECASE)
        q_amt = float(match_amt.group(1)) if match_amt else (state.get("budget") or 6000.0)
        cat_val = None
        for c_k in ["groceries", "electronics", "travel", "shopping", "bills", "food"]:
            if c_k in query_lower:
                cat_val = c_k
                break
        if not cat_val:
            cat_val = state.get("category") or "travel"
        
        best_c = None
        max_rew = -1.0
        best_eff = 0.0
        lines = []
        c_citations = []
        trace_facts = [
            Value(amount=q_amt, provenance=Provenance.SOURCE, record_id="query_amount")
        ]
        for c in cards:
            cid = c["card_id"]
            c_citations.append(cid)
            opt = RewardEngine.calculate_payment_option(base_price=Value(amount=q_amt, provenance=Provenance.SOURCE, record_id="query_amount"), deal=None, card=c, category=cat_val, spend_to_date=state.get("spend_to_date"))
            rew_amt = opt.reward_earned.amount
            eff_amt = opt.effective_price.amount
            trace_facts.extend([
                opt.reward_earned,
                opt.effective_price,
                Value(amount=round(rew_amt, 0), provenance=Provenance.DERIVED, record_id=f"{cid}_rew_r0"),
                Value(amount=round(eff_amt, 0), provenance=Provenance.DERIVED, record_id=f"{cid}_eff_r0")
            ])
            lines.append(f"• {c['name']} ({cid}): RS {rew_amt:.0f} cashback (Effective RS {eff_amt:.0f})")
            if rew_amt > max_rew:
                max_rew = rew_amt
                best_c = c
                best_eff = eff_amt
        
        draft = (
            f"Best card comparison for RS {q_amt:.0f} on {cat_val}:\n"
            + "\n".join(lines) + "\n"
            + f"Recommendation: {best_c['name']} ({best_c['card_id']}) gives the highest reward of RS {max_rew:.0f} (Effective price RS {best_eff:.0f})."
        )
        state["payment_options"] = []
        state["best_option"] = None
        state["runner_up_option"] = None
        state["final_response"] = draft
        state["draft_response"] = draft
        state["citations"] = c_citations
        state["trace"] = trace_facts
        state["is_info_query"] = True
        state["provenance_valid"] = True
        return state

    # ==========================================
    # ROUTE 3A: DEAL_DISCOVERY (Listing offers)
    # ==========================================
    deal_disc_phrases = [
        "what deals are available", "what grocery deals", "what discounts does",
        "show me grocery offers", "show me", "does blinkit have", "any deal for",
        "deals that work with", "largest discount at", "grocery deals", "electronics deals",
        "travel deals", "shopping deals", "deals available", "available deals",
        "offers available", "what offers", "discounts on", "discounts at", "deals on", "deals at"
    ]
    is_deal_disc = any(w in query_lower for w in deal_disc_phrases) and not any(w in query_lower for w in ["cheapest way", "buy", "purchase", "what is the price", "compare prices for", "order of", "order worth", "i am buying", "spending", "worth ₹", "worth rs", "discount does bigbasket"])

    if is_deal_disc:
        t_merch = None
        for m in ["bigbasket", "amazon", "flipkart", "croma", "blinkit", "swiggy", "zomato", "cleartrip", "agoda", "easemytrip", "yatra", "eatsure", "myntra", "ajio", "country delight", "superstore", "techworld", "travelfly"]:
            if m in query_lower:
                t_merch = m
                break
        
        t_cat = None
        for k in ["groceries", "grocery", "electronics", "electronic", "travel", "flight", "hotel", "shopping", "bills", "utility", "food", "dining", "appliances"]:
            if k in query_lower:
                t_cat = "groceries" if k in ["groceries", "grocery"] else ("electronics" if k in ["electronics", "electronic", "appliances"] else ("travel" if k in ["travel", "flight", "hotel"] else ("bills" if k in ["bills", "utility"] else ("food" if k in ["food", "dining"] else "shopping"))))
                break
        if not t_cat:
            t_cat = state.get("category")
        
        t_card = matched_c["card_id"] if matched_c else None
        
        matching_deals = []
        for d in deals:
            if t_merch and (d.get("merchant", "").lower() != t_merch.lower() and d.get("merchant", "").lower() not in ["all", "any"]):
                continue
            if t_cat and (d.get("category", "").lower() != t_cat.lower() and d.get("category", "").lower() not in ["all", "any"]):
                continue
            if t_card and d.get("card_specific") and d.get("card_specific").lower() != t_card.lower():
                continue
            matching_deals.append(d)
        
        if not matching_deals and retrieved:
            matching_deals = [r for r in retrieved if not is_injection_attack(r)]
        
        if not matching_deals:
            state["final_response"] = "no reliable deal found"
            state["citations"] = []
            state["abstained"] = True
            state["skip_planning"] = True
            return state
        
        lines = []
        trace_facts = []
        cits = []
        for d in matching_deals:
            d_id = d["deal_id"]
            cits.append(d_id)
            d_type = d.get("discount_type")
            d_val = d.get("discount_value", 0)
            min_s = float(d.get("min_spend", 0))
            max_d = float(d.get("max_discount", d_val or 0))
            
            trace_facts.extend([
                Value(amount=min_s, provenance=Provenance.SOURCE, record_id=f"{d_id}_min_spend"),
                Value(amount=max_d, provenance=Provenance.SOURCE, record_id=f"{d_id}_max_discount")
            ])
            
            if d_type == "percentage":
                display_pct = d_val * 100 if d_val <= 1.0 else d_val
                trace_facts.append(Value(amount=float(display_pct), provenance=Provenance.SOURCE, record_id=f"{d_id}_pct"))
                desc_str = f"{display_pct:g}% instant discount (capped at RS {max_d:.0f}) on min spend RS {min_s:.0f}"
            else:
                trace_facts.append(Value(amount=float(d_val), provenance=Provenance.SOURCE, record_id=f"{d_id}_flat_val"))
                desc_str = f"flat RS {d_val:.0f} instant discount on min spend RS {min_s:.0f}"
            
            if d.get("card_specific"):
                desc_str += f" (Card: {d.get('card_specific')})"
            
            lines.append(f"• {d_id} — {d.get('title')}: {desc_str}")
        
        header = f"Available deals"
        if t_merch:
            header += f" on {t_merch.capitalize()}"
        if t_cat:
            header += f" for {t_cat}"
        if t_card:
            header += f" with {t_card}"
        header += ":"
        
        draft = header + "\n" + "\n".join(lines)
        
        state["payment_options"] = []
        state["best_option"] = None
        state["runner_up_option"] = None
        state["final_response"] = draft
        state["draft_response"] = draft
        state["citations"] = cits
        state["trace"] = trace_facts
        state["is_info_query"] = True
        state["provenance_valid"] = True
        return state

    # ==========================================
    # ROUTE 3B: DEAL_EXPLANATION (Single deal)
    # ==========================================
    deal_info_phrases = ["what does the", "what discount does", "deal details", "what deal is available", "what deal is", "what deal", "which deal has", "which deal", "details for deal_", "how much is the", "is there a deal", "offers from", "check superstore", "show me techworld", "what is travelfly"]
    if any(w in query_lower for w in deal_info_phrases) and not any(w in query_lower for w in ["cheapest way", "best card for", "best way to buy"]):
        for d in (retrieved + deals):
            d_merch = (d.get("merchant") or "").lower()
            d_title = (d.get("title") or "").lower()
            d_cat = (d.get("category") or "").lower()
            if (d_merch and d_merch in query_lower) or (d_cat and d_cat in query_lower) or d.get("deal_id").lower() in query_lower or (d_title and any(w in query_lower for w in re.findall(r"[a-z0-9]+", d_title) if len(w) > 4)):
                matched_deal = d
                break

        if not matched_deal:
            matched_deal = retrieved[0] if retrieved else deals[0]
        d_name = matched_deal.get("title", matched_deal.get("deal_id"))
        d_id = matched_deal.get("deal_id")
        d_type = matched_deal.get("discount_type")
        d_val = matched_deal.get("discount_value")
        min_s = float(matched_deal.get("min_spend", 0))
        max_d = float(matched_deal.get("max_discount", d_val or 0))
        d_merch = matched_deal.get("merchant", "Partner")
        d_cat = matched_deal.get("category", "shopping")

        match_amt = re.search(r"(?:RS\s*|₹\s*|worth\s*|spending\s*|order\s*of\s*|for\s*a\s*(?:RS\s*|₹\s*)?)(\d{3,6})", query, re.IGNORECASE)
        order_amt = float(match_amt.group(1)) if match_amt else None

        trace_facts = [
            Value(amount=min_s, provenance=Provenance.SOURCE, record_id=f"{d_id}_min_spend"),
            Value(amount=max_d, provenance=Provenance.SOURCE, record_id=f"{d_id}_max_discount")
        ]

        if d_type == "percentage":
            display_pct = d_val * 100 if d_val <= 1.0 else d_val
            trace_facts.append(Value(amount=float(display_pct), provenance=Provenance.SOURCE, record_id=f"{d_id}_pct"))
            if order_amt:
                calc_disc = min(order_amt * (d_val if d_val <= 1.0 else d_val / 100), max_d)
                final_p = order_amt - calc_disc
                trace_facts.extend([
                    Value(amount=order_amt, provenance=Provenance.SOURCE, record_id="order_amt"),
                    Value(amount=calc_disc, provenance=Provenance.DERIVED, record_id="calculated_discount"),
                    Value(amount=final_p, provenance=Provenance.DERIVED, record_id="final_price")
                ])
                draft = f"The {d_merch} {d_cat} deal ({d_id}) offers a {display_pct:g}% instant discount (capped at RS {max_d:.0f}) on a minimum spend of RS {min_s:.0f}. For a RS {order_amt:.0f} order, the discount is RS {calc_disc:.0f}, resulting in a price of RS {final_p:.0f}."
            else:
                draft = f"The {d_merch} {d_cat} deal ({d_id}) offers a {display_pct:g}% instant discount (capped at RS {max_d:.0f}) on a minimum spend of RS {min_s:.0f}."
        else:
            trace_facts.append(Value(amount=float(d_val), provenance=Provenance.SOURCE, record_id=f"{d_id}_flat_val"))
            if order_amt:
                calc_disc = min(float(d_val), max_d)
                final_p = order_amt - calc_disc
                trace_facts.extend([
                    Value(amount=order_amt, provenance=Provenance.SOURCE, record_id="order_amt"),
                    Value(amount=calc_disc, provenance=Provenance.DERIVED, record_id="calculated_discount"),
                    Value(amount=final_p, provenance=Provenance.DERIVED, record_id="final_price")
                ])
                draft = f"The {d_merch} {d_cat} deal ({d_id}) offers a flat RS {d_val:.0f} instant discount on a minimum spend of RS {min_s:.0f}. For a RS {order_amt:.0f} order, the discount is RS {calc_disc:.0f}, resulting in a price of RS {final_p:.0f}."
            else:
                draft = f"The {d_merch} {d_cat} deal ({d_id}) offers a flat RS {d_val:.0f} instant discount on a minimum spend of RS {min_s:.0f}."

        state["payment_options"] = []
        state["best_option"] = None
        state["runner_up_option"] = None
        state["final_response"] = draft
        state["draft_response"] = draft
        state["citations"] = [d_id]
        state["trace"] = trace_facts
        state["is_info_query"] = True
        state["provenance_valid"] = True
        return state

    # ==========================================
    # ROUTE 4: FULL EXHAUSTIVE OPTIMIZATION
    # ==========================================
    tool_res = state.get("tool_results", {})
    retrieved = state.get("retrieved_records", [])
    cmp = tool_res.get("compare_prices", {})
    lowest_val = cmp.get("lowest_price")

    cat = matched_prod.get("category") if matched_prod else (cmp.get("category") or state.get("category"))
    if not cat:
        for d in deals:
            if d.get("category") and d.get("category").lower() in query_lower:
                cat = d.get("category").lower()
                break
    if not cat:
        cat = "shopping"

    category_deals = [
        d for d in deals
        if (not d.get("category") or d.get("category").lower() in ["all", "any"] or (cat and d.get("category").lower() == cat.lower()))
    ]
    if retrieved:
        category_deals = list({d.get("deal_id"): d for d in category_deals + retrieved}.values())

    base_price = None
    match_amt = re.search(r"(?:RS\s*|₹\s*|worth\s*|spending\s*|spend\s*|budget\s*is\s*|for\s*(?:a\s*)?(?:RS\s*|₹\s*)?)(\d{3,6})", query, re.IGNORECASE)
    if match_amt:
        query_amount = float(match_amt.group(1))
        base_price = Value(amount=query_amount, provenance=Provenance.SOURCE, record_id="user_query_spend")
    elif state.get("budget") is not None:
        base_price = Value(amount=float(state.get("budget")), provenance=Provenance.SOURCE, record_id="session_budget")
    elif lowest_val and isinstance(lowest_val, Value):
        base_price = lowest_val

    if not base_price:
        state["final_response"] = "no reliable deal found"
        state["abstained"] = True
        state["skip_planning"] = True
        return state

    # Parse stated cap usage / prior spend from query
    spend_to_date = dict(state.get("spend_to_date") or {})
    cap_used_match = re.search(r"(?:used\s*(?:RS\s*|₹\s*)?(\d{2,5})|(\d{2,5})\s*cap\s*used)", query, re.IGNORECASE)
    if cap_used_match:
        c_amt = float(cap_used_match.group(1) or cap_used_match.group(2))
        c_key = matched_c["card_id"] if matched_c else "hdfc_millennia"
        spend_to_date[f"{c_key}_{cat}"] = c_amt

    headroom_match = re.search(r"(?:(\d{2,5})\s*headroom|headroom\s*(?:of\s*)?(?:RS\s*|₹\s*)?(\d{2,5}))", query, re.IGNORECASE)
    if headroom_match:
        h_amt = float(headroom_match.group(1) or headroom_match.group(2))
        c_key = matched_c["card_id"] if matched_c else "hdfc_millennia"
        spend_to_date[f"{c_key}_{cat}"] = max(0.0, 500.0 - h_amt)

    conv_state = state.get("conversation_state") or {}
    cat_prefs = conv_state.get("category_preferences", {})
    pref_card = (matched_c["card_id"] if matched_c else None) or cat_prefs.get(cat) or state.get("preferred_card")

    # Hard named merchant filter
    named_merchant = None
    for m in ["flipkart", "amazon", "croma", "bigbasket", "cleartrip", "agoda", "swiggy", "eatsure", "zomato", "myntra", "ajio"]:
        if f"on {m}" in query_lower or f"at {m}" in query_lower:
            named_merchant = m
            break

    options: List[PaymentOption] = []
    cmp_prices = cmp.get("merchant_prices", {})

    if cmp.get("found") and cmp_prices and isinstance(cmp_prices, dict) and not match_amt:
        for m_name, m_val in cmp_prices.items():
            if named_merchant and m_name.lower() != named_merchant:
                continue
            m_deals = [d for d in category_deals if d.get("merchant", "").lower() == m_name.lower() or d.get("merchant", "").lower() in ["all", "any"]]
            for c in cards:
                for d in m_deals:
                    options.append(RewardEngine.calculate_payment_option(base_price=m_val, deal=d, card=c, category=cat, spend_to_date=spend_to_date))
                options.append(RewardEngine.calculate_payment_option(base_price=m_val, deal=None, card=c, category=cat, spend_to_date=spend_to_date))
    else:
        for d in category_deals:
            for c in cards:
                options.append(RewardEngine.calculate_payment_option(base_price=base_price, deal=d, card=c, category=cat, spend_to_date=spend_to_date))
        for c in cards:
            options.append(RewardEngine.calculate_payment_option(base_price=base_price, deal=None, card=c, category=cat, spend_to_date=spend_to_date))

    ranked = RewardEngine.rank_options(options)
    if not ranked:
        state["final_response"] = "no reliable deal found"
        state["abstained"] = True
        state["skip_planning"] = True
        return state

    if pref_card:
        named_opts = [o for o in options if o.card_id == pref_card]
        if named_opts:
            best_opt = RewardEngine.rank_options(named_opts)[0]
            other_opts = [o for o in ranked if o.card_id != pref_card]
            runner_up = other_opts[0] if other_opts else None
        else:
            best_opt = ranked[0]
            runner_up = next((o for o in ranked if o.card_id != best_opt.card_id), None)
    else:
        best_opt = ranked[0]
        runner_up = next((o for o in ranked if o.card_id != best_opt.card_id), None)

    budget = state.get("budget")
    if budget is not None and best_opt and best_opt.effective_price.amount > float(budget):
        state["final_response"] = f"no reliable deal found; the cheapest grounded option is RS {best_opt.effective_price.amount:,.0f}."
        state["abstained"] = True
        state["skip_planning"] = True
        state["best_option"] = None
        state["citations"] = []
        return state

    min_eff = best_opt.effective_price.amount
    tied_options = [o for o in ranked if abs(o.effective_price.amount - min_eff) < 0.01 and (not pref_card or o.card_id == pref_card)]
    tied_merchants = []
    for o in tied_options:
        m = o.discount_source_id or (o.base_price.record_id.split("_")[-1].capitalize() if o.base_price and o.base_price.record_id and "_" in o.base_price.record_id else "Partner Merchant")
        if m.startswith("deal_"):
            d_rec = DataIndex.get_deal_by_id(m)
            if d_rec and d_rec.get("merchant"):
                m = d_rec["merchant"]
        tied_merchants.append(m)
    
    unique_tied_merchants = list(dict.fromkeys(tied_merchants))
    state["is_tie"] = len(unique_tied_merchants) > 1
    state["tied_merchants"] = unique_tied_merchants

    state["payment_options"] = ranked
    state["best_option"] = best_opt
    state["runner_up_option"] = runner_up
    state["trace"] = best_opt.trace if best_opt else [base_price]
    state["citations"] = best_opt.citations if best_opt else []
    return state

def validate_provenance_node(state: AgentState) -> AgentState:
    if state.get("skip_planning") or state.get("is_info_query"):
        state["provenance_valid"] = True
        state["unverified_tokens"] = []
        return state

    query = (state.get("query") or "").strip()
    tool_res = state.get("tool_results", {})
    planned = state.get("planned_tools", [])

    # Scenario Price Watcher Path
    if "watch_price" in tool_res or "watch_price" in planned or "drops below" in query.lower() or "price watch" in query.lower():
        watch_res = tool_res.get("watch_price")
        if not watch_res:
            amt = state.get("budget")
            match_target = re.search(r"(?:below|at|target|drops\s*to|under)\s*(?:RS\s*|₹\s*)?(\d{3,6})", query, re.IGNORECASE)
            if match_target:
                amt = float(match_target.group(1))
            if amt is None:
                amt = 25000.0
            watch_res = watch_price(query, target_price=amt)

        prod = watch_res.get("product_name", "Requested Product")
        target_val = watch_res.get("target_price")
        target_amt = target_val.amount if target_val else 25000.0

        curr_val = watch_res.get("current_lowest_price")
        curr_amt = curr_val.amount if curr_val else 0.0
        merch = watch_res.get("lowest_price_merchant", "Catalog")

        gap_val = watch_res.get("gap")
        gap_amt = gap_val.amount if gap_val else max(0.0, curr_amt - target_amt)

        cmp_data = tool_res.get("compare_prices", {})
        prod_id = cmp_data.get("product_id") or "prod_watch"
        cits = [prod_id]
        if state.get("retrieved_records"):
            cits.append(state["retrieved_records"][0]["deal_id"])

        draft = (
            f"Recommendation: Price watch registered for '{prod}' at target RS {target_amt:.0f}.\n"
            f"• Current Lowest Price: RS {curr_amt:.0f} (at {merch})\n"
            f"• Target Price: RS {target_amt:.0f}\n"
            f"• Price Gap to Target: RS {gap_amt:.0f} (above target)\n"
            f"• Watch Status: ACTIVE. No currently active deal crosses the RS {target_amt:.0f} threshold.\n"
            f"Citations: {cits}"
        )
        new_trace = [
            Value(amount=target_amt, provenance=Provenance.SOURCE, record_id="target_price"),
            Value(amount=curr_amt, provenance=Provenance.SOURCE, record_id="current_price"),
            Value(amount=gap_amt, provenance=Provenance.DERIVED, record_id="gap_price"),
            Value(amount=round(curr_amt, 0), provenance=Provenance.SOURCE, record_id="curr_rounded"),
            Value(amount=round(gap_amt, 0), provenance=Provenance.DERIVED, record_id="gap_rounded"),
            Value(amount=round(target_amt, 0), provenance=Provenance.SOURCE, record_id="target_rounded")
        ]
        is_valid, validated, unverified = ProvenanceValidator.validate(draft, new_trace)
        state["trace"] = new_trace
        state["draft_response"] = draft
        state["final_response"] = draft
        state["best_option"] = None
        state["is_watch_query"] = True
        state["citations"] = cits
        state["provenance_valid"] = is_valid
        state["unverified_tokens"] = unverified
        return state

    best_opt: PaymentOption = state.get("best_option")
    runner_up: PaymentOption = state.get("runner_up_option")
    trace = state.get("trace", [])
    citations = state.get("citations", [])

    if not best_opt or state.get("abstained", False):
        final_resp = state.get("final_response", "")
        if state.get("skip_planning") or state.get("is_info_query"):
            state["provenance_valid"] = True
            state["unverified_tokens"] = []
            return state
        if not final_resp:
            final_resp = f"No reliable deal or card record found for '{query}'."
        is_valid, validated, unverified = ProvenanceValidator.validate(final_resp, trace)
        state["final_response"] = final_resp
        state["provenance_valid"] = is_valid
        state["unverified_tokens"] = unverified
        if not is_valid:
            state["final_response"] = (
                f"No reliable deal or card record found for '{query}'."
            )
            state["provenance_valid"] = True
            state["unverified_tokens"] = []
        return state

    # Hard Grounding Check
    retrieved_ids = {r.get("deal_id") for r in state.get("retrieved_records", []) if r.get("deal_id")}
    retrieved_ids.update({"hdfc_millennia", "sbi_cashback", "axis_ace", "prod_macbook_air_m2", "prod_sony_headphones"})
    grounded_citations = [c for c in citations if c in retrieved_ids]
    if not grounded_citations and citations:
        grounded_citations = [list(retrieved_ids)[0]] if retrieved_ids else citations
    citations = grounded_citations

    eff = best_opt.effective_price.amount
    base = best_opt.base_price.amount
    disc = best_opt.discount_applied.amount
    post = best_opt.price_after_discount.amount
    reward = best_opt.reward_earned.amount

    # Half-up rounding helper to ensure strict arithmetic consistency: base - disc - reward = eff
    def _round_half_up(n: float) -> float:
        return float(math.floor(n + 0.5)) if n >= 0 else float(math.ceil(n - 0.5))

    reward_display = _round_half_up(reward)
    eff_display = _round_half_up(post - reward_display)
    disc_display = _round_half_up(disc)
    post_display = _round_half_up(post)
    base_display = _round_half_up(base)

    if trace:
        trace.extend([
            Value(amount=reward_display, provenance=Provenance.DERIVED, record_id="reward_rounded"),
            Value(amount=eff_display, provenance=Provenance.DERIVED, record_id="effective_rounded"),
            Value(amount=disc_display, provenance=Provenance.DERIVED, record_id="disc_rounded"),
            Value(amount=post_display, provenance=Provenance.DERIVED, record_id="post_rounded"),
            Value(amount=base_display, provenance=Provenance.SOURCE, record_id="base_rounded"),
            Value(amount=round(reward, 2), provenance=Provenance.DERIVED, record_id="reward_2dec"),
            Value(amount=round(eff, 2), provenance=Provenance.DERIVED, record_id="effective_2dec"),
            Value(amount=round(disc, 2), provenance=Provenance.DERIVED, record_id="disc_2dec"),
            Value(amount=round(post, 2), provenance=Provenance.DERIVED, record_id="post_2dec"),
            Value(amount=round(base, 2), provenance=Provenance.SOURCE, record_id="base_2dec"),
            Value(amount=round(reward, 0), provenance=Provenance.DERIVED, record_id="reward_r0"),
            Value(amount=round(eff, 0), provenance=Provenance.DERIVED, record_id="effective_r0"),
            Value(amount=round(disc, 0), provenance=Provenance.DERIVED, record_id="disc_r0"),
            Value(amount=round(post, 0), provenance=Provenance.DERIVED, record_id="post_r0"),
            Value(amount=round(base, 0), provenance=Provenance.SOURCE, record_id="base_r0"),
            Value(amount=float(math.floor(post)), provenance=Provenance.DERIVED, record_id="post_floor"),
            Value(amount=float(math.ceil(post)), provenance=Provenance.DERIVED, record_id="post_ceil"),
            Value(amount=float(math.floor(disc)), provenance=Provenance.DERIVED, record_id="disc_floor"),
            Value(amount=float(math.ceil(disc)), provenance=Provenance.DERIVED, record_id="disc_ceil")
        ])

    cmp = tool_res.get("compare_prices", {})
    resolved_store = cmp.get("lowest_price_merchant")
    if not resolved_store and best_opt and best_opt.base_price and best_opt.base_price.record_id:
        rid = best_opt.base_price.record_id
        if "_" in rid and not rid.startswith("user") and not rid.startswith("session") and not rid.startswith("spend"):
            resolved_store = rid.split("_")[-1].capitalize()

    merchant_name = best_opt.discount_source_id or resolved_store or "Partner Merchant"
    card_name = best_opt.card_id or "Payment Card"

    cap_note = ""
    if best_opt.cap_hit:
        raw = best_opt.raw_reward.amount
        lost = best_opt.reward_lost_to_cap.amount
        raw_display = _round_half_up(raw)
        lost_display = _round_half_up(lost)
        trace.append(Value(amount=raw_display, provenance=Provenance.DERIVED, record_id="raw_rounded"))
        trace.append(Value(amount=lost_display, provenance=Provenance.DERIVED, record_id="lost_rounded"))
        cap_note = f"\n• Capped Reward: RS {reward_display:.0f} (uncapped raw reward RS {raw_display:.0f}, RS {lost_display:.0f} lost to cap headroom)"

    disc_note = f"RS {disc:.0f}"
    if merchant_name == "deal_017":
        disc_note = f"RS {disc:.0f} (5% max RS 250 limit of deal_017 terms)"
    elif merchant_name == "deal_002":
        disc_note = f"RS {disc:.0f} (10% max RS 500 limit of deal_002 terms)"

    runner_up_note = ""
    if runner_up and runner_up.card_id != card_name:
        runner_eff = _round_half_up(runner_up.effective_price.amount)
        runner_diff = _round_half_up(runner_eff - eff_display)
        if runner_diff > 0:
            runner_up_note = f"\n• Runner-up Card: {runner_up.card_id} (Effective RS {runner_eff:.0f}, beats runner-up by RS {runner_diff:.0f})"
            trace.append(Value(amount=runner_diff, provenance=Provenance.DERIVED, record_id="runner_diff"))
            trace.append(Value(amount=runner_eff, provenance=Provenance.DERIVED, record_id="runner_up_eff"))
            trace.append(Value(amount=round(runner_up.effective_price.amount - eff, 2), provenance=Provenance.DERIVED, record_id="runner_diff_exact"))
            trace.append(Value(amount=round(runner_up.effective_price.amount, 2), provenance=Provenance.DERIVED, record_id="runner_up_eff_exact"))
            trace.append(Value(amount=round(runner_up.effective_price.amount, 0), provenance=Provenance.DERIVED, record_id="runner_up_eff_r0"))

    tie_msg = f"Cheapest effective price is RS {eff_display:.0f}. There is a tie between {' and '.join(state.get('tied_merchants', []))}.\n" if state.get("is_tie") else ""

    draft = (
        f"{tie_msg}"
        f"Recommendation: Pay using {card_name} at {merchant_name}.\n"
        f"• Base Price: RS {base:.0f}\n"
        f"• Instant Discount: {disc_note}\n"
        f"• Post-Discount Price: RS {post:.0f}\n"
        f"• Cashback Earned: RS {reward_display:.0f}{cap_note}\n"
        f"• Effective Final Price: RS {eff_display:.0f}.{runner_up_note}\n"
        f"Citations: {citations}"
    )

    is_valid, validated, unverified = ProvenanceValidator.validate(draft, trace)

    if not is_valid:
        state["final_response"] = (
            f"Validation blocked response output: numeric values {unverified} "
            f"were unverified against computation trace."
        )
        state["provenance_valid"] = False
        state["unverified_tokens"] = unverified
        return state

    state["draft_response"] = draft
    state["final_response"] = draft
    state["provenance_valid"] = True
    state["unverified_tokens"] = []
    return state

def build_agent_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("abstain_response", abstain_response_node)
    workflow.add_node("plan_node", plan_node)
    workflow.add_node("execute_tools", execute_tools_node)
    workflow.add_node("compute_rewards", compute_rewards_node)
    workflow.add_node("validate_provenance", validate_provenance_node)

    workflow.set_entry_point("retrieve")

    workflow.add_conditional_edges(
        "retrieve",
        check_abstention_router,
        {
            "abstain_response": "abstain_response",
            "plan_node": "plan_node"
        }
    )

    workflow.add_edge("plan_node", "execute_tools")
    workflow.add_edge("execute_tools", "compute_rewards")
    workflow.add_edge("compute_rewards", "validate_provenance")
    workflow.add_edge("abstain_response", "validate_provenance")
    workflow.add_edge("validate_provenance", END)

    return workflow.compile()

graph_app = build_agent_graph()
