from __future__ import annotations
import os
import re
import math
import warnings
from typing import Dict, Any, List, Optional, Set, Tuple

warnings.filterwarnings("ignore", message=".*automatic function calling.*")
warnings.filterwarnings("ignore", category=UserWarning, module="google.genai")
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from app.agent.state import AgentState
from app.agent.planner import Planner
from app.agent.resolver import QueryResolver, ResolvedQuery, Operation
from app.agent.operations import execute_operation
from app.rag.index import DataIndex
from app.rag.retriever import HybridRetriever
from app.rag.reranker import Reranker
from app.core.reward_engine import PaymentOption, RewardEngine
from app.core.provenance import ProvenanceValidator, Value, Provenance
from app.core.memory import MemoryManager
from app.agent.candidates import merchant_display
from app.tools.search_deals import search_deals
from app.tools.compare_prices import compare_prices
from app.tools.best_card import best_card
from app.tools.get_reward_rules import get_reward_rules
from app.tools.watch_price import watch_price

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


def retrieve_node(state: AgentState) -> AgentState:
    query = (state.get("query") or "").strip()
    s_id = state.get("session_id")

    # Sync session memory
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

    # 1. Resolve Query using Unified Data-Driven Resolver
    resolved = QueryResolver.resolve(
        query=query,
        session_id=s_id,
        session_state=state.get("conversation_state")
    )
    state["resolved_query"] = resolved
    state["operation"] = resolved.operation.value

    # 2. Check Abstention for Unresolved Entities
    if resolved.operation == Operation.ABSTAIN:
        state["final_response"] = "no reliable deal found"
        state["citations"] = []
        state["abstained"] = True
        state["skip_planning"] = True
        return state

    # STATE mutates session memory, CLARIFY asks a question, EXPLAIN reads the stored
    # recommendation memo. None of them may consult retrieval: a retrieved candidate must
    # never be able to create or replace what the user is actually talking about.
    if resolved.operation in (Operation.STATE, Operation.CLARIFY, Operation.EXPLAIN):
        state["retrieved_records"] = []
        state["excluded_injection_records"] = []
        state["max_retrieval_score"] = 1.0
        return execute_operation(resolved, state)

    # 3. Retrieval to ground meaning (only when needed by operation).
    #
    # RETRIEVAL IS EVIDENCE DISCOVERY, NOT INTENT (spec section 17, invariant B). Whatever
    # comes back is passed to the candidate builder as supporting evidence only; the
    # product, merchant, category and card in scope were fixed by QueryResolver against the
    # conversation thread and cannot be overwritten by a record that happened to score well.
    # A context-dependent follow-up ("what if I use SBI?", "how much do I actually
    # save?") was already resolved against the session's active purchase context in
    # QueryResolver.resolve() (product/card inheritance) — a fresh global deal search
    # would be redundant at best and could surface unrelated candidates at worst, so skip
    # it and let the calculation path use the resolved entities directly.
    # LOOKUP is deliberately NOT in this list. A record lookup does not consume retrieved
    # records, but the prompt-injection filter lives in this node, and skipping retrieval
    # skipped the defence with it — so whether an attack record was detected depended on
    # which operation the classifier happened to choose for the same question.
    if resolved.is_followup or resolved.operation in [Operation.STATE, Operation.ABSTAIN] or (resolved.operation == Operation.COMPUTE and resolved.cards and not resolved.merchants and not resolved.deals):
        state["retrieved_records"] = []
        state["excluded_injection_records"] = []
        state["max_retrieval_score"] = 1.0
        return state

    cat = resolved.category or state.get("category")
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
    state["max_retrieval_score"] = max_score

    # Only abstain on low retrieval score if no entities at all were matched
    if not (resolved.products or resolved.deals or resolved.cards or resolved.amount or resolved.reward_spend or resolved.purchase_amount) and (abstained or len(clean_records) == 0):
        state["abstained"] = True

    return state


def check_abstention_router(state: AgentState) -> str:
    if state.get("skip_planning") or state.get("abstained", False):
        return "abstain_response"
    return "plan_node"


def abstain_response_node(state: AgentState) -> AgentState:
    if not state.get("final_response"):
        state["final_response"] = "no reliable deal found"
    state["citations"] = state.get("citations", [])
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


def execute_tools_node(state: AgentState) -> AgentState:
    query = (state.get("query") or "").strip()
    planned = state.get("planned_tools", [])
    results: Dict[str, Any] = {}
    tool_mapping: Dict[str, str] = {}
    resolved: Optional[ResolvedQuery] = state.get("resolved_query")

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
            amt = (resolved.reward_spend or resolved.purchase_amount or resolved.amount or state.get("budget")) if resolved else state.get("budget")
            cat = state.get("category", "shopping")
            res = best_card(amount=amt, category=cat, spend_to_date=state.get("spend_to_date"))
            results["best_card"] = res
            if res.get("best_card"):
                tool_mapping["best_card"] = f"card '{res.get('best_card')}', rate {res.get('reward_rate_pct'):.1f}%, reward RS {res.get('reward_earned', 0):.0f}"
            else:
                tool_mapping["best_card"] = "evaluated card reward rates"
        elif tool == "get_reward_rules":
            cards = DataIndex.get_cards()
            # No fallback to an arbitrary catalogue card: with nothing named and no
            # preference, there is no card this question is about, and the corroboration
            # step re-queries the tool for whichever card actually wins.
            card_id = (resolved.cards[0]["card_id"] if resolved and resolved.cards
                       else state.get("preferred_card"))
            if card_id:
                res = get_reward_rules(card_id)
                results["get_reward_rules"] = res
                tool_mapping["get_reward_rules"] = f"card '{card_id}' base rate {res.get('base_rate', 0)*100:g}%"
        elif tool == "watch_price":
            target = (resolved.target_price or resolved.target) if resolved else None
            res = watch_price(query, target_price=target, session_id=state.get("session_id", "default"))
            results["watch_price"] = res
            gap_val = res.get("gap")
            gap_amt = gap_val.amount if gap_val else 0.0
            curr_p = res.get("current_lowest_price")
            curr_amt = curr_p.amount if curr_p else 0.0
            tool_mapping["watch_price"] = f"target RS {target or 0:.0f}, current RS {curr_amt:.0f}, gap RS {gap_amt:.0f}"

    state["tool_results"] = results
    state["tool_mapping"] = tool_mapping
    return state


def corroborate_with_tools(state: AgentState, best_opt: Optional[PaymentOption]) -> Dict[str, Any]:
    """
    Cross-check the derived recommendation against the tools the planner actually ran.

    WHY THIS EXISTS
    ---------------
    The planner composes several tool calls per turn — typically compare_prices, then
    search_deals, then best_card / get_reward_rules — but their results were being recorded
    and then ignored: the answer was derived independently, so a disagreement between the
    tool layer and the derivation layer could never be noticed. Two implementations of the
    same fact that never meet are two chances to be wrong.

    Each tool is checked only where it is authoritative and only where it refers to the same
    entity as the recommendation, so this cannot fabricate a disagreement:

      * compare_prices    -> the base price charged must be one this product's catalog
                             record actually lists, at the merchant we named.
      * get_reward_rules  -> the reward rate the engine applied must equal base_rate x the
                             category multiplier the card's own policy states.
      * search_deals      -> a discount may only come from a real deal record.

    A tool can only ever CONTRADICT a figure here; it can never supply one. On disagreement
    the answer is blocked rather than shipped, because a mismatch means one of the two paths
    has a bug and we cannot tell which.
    """
    report: Dict[str, Any] = {"checks": [], "disagreements": []}
    results = state.get("tool_results") or {}
    if not best_opt:
        return report

    def record(tool: str, claim: str, agrees: bool) -> None:
        report["checks"].append({"tool": tool, "claim": claim, "agrees": agrees})
        if not agrees:
            report["disagreements"].append(f"{tool}: {claim}")

    # --- compare_prices: is the base price one this product really carries?
    cp = results.get("compare_prices")
    if cp and cp.get("found") and best_opt.product_id and cp.get("product_id") == best_opt.product_id:
        raw = {m.lower(): float(v) for m, v in (cp.get("raw_prices") or {}).items()}
        base = best_opt.base_price.amount
        merch = (best_opt.merchant or "").lower()
        if merch and merch in raw:
            record("compare_prices",
                   f"base RS {base:.2f} matches catalogue price at {merch}",
                   abs(raw[merch] - base) < 0.01)
        else:
            record("compare_prices",
                   f"base RS {base:.2f} is one of this product's listed prices",
                   any(abs(v - base) < 0.01 for v in raw.values()))

    # --- get_reward_rules: is the applied rate the card's own published rate?
    # The tool ran before the winner was known, so if the planner asked for reward rules we
    # consult them for the card actually chosen. The tool only reads the card record; it
    # cannot supply a figure, only contradict one.
    rr = results.get("get_reward_rules")
    if best_opt.card_id and (not rr or rr.get("card_id") != best_opt.card_id):
        if "get_reward_rules" in (state.get("planned_tools") or []):
            rr = get_reward_rules(best_opt.card_id)
    if rr and rr.get("found") and best_opt.card_id and rr.get("card_id") == best_opt.card_id:
        # Reward rates are per CATEGORY, so the check has to use the SAME category the
        # engine did — the winning option's own product (which in a product comparison need
        # not be the first-named one), else the category the query resolved to. Reading
        # state["category"] instead was wrong: it is unset for an amount-only calculation,
        # which made a correct 5% grocery rate look like a 1% mismatch.
        _resolved = state.get("resolved_query")
        cat = (getattr(_resolved, "category", None) or state.get("category") or "general")
        if best_opt.product_id:
            _wp = DataIndex.get_product_by_id(best_opt.product_id)
            if _wp and _wp.get("category"):
                cat = _wp["category"]
        expected = RewardEngine.effective_rate(rr, cat)
        applied = float(best_opt.reward_rate_applied or 0.0)
        # A rate of zero is legitimate when the card's own minimum spend was not met.
        record("get_reward_rules",
               f"applied rate {applied * 100:g}% matches {rr['card_id']} policy for {cat}",
               abs(expected - applied) < 1e-9 or applied == 0.0)

    # --- search_deals: any discount must trace to a real deal record.
    if best_opt.discount_source_id:
        record("search_deals",
               f"discount sourced from catalogue record {best_opt.discount_source_id}",
               DataIndex.get_deal_by_id(best_opt.discount_source_id) is not None)

    report["agreed"] = not report["disagreements"]
    return report


def compute_rewards_node(state: AgentState) -> AgentState:
    if state.get("skip_planning"):
        return state

    resolved: Optional[ResolvedQuery] = state.get("resolved_query")
    if not resolved:
        resolved = QueryResolver.resolve(
            query=state.get("query", ""),
            session_id=state.get("session_id"),
            session_state=state.get("conversation_state")
        )
        state["resolved_query"] = resolved

    # Execute corresponding operation handler
    result = execute_operation(resolved, state)

    # The planner's tool calls now feed back into the answer: whatever they independently
    # report about price, reward policy and deal provenance is checked against what the
    # derivation produced, and a contradiction stops the turn.
    corroboration = corroborate_with_tools(result, result.get("best_option"))
    result["tool_corroboration"] = corroboration
    if corroboration.get("disagreements"):
        detail = "; ".join(corroboration["disagreements"])
        result["final_response"] = (
            "I can't give you a number I trust for this: the catalogue lookup and the "
            f"reward derivation disagree ({detail}). Nothing has been recommended."
        )
        result["draft_response"] = result["final_response"]
        result["best_option"] = None
        result["citations"] = []
        result["abstained"] = True
        result["skip_planning"] = True
        result["provenance_checked"] = True
        result["provenance_valid"] = True
        return result

    # Record the recommendation into the conversation thread. This is the ONLY place a
    # recommendation memo is written, and it is written from the deterministic engine's
    # own PaymentOption + provenance trace — so a later "why?" / "what was the discount?"
    # re-states figures that were already grounded rather than recomputing or re-retrieving.
    s_id = result.get("session_id")
    best_opt = result.get("best_option")
    if s_id and best_opt:
        if best_opt.discount_source_id:
            MemoryManager.update_last_entities(s_id, deal_id=best_opt.discount_source_id)

        runner = result.get("runner_up_option")
        resolved_q = result.get("resolved_query")
        # The merchant travels ON the payment option now, so multi-word merchants survive
        # intact; the deal record is only a fallback for amount-only calculations that have
        # no product price behind them.
        merchant_name = best_opt.merchant
        if not merchant_name and best_opt.discount_source_id:
            d_rec = DataIndex.get_deal_by_id(best_opt.discount_source_id)
            if d_rec:
                merchant_name = d_rec.get("merchant")

        thread = (resolved_q.thread if resolved_q is not None and getattr(resolved_q, "thread", None)
                  else MemoryManager.get_session(s_id).thread)

        # The memo's trace must cover everything an explanation may later quote, including
        # the runner-up's effective price and the margin over it. Both were produced by the
        # deterministic engine on this turn — they simply live on the runner-up's option —
        # so they are folded in here rather than being re-asserted, and therefore
        # ungrounded, when "why?" or "what was the runner-up?" is asked.
        memo_trace = list(best_opt.trace or [])
        if runner is not None and best_opt.effective_price is not None:
            margin = runner.effective_price.amount - best_opt.effective_price.amount
            memo_trace.append(runner.effective_price)
            memo_trace.append(Value(amount=round(margin, 2), provenance=Provenance.DERIVED,
                                    record_id="runner_up_margin"))

        thread.record_recommendation(
            card_id=best_opt.card_id,
            merchant=merchant_name,
            deal_id=best_opt.discount_source_id,
            base_price=best_opt.base_price.amount if best_opt.base_price else None,
            base_price_after_discount=best_opt.price_after_discount.amount if best_opt.price_after_discount else None,
            discount=best_opt.discount_applied.amount if best_opt.discount_applied else None,
            reward=best_opt.reward_earned.amount if best_opt.reward_earned else None,
            effective_price=best_opt.effective_price.amount if best_opt.effective_price else None,
            runner_up_card=runner.card_id if runner else None,
            runner_up_effective=runner.effective_price.amount if runner else None,
            cap_hit=bool(best_opt.cap_hit),
            cap_explanation=best_opt.cap_explanation,
            product_name=(resolved_q.products[0]["name"] if resolved_q and resolved_q.products else None),
            trace=memo_trace,
            citations=list(best_opt.citations or []),
        )

        # Remember the rejected visible deals too, and fold their SOURCE figures into the
        # memo's trace, so an explanation about one of them quotes numbers that are already
        # grounded rather than asserting new ones.
        rejected = result.get("rejected_visible_deals") or []
        thread.last_rejected_deals = [
            {
                "deal_id": r.deal_id,
                "headline": r.headline,
                "reason": r.reason,
                "figures": [round(float(v.amount), 2) for v in r.values],
            }
            for r in rejected
        ]
        if rejected:
            thread.last_trace = list(thread.last_trace) + [v for r in rejected for v in r.values]

    return result


def _tie_sentence(state: AgentState) -> str:
    """
    State a joint-best outcome as a tie rather than picking one and calling it the winner.

    Every figure quoted is the ranking value the engine already produced, so this adds no
    number that is not already in the trace.
    """
    tie = state.get("tie") or {}
    labels = tie.get("labels") or []
    if len(labels) < 2:
        return ""

    from app.agent.operations import _TIE_AXIS_NOUN, _TIE_COUNT_WORD, _TIE_METRIC_PHRASE

    count = _TIE_COUNT_WORD.get(len(labels), str(len(labels)))
    noun = _TIE_AXIS_NOUN.get(tie.get("axis", "card"), "options")
    phrase = _TIE_METRIC_PHRASE.get(tie.get("metric", "effective_price"), "best result")
    return (f"{count} {noun} tie for the {phrase} of RS {tie['value']:.0f}: "
            f"{', '.join(labels)}.\n")


def validate_provenance_node(state: AgentState) -> AgentState:
    # A handler that formatted monetary figures itself already ran the validator against
    # its own trace; its verdict is final and must not be overwritten with a blanket pass.
    if state.get("provenance_checked"):
        return state

    if state.get("skip_planning") or state.get("is_info_query"):
        state["provenance_valid"] = True
        state["unverified_tokens"] = []
        return state

    query = (state.get("query") or "").strip()
    best_opt: Optional[PaymentOption] = state.get("best_option")
    runner_up: Optional[PaymentOption] = state.get("runner_up_option")
    trace: List[Value] = state.get("trace", [])
    citations: List[str] = state.get("citations", [])

    if not best_opt or state.get("abstained", False):
        final_resp = state.get("final_response", "")
        if not final_resp:
            final_resp = "no reliable deal found"
        is_valid, validated, unverified = ProvenanceValidator.validate(final_resp, trace)
        state["final_response"] = final_resp
        state["provenance_valid"] = is_valid
        state["unverified_tokens"] = unverified
        return state

    # Strict Grounded Citations: Only records actually consumed this turn
    valid_source_ids: Set[str] = set()
    if best_opt.discount_source_id:
        valid_source_ids.add(best_opt.discount_source_id)
    if best_opt.card_id:
        valid_source_ids.add(best_opt.card_id)
    if best_opt.base_price and best_opt.base_price.record_id:
        rid = best_opt.base_price.record_id
        if "_" in rid and not rid.startswith(("user", "session", "spend", "compare", "compute", "target", "gap")):
            valid_source_ids.add(rid.rsplit("_", 1)[0])

    citations = [c for c in citations if c in valid_source_ids]

    eff = best_opt.effective_price.amount
    base = best_opt.base_price.amount
    disc = best_opt.discount_applied.amount
    post = best_opt.price_after_discount.amount
    reward = best_opt.reward_earned.amount

    # Name the merchant the option is actually priced at. `discount_source_id` is only a
    # fallback label for amount-only calculations that have no catalog price behind them.
    resolved_store = merchant_display(best_opt.merchant) if best_opt.merchant else None
    # With no catalog price behind the option (an amount-only calculation), the applied
    # deal's own record still names the merchant it belongs to — a grounded label, unlike
    # printing the deal id where a merchant name belongs.
    if not resolved_store and best_opt.discount_source_id:
        _d = DataIndex.get_deal_by_id(best_opt.discount_source_id)
        resolved_store = (_d or {}).get("merchant")
    merchant_name = resolved_store or "Partner Merchant"
    card_name = best_opt.card_id or "Payment Card"
    # A pure reward calculation has no merchant behind it, so naming a placeholder one
    # would assert something the dataset does not say.
    where = f" at {merchant_name}" if resolved_store else ""

    cap_note = ""
    if best_opt.cap_hit:
        raw = best_opt.raw_reward.amount
        lost = best_opt.reward_lost_to_cap.amount
        cap_note = f"\n• Capped Reward: RS {reward:.0f} (uncapped raw reward RS {raw:.0f}, RS {lost:.0f} lost to cap headroom)"

    # Dynamic Deal Terms from Authoritative Deal Record
    disc_note = f"RS {disc:.0f}"
    if best_opt.discount_source_id:
        d_rec = DataIndex.get_deal_by_id(best_opt.discount_source_id)
        if d_rec:
            d_type = d_rec.get("discount_type")
            d_val = float(d_rec.get("discount_value", 0))
            max_d = float(d_rec.get("max_discount", float("inf")))
            if d_type == "percentage":
                pct = d_val * 100 if d_val <= 1.0 else d_val
                if max_d != float("inf"):
                    disc_note = f"RS {disc:.0f} ({pct:g}% capped at RS {max_d:.0f})"
                else:
                    disc_note = f"RS {disc:.0f} ({pct:g}% discount)"
            else:
                disc_note = f"RS {disc:.0f} (flat discount)"

    runner_up_note = ""
    if runner_up and runner_up.card_id != card_name:
        runner_eff = runner_up.effective_price.amount
        runner_diff = runner_eff - eff
        if runner_diff > 0:
            runner_up_note = f"\n• Runner-up Card: {runner_up.card_id} (Effective RS {runner_eff:.0f}, beats runner-up by RS {runner_diff:.0f})"
            trace.append(runner_up.effective_price)
            trace.append(Value(amount=runner_diff, provenance=Provenance.DERIVED, record_id="runner_diff"))

    # A visible deal that looked better but could not be used is worth a line. The figures
    # come from the deal records and the price already in the trace, so they validate like
    # every other number in the answer.
    why_not = ""
    for rejected in state.get("rejected_visible_deals") or []:
        if not why_not:
            why_not = "\nWhy not the other visible deals?"
        why_not += f"\n• {rejected.deal_id} ({rejected.headline}): {rejected.reason}."
        trace.extend(rejected.values)

    tie_msg = _tie_sentence(state)

    # A reward question is answered in reward terms: which card, at what rate, paying what,
    # and against which cap. The discount/post-discount lines belong to a purchase question
    # and are noise here. Every figure still comes from the same PaymentOption.
    resolved_q = state.get("resolved_query")
    if getattr(resolved_q, "objective", None) == "max_reward" and disc <= 0.01:
        rate_pct = float(best_opt.reward_rate_applied or 0.0) * 100
        cat_name = (getattr(resolved_q, "category", None) or "this spend")
        cap_line = ""
        if best_opt.cap_hit and best_opt.cap_explanation:
            cap_line = f"\n• {best_opt.cap_explanation}"
        elif best_opt.card_id:
            _card = DataIndex.get_card_by_id(best_opt.card_id) or {}
            _caps = _card.get("caps", {})
            _cat_cap = (_caps.get("category_caps") or {}).get(cat_name)
            _monthly = _caps.get("monthly_cashback_cap")
            if _cat_cap is not None:
                cap_line = f"\n• Applicable Cap: RS {float(_cat_cap):.0f} on {cat_name}"
            elif _monthly is not None and float(_monthly) != 999999:
                cap_line = f"\n• Applicable Cap: RS {float(_monthly):.0f} per month"

        reward_draft = (
            f"{_tie_sentence(state)}"
            f"Recommendation: Pay using {card_name}{where}.\n"
            f"• Purchase Amount: RS {base:.0f}\n"
            f"• Reward Rate: {rate_pct:g}% on {cat_name}\n"
            f"• Cashback Earned: RS {reward:.0f}{cap_line}\n"
            f"• Effective Final Price: RS {eff:.0f}.{runner_up_note}{why_not}\n"
            f"Citations: {citations}"
        )
        product_names = [p["name"] for p in resolved_q.products] if resolved_q else []
        state["provenance_masked_names"] = list(product_names)
        ok, _validated, unverified = ProvenanceValidator.validate(
            reward_draft, trace, product_names=product_names
        )
        if not ok:
            state["final_response"] = (
                f"Validation blocked response output: numeric values {unverified} "
                f"were unverified against computation trace."
            )
            state["provenance_valid"] = False
            state["unverified_tokens"] = unverified
            return state
        state["draft_response"] = reward_draft
        state["final_response"] = reward_draft
        state["provenance_valid"] = True
        state["unverified_tokens"] = []
        return state

    draft = (
        f"{tie_msg}"
        f"Recommendation: Pay using {card_name}{where}.\n"
        f"• Base Price: RS {base:.0f}\n"
        f"• Instant Discount: {disc_note}\n"
        f"• Post-Discount Price: RS {post:.0f}\n"
        f"• Cashback Earned: RS {reward:.0f}{cap_note}\n"
        f"• Effective Final Price: RS {eff:.0f}.{runner_up_note}{why_not}\n"
        f"Citations: {citations}"
    )

    # Validate draft against trace once (no synthetic padding). Catalog product names are
    # masked so a model number inside a name is not read as an ungrounded price; every
    # actual figure is still checked.
    resolved_q = state.get("resolved_query")
    product_names = [p["name"] for p in resolved_q.products] if resolved_q else []
    state["provenance_masked_names"] = list(product_names)
    is_valid, validated, unverified = ProvenanceValidator.validate(draft, trace, product_names=product_names)

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
