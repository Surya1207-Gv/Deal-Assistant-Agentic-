"""
Operation handlers.

Each handler declares a CANDIDATE SPACE and a RANKING AXIS, then hands both to the shared
candidate builder (agent/candidates.py). None of them performs its own arithmetic: every
rupee in every answer comes from `RewardEngine` via `enumerate_options`, and every
eligibility decision from the canonical `is_deal_eligible`. What differs between handlers
is which alternatives are in scope and how the result is narrated — never the math
(spec section 30, invariant E).
"""

from __future__ import annotations
import dataclasses
import math
from typing import Dict, Any, List, Optional, Tuple


from app.rag.index import DataIndex
from app.core.provenance import Value, Provenance, ProvenanceValidator
from app.core.reward_engine import RewardEngine, PaymentOption
from app.core.primitives import filter_deals, filter_products, aggregate_records, enumerate_options, explain_comparison, is_deal_eligible
from app.agent.resolver import ResolvedQuery, Operation
from app.agent.candidates import (
    FinancialCandidate,
    PurchaseFrame,
    build_candidates,
    build_payment_options,
    build_purchase_frame,
    evidence_report,
    explain_rejected_visible,
    merchant_display,
    merged_evidence,
)
from app.tools.watch_price import watch_price
from app.core.memory import MemoryManager


def _thread_of(resolved: ResolvedQuery, state: Dict[str, Any]):
    """
    The canonical conversation thread for this turn.

    It travels ON the resolved query, so handlers never re-derive context from the session
    or the raw text (invariant A). The session lookup is only a fallback for callers that
    built a ResolvedQuery without one.
    """
    if getattr(resolved, "thread", None) is not None:
        return resolved.thread
    s_id = state.get("session_id")
    return MemoryManager.get_session(s_id).thread if s_id else None


def _emit(state: Dict[str, Any], draft: str, citations: List[str], trace: List[Value],
          *, validate: bool = False, product_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Attach a finished answer to the state.

    When `validate` is set the draft is checked against its own trace here, and the verdict
    is marked final so the graph's validation node does not wave it through: a handler that
    formats monetary figures itself must be held to the same grounding rule as the
    optimizer's generated recommendation.
    """
    state["final_response"] = draft
    state["draft_response"] = draft
    state["citations"] = citations
    state["trace"] = trace
    state["is_info_query"] = True
    # Recorded so an independent re-check can validate this draft against the SAME inputs
    # the pipeline used, rather than re-deriving a verdict with less information.
    state["provenance_masked_names"] = list(product_names or [])

    if validate:
        is_valid, _validated, unverified = ProvenanceValidator.validate(
            draft, trace, product_names=product_names
        )
        state["provenance_checked"] = True
        state["provenance_valid"] = is_valid
        state["unverified_tokens"] = unverified
        if not is_valid:
            blocked = (
                f"Validation blocked response output: numeric values {unverified} "
                f"were unverified against computation trace."
            )
            state["final_response"] = blocked
    else:
        state["provenance_valid"] = True
    return state


def _catalog_text(deals: Optional[List[Dict[str, Any]]] = None,
                  cards: Optional[List[Dict[str, Any]]] = None,
                  products: Optional[List[Dict[str, Any]]] = None) -> List[str]:
    """
    The catalog wording an answer quotes verbatim, for provenance masking.

    Record titles and names routinely contain digits that are part of the name rather than
    a monetary claim. Passing them to the validator lets it check the figures the answer
    actually asserts, instead of flagging a record's own title.
    """
    names: List[str] = []
    for d in deals or []:
        if d.get("title"):
            names.append(d["title"])
    for c in cards or []:
        if c.get("name"):
            names.append(c["name"])
    for p in products or []:
        if p.get("name"):
            names.append(p["name"])
    return names


def _rate_card_response(state: Dict[str, Any], cards: List[Dict[str, Any]], category: str) -> Dict[str, Any]:
    """
    No spend figure exists, so no monetary answer can be grounded. Report the reward rates
    the card records actually state and ask for an amount — never invent a placeholder
    spend to have something to rank (spec section 28).
    """
    lines = [f"Reward rates for {category}:"]
    cits: List[str] = []
    trace: List[Value] = []
    for c in cards:
        cid = c["card_id"]
        cits.append(cid)
        pct = RewardEngine.effective_rate(c, category) * 100
        trace.append(Value(amount=pct, provenance=Provenance.SOURCE, record_id=f"{cid}_{category}_rate"))
        lines.append(f"• {c.get('name', cid)}: {pct:g}% reward rate on {category}")
    draft = "\n".join(lines) + "\nPlease provide a spend amount to calculate exact monetary savings."
    return _emit(state, draft, cits, trace)


def _clarify_purchase(state: Dict[str, Any], frame: PurchaseFrame) -> Dict[str, Any]:
    cats = sorted({(p.get("category") or "").lower() for p in DataIndex.get_products() if p.get("category")})
    draft = (
        "I need to know what this purchase is before I can price it. "
        f"Which product or category is it — {', '.join(cats)}?"
    )
    return _emit(state, draft, [], [])


def _frame_for(resolved: ResolvedQuery, state: Dict[str, Any]) -> PurchaseFrame:
    frame = build_purchase_frame(resolved, state, state.get("retrieved_records"))
    # Recorded for transparency only: which retrieved records the candidate space admitted
    # after eligibility, and which it declined. Retrieval informs; the thread decides.
    state["evidence_report"] = evidence_report(frame)
    return frame


def _record_winner(state: Dict[str, Any], best: Optional[PaymentOption],
                   runner_up: Optional[PaymentOption],
                   options: Optional[List[PaymentOption]] = None) -> None:
    # The FULL candidate space is published alongside the winner. Every one of these was
    # costed by the deterministic engine before anything was chosen, which is what makes
    # the choice auditable rather than merely asserted.
    if options is not None:
        state["payment_options"] = options
    state["best_option"] = best
    state["runner_up_option"] = runner_up
    if best is not None:
        state["trace"] = best.trace
        state["citations"] = best.citations


def _candidate_cards_for_compute(resolved: ResolvedQuery) -> List[Dict[str, Any]]:
    """
    COMPUTE calculates ONE specified combination, not the globally cheapest one (that's
    OPTIMIZE). So the candidate card set must be: the card named in this turn, else the
    card the user already told us to use (session preference), else — only when neither
    is known — every card (e.g. a bare rate-lookup with no card context at all).
    """
    if resolved.cards:
        return resolved.cards
    pref_card_id = resolved.constraints.get("preferred_card")
    if pref_card_id:
        pref_obj = DataIndex.get_card_by_id(pref_card_id)
        if pref_obj:
            return [pref_obj]
    return DataIndex.get_cards()


def execute_operation(resolved: ResolvedQuery, state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dispatches the resolved operation to its corresponding pure handler.
    """
    op = resolved.operation

    if op == Operation.ABSTAIN:
        return handle_abstain(resolved, state)
    elif op == Operation.CLARIFY:
        return handle_clarify(resolved, state)
    elif op == Operation.EXPLAIN:
        return handle_explain(resolved, state)
    elif op == Operation.STATE:
        return handle_state(resolved, state)
    elif op == Operation.WATCH:
        return handle_watch(resolved, state)
    elif op == Operation.LOOKUP:
        return handle_lookup(resolved, state)
    elif op == Operation.LIST:
        return handle_list(resolved, state)
    elif op == Operation.AGGREGATE:
        return handle_aggregate(resolved, state)
    elif op == Operation.ELIGIBILITY:
        return handle_eligibility(resolved, state)
    elif op == Operation.COMPARE:
        return handle_compare(resolved, state)
    elif op == Operation.COMPUTE:
        return handle_compute(resolved, state)
    elif op == Operation.OPTIMIZE:
        return handle_optimize(resolved, state)
    else:
        return handle_abstain(resolved, state)


def _explain_comparison_lines(thread) -> List[str]:
    """
    Restate a stored comparison from its structured rows.

    Every figure here was produced by the deterministic engine when the comparison was
    first computed and memoized alongside its provenance trace, so "which one was cheaper?",
    "why wasn't <loser> cheaper?" and "what was the runner-up?" are answered from the same
    numbers the user already saw (spec sections 12 and 13).
    """
    memo = thread.last_comparison or {}
    rows = memo.get("rows", [])
    if not rows:
        return []

    axis = memo.get("axis", "option")
    metric = memo.get("metric", "effective_price")
    subject = memo.get("product_name")
    lines = [
        f"That comparison was across {axis}s"
        + (f" for {subject}" if subject and axis != "product" else "")
        + f", ranked by {_METRIC_HEADING.get(metric, 'effective final price')}:"
    ]
    for r in rows:
        lines.append("• " + _comparison_row_text(r))

    winner_key = memo.get("winner_key")
    if winner_key is None:
        # No winner was chosen, because the objective was never stated and the readings
        # disagreed. Do not invent one now.
        lines.append("No single option was best: the answer depended on what you were "
                     "optimising for.")
        return lines

    key = _METRIC_ROW_KEY.get(metric, "effective_price")
    winner = next((r for r in rows if r.get("key") == winner_key), None)
    loser = next((r for r in rows if r.get("key") != winner_key), None)
    if winner and loser:
        delta = abs(loser.get(key, 0.0) - winner.get(key, 0.0))
        if delta > 0.01:
            lines.append(f"{winner['label']} won by RS {delta:,.0f} on "
                         f"{_METRIC_HEADING.get(metric, 'effective final price')}.")
        else:
            lines.append(f"{winner['label']} and {loser['label']} came out level.")
    return lines


def _comparison_row_text(r: Dict[str, Any]) -> str:
    """One comparison row, rendered from already-derived figures only."""
    parts = [f"Base RS {r['base_price']:,.0f}"]
    if r.get("discount"):
        src = f" ({r['deal_id']})" if r.get("deal_id") else ""
        parts.append(f"discount RS {r['discount']:,.0f}{src}")
    else:
        parts.append("no eligible deal")
    if r.get("card_id"):
        parts.append(f"cashback RS {r['reward']:,.0f} on {r['card_id']}")
    parts.append(f"effective RS {r['effective_price']:,.0f}")
    return f"{r['label']} — " + "; ".join(parts)


def _referenced_rejected_deals(resolved: ResolvedQuery, state: Dict[str, Any], thread):
    """
    Which rejected deal, if any, this turn is asking about — by its id, or by one of the
    figures the previous answer quoted for it.
    """
    rows = list(getattr(thread, "last_rejected_deals", None) or [])
    if not rows:
        return []
    query = (state.get("query") or "").lower()
    by_id = [r for r in rows if r["deal_id"].lower() in query]
    if by_id:
        return by_id
    if resolved.amount is not None:
        return [r for r in rows
                if any(abs(float(f) - resolved.amount) < 0.01 for f in r.get("figures", []))]
    return []


def handle_explain(resolved: ResolvedQuery, state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Answer a question about the result we already produced, using ONLY the memos recorded
    by the deterministic engine (numbers + the original provenance trace).

    Nothing here recomputes or re-retrieves: every figure quoted was already grounded when
    it was first derived, and the original trace is re-attached so provenance validation
    checks these numbers against the same evidence that justified them (invariant G).
    """
    thread = _thread_of(resolved, state)
    rec = thread.last_recommendation if thread else None

    # Asking about a deal the last recommendation could not use. Answered from the stored
    # rejection memo — the reason and every figure in it were derived when the candidate
    # space was built, so nothing is recomputed and nothing new is asserted.
    referenced = _referenced_rejected_deals(resolved, state, thread)
    if referenced:
        lines = []
        for row in referenced:
            lines.append(f"The {row['headline']} offer ({row['deal_id']}) could not be applied: "
                         f"it {row['reason']}.")
        if rec and rec.get("effective_price") is not None:
            used = rec["deal_id"] if rec.get("deal_id") else "no merchant offer"
            lines.append(f"The recommendation used {used}, for an effective "
                         f"RS {rec['effective_price']:,.0f}.")
        state["skip_planning"] = True
        return _emit(state, "\n".join(lines), list(thread.last_citations),
                     list(thread.last_trace), validate=True,
                     product_names=[rec.get("product_name") or ""] if rec else [])

    # When the newest thing we computed was a COMPARISON, an explanation has to include the
    # alternatives that lost — those rows are exactly what a "why"/"which one" question is
    # about, and they exist nowhere else once the turn is over.
    if thread is not None and thread.comparison_is_current():
        lines = _explain_comparison_lines(thread)
        if lines:
            draft = "\n".join(lines)
            return _emit(
                state, draft, list(thread.last_citations), list(thread.last_trace),
                validate=True,
                product_names=[(thread.last_comparison or {}).get("product_name") or ""],
            )

    if not rec:
        return handle_abstain(resolved, state)

    card = rec.get("card_id") or "the selected card"
    merchant = (rec.get("merchant") or "the selected merchant")
    if isinstance(merchant, str):
        merchant = merchant.capitalize()
    base = rec.get("base_price")
    disc = rec.get("discount")
    post = rec.get("price_after_discount")
    reward = rec.get("reward")
    eff = rec.get("effective_price")
    deal_id = rec.get("deal_id")
    product_name = rec.get("product_name")

    lines: List[str] = []
    subject = f"{product_name} " if product_name else ""
    lines.append(f"Here is how that {subject}recommendation was built:")
    if base is not None:
        lines.append(f"• Base Price: RS {base:,.0f} at {merchant}")
    if disc is not None:
        src = f" (deal {deal_id})" if deal_id else " (no deal applied)"
        lines.append(f"• Instant Discount: RS {disc:,.0f}{src}")
    if post is not None:
        lines.append(f"• Post-Discount Price: RS {post:,.0f}")
    if reward is not None:
        lines.append(f"• Cashback on {card}: RS {reward:,.0f}")
    if rec.get("cap_hit") and rec.get("cap_explanation"):
        lines.append(f"• {rec['cap_explanation']}")
    if eff is not None:
        lines.append(f"• Effective Final Price: RS {eff:,.0f}")

    # Why this option rather than the alternative — stated as a delta between two
    # already-computed effective prices, never as a fresh judgement.
    ru_card = rec.get("runner_up_card")
    ru_eff = rec.get("runner_up_effective")
    if ru_card and ru_eff is not None and eff is not None:
        delta = ru_eff - eff
        if delta > 0:
            lines.append(
                f"• It beat the next-best option ({ru_card}, effective RS {ru_eff:,.0f}) by RS {delta:,.0f}."
            )
        elif abs(delta) < 0.01:
            lines.append(f"• It tied with {ru_card} at RS {ru_eff:,.0f}; the card id ordering broke the tie.")
    else:
        lines.append("• No other eligible option was cheaper for this purchase.")

    draft = "\n".join(lines)
    state["skip_planning"] = True
    return _emit(
        state, draft, list(thread.last_citations), list(thread.last_trace),
        validate=True, product_names=[product_name or ""],
    )


def handle_clarify(resolved: ResolvedQuery, state: Dict[str, Any]) -> Dict[str, Any]:
    """
    A reference was genuinely ambiguous and the rival readings would produce different
    money. Ask a concise question instead of guessing. No citations and an empty trace:
    nothing financial is being asserted, so there is nothing to ground.
    """
    resp = resolved.clarification or "Could you clarify what you meant?"
    state["final_response"] = resp
    state["draft_response"] = resp
    state["citations"] = []
    state["trace"] = []
    state["skip_planning"] = True
    state["is_info_query"] = True
    state["provenance_valid"] = True
    return state


def handle_abstain(resolved: ResolvedQuery, state: Dict[str, Any]) -> Dict[str, Any]:
    state["final_response"] = "no reliable deal found"
    state["draft_response"] = "no reliable deal found"
    state["citations"] = []
    state["trace"] = []
    state["abstained"] = True
    state["skip_planning"] = True
    state["provenance_valid"] = True
    return state


def handle_state(resolved: ResolvedQuery, state: Dict[str, Any]) -> Dict[str, Any]:
    s_id = state.get("session_id")
    conv_state = state.get("conversation_state") or {}
    trace = []

    if resolved.clear_preference or resolved.clear_budget:
        cleared = []
        if resolved.clear_preference:
            state["preferred_card"] = None
            conv_state["preferred_card"] = None
            conv_state["category_preferences"] = {}
            thread = _thread_of(resolved, state)
            if thread is not None:
                thread.clear_card_dimension()
            if s_id:
                MemoryManager.clear_card(s_id)
            cleared.append("preferred card")
        if resolved.clear_budget:
            state["budget"] = None
            conv_state["budget"] = None
            thread = _thread_of(resolved, state)
            if thread is not None:
                thread.clear_budget_dimension()
            if s_id:
                MemoryManager.clear_budget(s_id)
            cleared.append("budget")
        resp = f"Got it — I've cleared your {' and '.join(cleared)} for this conversation."
        state["final_response"] = resp
        state["draft_response"] = resp
        state["citations"] = []
        state["trace"] = []
        state["skip_planning"] = True
        state["is_info_query"] = True
        state["provenance_valid"] = True
        return state

    if resolved.amount is not None:
        # STATE was already chosen, so a figure in this turn is a budget declaration. The
        # budget is written to the persistent slot AND to the thread's `active_budget`, and
        # deliberately NOT to `active_amount`: a ceiling is not a figure to spend
        # (spec section 15).
        b_val = resolved.budget if resolved.budget is not None else resolved.amount
        state["budget"] = b_val
        conv_state["budget"] = b_val
        thread = _thread_of(resolved, state)
        if thread is not None:
            thread.active_budget = b_val
        if s_id:
            MemoryManager.update_budget(s_id, b_val)
        cat = resolved.category or state.get("category")
        if cat:
            state["category"] = cat
            if s_id:
                MemoryManager.update_category(s_id, cat)
            resp = f"Got it — I’ll use ₹{b_val:,.0f} as your {cat} budget for this conversation."
        else:
            resp = f"Got it — I’ll use ₹{b_val:,.0f} as your budget for this conversation."
        trace.append(Value(amount=b_val, provenance=Provenance.SOURCE, record_id="user_budget"))
    elif resolved.cards:
        chosen_card = resolved.cards[0]["card_id"]
        cat = resolved.category
        thread = _thread_of(resolved, state)
        if thread is not None and not cat:
            # A stated preference is PERSISTENT state; it also becomes the card in play,
            # whereas a hypothetical only ever moves `active_card` (spec section 14).
            thread.preferred_card = chosen_card
            thread.active_card = chosen_card
        if cat:
            conv_state.setdefault("category_preferences", {})[cat] = chosen_card
            if s_id:
                MemoryManager.update_category_preference(s_id, cat, chosen_card)
            resp = f"Got it — I’ll prefer {chosen_card} for {cat}."
        else:
            state["preferred_card"] = chosen_card
            conv_state["preferred_card"] = chosen_card
            if s_id:
                MemoryManager.update_card(s_id, chosen_card)
            resp = f"Got it — I’ll prefer {chosen_card} for this conversation."
    else:
        resp = "Session preferences updated."

    state["final_response"] = resp
    state["draft_response"] = resp
    state["citations"] = []
    state["trace"] = trace
    state["skip_planning"] = True
    state["is_info_query"] = True
    state["provenance_valid"] = True
    return state


def handle_watch(resolved: ResolvedQuery, state: Dict[str, Any]) -> Dict[str, Any]:
    query = state.get("query", "")
    prod_name = resolved.products[0]["name"] if resolved.products else query
    target_price = resolved.target or (resolved.amount if resolved.amount else 0.0)

    res = watch_price(prod_name, target_price=target_price, session_id=state.get("session_id", "default"))

    # A watch is a commitment about a real product's real price. If the catalogue does not
    # contain the product, there is no current price to compare a threshold against and
    # nothing to watch — confirming one would be inventing the subject of the promise.
    # ("Rolex Submariner Gold Watch discount on Chrono24" routed here purely because the
    # product's own name contains the word "watch".)
    if not res.get("found") or res.get("current_lowest_price") is None:
        return handle_abstain(resolved, state)

    curr_lowest = res.get("current_lowest_price")
    curr_amt = curr_lowest.amount if curr_lowest else 0.0
    merch = res.get("lowest_price_merchant") or "Catalog"
    t_val = target_price or curr_amt
    gap_amt = max(0.0, curr_amt - t_val)

    prod_id = res.get("product_id") or (resolved.products[0]["product_id"] if resolved.products else "prod_watch")
    cits = [prod_id]

    draft = (
        f"Recommendation: Price watch registered for '{res.get('product_name', prod_name)}' at target RS {t_val:.0f}.\n"
        f"• Current Lowest Price: RS {curr_amt:.0f} (at {merch})\n"
        f"• Target Price: RS {t_val:.0f}\n"
        f"• Price Gap to Target: RS {gap_amt:.0f} (above target)\n"
        f"• Watch Status: ACTIVE. No currently active deal crosses the RS {t_val:.0f} threshold.\n"
        f"Citations: {cits}"
    )
    trace = [
        Value(amount=t_val, provenance=Provenance.SOURCE, record_id="target_price"),
        Value(amount=curr_amt, provenance=Provenance.SOURCE, record_id="current_price"),
        Value(amount=gap_amt, provenance=Provenance.DERIVED, record_id="gap_price")
    ]
    state["final_response"] = draft
    state["draft_response"] = draft
    state["citations"] = cits
    state["trace"] = trace
    state["is_info_query"] = True
    state["provenance_valid"] = True
    return state


def handle_lookup(resolved: ResolvedQuery, state: Dict[str, Any]) -> Dict[str, Any]:
    trace: List[Value] = []
    cits: List[str] = []

    # Deal attribute lookup
    deals_to_check = resolved.deals
    if not deals_to_check and resolved.merchants:
        deals_to_check = filter_deals(merchant=resolved.merchants[0], category=resolved.category)
        if not deals_to_check:
            deals_to_check = filter_deals(merchant=resolved.merchants[0])

    if deals_to_check:
        d = deals_to_check[0]
        d_id = d["deal_id"]
        cits.append(d_id)
        d_type = d.get("discount_type")
        d_val = float(d.get("discount_value", 0))
        min_s = float(d.get("min_spend", 0))
        max_d = float(d.get("max_discount", d_val or 0))
        d_merch = d.get("merchant", "Partner")
        d_cat = d.get("category", "shopping")

        trace.extend([
            Value(amount=min_s, provenance=Provenance.SOURCE, record_id=f"{d_id}_min_spend"),
            Value(amount=max_d, provenance=Provenance.SOURCE, record_id=f"{d_id}_max_discount")
        ])

        if d_type == "percentage":
            display_pct = RewardEngine.display_percentage(d)
            trace.append(Value(amount=float(display_pct), provenance=Provenance.SOURCE, record_id=f"{d_id}_pct"))
            if resolved.amount:
                calc_disc = RewardEngine.deal_discount(d, resolved.amount)
                final_p = resolved.amount - calc_disc
                trace.extend([
                    Value(amount=resolved.amount, provenance=Provenance.SOURCE, record_id="order_amt"),
                    Value(amount=calc_disc, provenance=Provenance.DERIVED, record_id="calculated_discount"),
                    Value(amount=final_p, provenance=Provenance.DERIVED, record_id="final_price")
                ])
                draft = f"The {d_merch} {d_cat} deal ({d_id}) offers a {display_pct:g}% instant discount (capped at RS {max_d:.0f}) on a minimum spend of RS {min_s:.0f}. For a RS {resolved.amount:.0f} order, the discount is RS {calc_disc:.0f}, resulting in a price of RS {final_p:.0f}."
            else:
                draft = f"The {d_merch} {d_cat} deal ({d_id}) offers a {display_pct:g}% instant discount (capped at RS {max_d:.0f}) on a minimum spend of RS {min_s:.0f}."
        else:
            trace.append(Value(amount=float(d_val), provenance=Provenance.SOURCE, record_id=f"{d_id}_flat_val"))
            if resolved.amount:
                calc_disc = RewardEngine.deal_discount(d, resolved.amount)
                final_p = resolved.amount - calc_disc
                trace.extend([
                    Value(amount=resolved.amount, provenance=Provenance.SOURCE, record_id="order_amt"),
                    Value(amount=calc_disc, provenance=Provenance.DERIVED, record_id="calculated_discount"),
                    Value(amount=final_p, provenance=Provenance.DERIVED, record_id="final_price")
                ])
                draft = f"The {d_merch} {d_cat} deal ({d_id}) offers a flat RS {d_val:.0f} instant discount on a minimum spend of RS {min_s:.0f}. For a RS {resolved.amount:.0f} order, the discount is RS {calc_disc:.0f}, resulting in a price of RS {final_p:.0f}."
            else:
                draft = f"The {d_merch} {d_cat} deal ({d_id}) offers a flat RS {d_val:.0f} instant discount on a minimum spend of RS {min_s:.0f}."

    # Card attribute lookup
    elif resolved.cards:
        c = resolved.cards[0]
        c_id = c["card_id"]
        cits.append(c_id)
        b_rate = float(c.get("base_rate", 0.01)) * 100
        caps = c.get("caps", {})
        m_cap = float(caps.get("monthly_cashback_cap", 0))
        cat_caps = caps.get("category_caps", {})

        trace.extend([
            Value(amount=b_rate, provenance=Provenance.SOURCE, record_id=f"{c_id}_base_rate"),
            Value(amount=m_cap, provenance=Provenance.SOURCE, record_id=f"{c_id}_monthly_cap")
        ])
        for ck, cv in cat_caps.items():
            trace.append(Value(amount=float(cv), provenance=Provenance.SOURCE, record_id=f"{c_id}_{ck}_cap"))

        if resolved.category and resolved.category in cat_caps:
            c_cap = float(cat_caps[resolved.category])
            draft = f"The reward cap on {c.get('name', c_id)} for {resolved.category} is RS {c_cap:.0f}."
        elif resolved.field == "cap":
            draft = f"The monthly cashback cap on {c.get('name', c_id)} is RS {m_cap:.0f}."
        else:
            cat_str = ", ".join([f"RS {cv:.0f} on {ck}" for ck, cv in cat_caps.items()]) if cat_caps else "none"
            draft = f"Reward Rules for {c.get('name', c_id)} ({c_id}): Base cashback rate is {b_rate:g}%. Monthly cashback cap is RS {m_cap:.0f}. Category caps: {cat_str}."

    # Product price lookup
    elif resolved.products:
        p = resolved.products[0]
        p_name = p["name"]
        p_id = p["product_id"]
        cits.append(p_id)
        prices = p.get("prices", {})
        lines = []
        lowest_p = float("inf")
        lowest_m = None
        for m_name, p_amt in prices.items():
            trace.append(Value(amount=float(p_amt), provenance=Provenance.SOURCE, record_id=f"{p_id}_{m_name.lower()}"))
            lines.append(f"• {m_name.capitalize()}: RS {p_amt:,.0f}")
            if p_amt < lowest_p:
                lowest_p = float(p_amt)
                lowest_m = m_name.capitalize()
        draft = (
            f"Price Comparison for {p_name}:\n"
            + "\n".join(lines) + "\n"
            + f"Lowest price is RS {lowest_p:,.0f} at {lowest_m}."
        )
    else:
        return handle_abstain(resolved, state)

    return _emit(state, draft, cits, trace, validate=True,
                 product_names=_catalog_text(deals_to_check, resolved.cards, resolved.products))


def handle_list(resolved: ResolvedQuery, state: Dict[str, Any]) -> Dict[str, Any]:
    matched_deals = filter_deals(
        category=resolved.category,
        merchant=resolved.merchants[0] if resolved.merchants else None,
        card=resolved.cards[0]["card_id"] if resolved.cards else None,
        product=resolved.products[0] if resolved.products else None
    )

    # A listing scoped to a PRODUCT is only about places that sell it. Category scoping alone
    # let an offer from a merchant that does not carry the product appear under "deals for
    # <product>" — a silent merchant introduction in listing form. Merchant-agnostic deals
    # stay. Derived from the product's own price map, so it holds for any catalogue.
    if resolved.products and not resolved.merchants:
        sellers = {m.lower() for m in (resolved.products[0].get("prices") or {})}
        if sellers:
            matched_deals = [
                d for d in matched_deals
                if not d.get("merchant") or d["merchant"].lower() in sellers
            ]

    if not matched_deals:
        return handle_abstain(resolved, state)

    lines = []
    trace: List[Value] = []
    cits: List[str] = []

    for d in matched_deals:
        d_id = d["deal_id"]
        cits.append(d_id)
        d_type = d.get("discount_type")
        d_val = float(d.get("discount_value", 0))
        min_s = float(d.get("min_spend", 0))
        max_d = float(d.get("max_discount", d_val or 0))

        trace.extend([
            Value(amount=min_s, provenance=Provenance.SOURCE, record_id=f"{d_id}_min_spend"),
            Value(amount=max_d, provenance=Provenance.SOURCE, record_id=f"{d_id}_max_discount")
        ])

        if d_type == "percentage":
            display_pct = RewardEngine.display_percentage(d)
            trace.append(Value(amount=float(display_pct), provenance=Provenance.SOURCE, record_id=f"{d_id}_pct"))
            desc_str = f"{display_pct:g}% instant discount (capped at RS {max_d:.0f}) on min spend RS {min_s:.0f}"
        else:
            trace.append(Value(amount=float(d_val), provenance=Provenance.SOURCE, record_id=f"{d_id}_flat_val"))
            desc_str = f"flat RS {d_val:.0f} instant discount on min spend RS {min_s:.0f}"

        if d.get("card_specific"):
            desc_str += f" (Card: {d.get('card_specific')})"
        lines.append(f"• {d_id} — {d.get('title')}: {desc_str}")

    header = "Available deals"
    if resolved.merchants:
        header += f" on {resolved.merchants[0].capitalize()}"
    if resolved.category:
        header += f" for {resolved.category}"
    if resolved.cards:
        header += f" with {resolved.cards[0]['card_id']}"
    header += ":"

    draft = header + "\n" + "\n".join(lines)
    return _emit(state, draft, cits, trace, validate=True,
                 product_names=_catalog_text(matched_deals, resolved.cards, resolved.products))


def handle_aggregate(resolved: ResolvedQuery, state: Dict[str, Any]) -> Dict[str, Any]:
    matched_deals = filter_deals(
        category=resolved.category,
        merchant=resolved.merchants[0] if resolved.merchants else None,
        card=resolved.cards[0]["card_id"] if resolved.cards else None,
        product=resolved.products[0] if resolved.products else None
    )

    if not matched_deals:
        return handle_abstain(resolved, state)

    val, win_deal = aggregate_records(
        matched_deals,
        field=resolved.field or "discount",
        op=resolved.aggregate or "max",
        amount=resolved.amount
    )

    if not win_deal:
        return handle_abstain(resolved, state)

    d_id = win_deal["deal_id"]
    min_s = float(win_deal.get("min_spend", 0))
    max_d = float(win_deal.get("max_discount", win_deal.get("discount_value", 0)))
    d_merch = win_deal.get("merchant", "Partner")
    d_title = win_deal.get("title", d_id)

    trace = [
        Value(amount=val, provenance=Provenance.DERIVED, record_id=f"{d_id}_best_val"),
        Value(amount=min_s, provenance=Provenance.SOURCE, record_id=f"{d_id}_min_spend"),
        Value(amount=max_d, provenance=Provenance.SOURCE, record_id=f"{d_id}_max_discount")
    ]

    merch_str = f" at {d_merch}" if d_merch else ""
    draft = f"The deal with the largest discount{merch_str} is {d_title} ({d_id}), offering a maximum discount of RS {val:.0f} (min spend RS {min_s:.0f})."

    return _emit(state, draft, [d_id], trace, validate=True,
                 product_names=_catalog_text([win_deal], resolved.cards, resolved.products))


def handle_eligibility(resolved: ResolvedQuery, state: Dict[str, Any]) -> Dict[str, Any]:
    trace: List[Value] = []
    cits: List[str] = []

    # Deal eligibility — routed through the SAME canonical is_deal_eligible() the
    # optimizer uses, so an eligibility answer can never diverge from what the
    # computation engine would actually do (single source of eligibility truth).
    if resolved.deals:
        deal = resolved.deals[0]
        d_id = deal["deal_id"]
        cits.append(d_id)
        min_s = float(deal.get("min_spend", 0))
        card_sp = deal.get("card_specific")
        card_obj = resolved.cards[0] if resolved.cards else None

        trace.append(Value(amount=min_s, provenance=Provenance.SOURCE, record_id=f"{d_id}_min_spend"))
        if resolved.amount is not None:
            trace.append(Value(amount=resolved.amount, provenance=Provenance.SOURCE, record_id="query_amount"))

        eligible, reason = is_deal_eligible(
            deal=deal,
            product=resolved.products[0] if resolved.products else None,
            merchant=resolved.merchants[0] if resolved.merchants else None,
            purchase_amount=resolved.amount,
            card=card_obj,
            category=resolved.category
        )

        if eligible:
            draft = f"Yes, deal {d_id} ({deal.get('title')}) applies. It requires a minimum spend of RS {min_s:.0f}" + (f" and card {card_sp}." if card_sp else ".")
        else:
            reason_text = {
                "MERCHANT_MISMATCH": f"the merchant does not match this deal's merchant ({deal.get('merchant')})",
                "CATEGORY_MISMATCH": f"the category does not match this deal's category ({deal.get('category')})",
                "PRODUCT_MISMATCH": "the named product is not within this deal's product scope",
                "CARD_MISMATCH": f"card {card_obj['card_id'] if card_obj else '(unspecified)'} does not match the required card {card_sp}",
                "MIN_SPEND_NOT_MET": f"order RS {resolved.amount:.0f} is below minimum spend RS {min_s:.0f}" if resolved.amount is not None else "the minimum spend is not met",
            }.get(reason, reason or "the stated eligibility conditions are not met")
            draft = f"No, deal {d_id} cannot be used because {reason_text}."

    # Card eligibility — grounded strictly in the card's own actual base_rate (never an
    # assumed/hardcoded percentage).
    elif resolved.cards:
        c = resolved.cards[0]
        c_id = c["card_id"]
        cits.append(c_id)
        m_cap = float(c.get("caps", {}).get("monthly_cashback_cap", 0))
        b_rate = float(c.get("base_rate", 0.01)) * 100
        trace.extend([
            Value(amount=m_cap, provenance=Provenance.SOURCE, record_id=f"{c_id}_monthly_cap"),
            Value(amount=b_rate, provenance=Provenance.SOURCE, record_id=f"{c_id}_base_rate")
        ])
        draft = f"Yes, you can use {c.get('name', c_id)} ({c_id}). It provides a {b_rate:g}% base cashback rate, capped at RS {m_cap:.0f} per month."
    else:
        return handle_abstain(resolved, state)

    return _emit(state, draft, cits, trace, validate=True,
                 product_names=_catalog_text(resolved.deals, resolved.cards, resolved.products))


# How each ranking metric reads when a comparison is summarised under it.
_VIEW_LABEL = {
    "effective_price": "Lowest effective price",
    "reward": "Most cashback",
    "discount": "Largest discount",
}

# How a comparison READS under each metric. The ranking already followed the objective; the
# narration did not, so a reward comparison was correctly ranked by reward and then described
# with an effective-price verdict — the right answer explained in the wrong terms.
_METRIC_HEADING = {
    "effective_price": "effective final price",
    "reward": "cashback earned",
    "discount": "discount applied",
}
_METRIC_ROW_KEY = {
    "effective_price": "effective_price",
    "reward": "reward",
    "discount": "discount",
}
# How each metric is named when asking which the user prioritises.
_METRIC_PRIORITY = {
    "effective_price": "the lowest final price",
    "reward": "cashback",
    "discount": "discount",
}


def _metric_verdict(metric: str, winner_label: str, loser_label: str,
                    winner_value: float, loser_value: float) -> str:
    delta = abs(winner_value - loser_value)
    if delta < 0.01:
        return f"{winner_label} and {loser_label} come out level at RS {winner_value:,.0f}."
    if metric == "reward":
        return (f"{winner_label} earns RS {delta:,.0f} more than {loser_label} "
                f"(RS {winner_value:,.0f} vs RS {loser_value:,.0f}).")
    if metric == "discount":
        return (f"{winner_label} discounts RS {delta:,.0f} more than {loser_label} "
                f"(RS {winner_value:,.0f} vs RS {loser_value:,.0f}).")
    return (f"{winner_label} is cheaper than {loser_label} by RS {delta:,.0f} "
            f"(RS {winner_value:,.0f} vs RS {loser_value:,.0f}).")


def _ranking_view(frame, options, axis: str, metric: str):
    """
    Rank the SAME payment options under one metric.

    Re-ranking only: the options were already costed by the engine, and every metric reads a
    field those options already carry. Nothing is recomputed, so two views can never disagree
    about a figure — only about which figure matters.
    """
    view_frame = dataclasses.replace(frame, metric=metric)
    candidates = build_candidates(view_frame, axis, options)
    if not candidates:
        return [], "", frozenset()

    extractor, _descending = RewardEngine.METRICS[metric]
    top = extractor(candidates[0].option)
    joint = [c for c in candidates if abs(extractor(c.option) - top) < 0.01]

    if len(joint) > 1:
        names = " and ".join(c.label for c in joint)
        summary = f"{names} tie at RS {top:,.0f}"
    elif len(candidates) > 1:
        runner = candidates[1]
        summary = (f"{candidates[0].label} at RS {top:,.0f} "
                   f"(vs {runner.label} RS {extractor(runner.option):,.0f})")
    else:
        summary = f"{candidates[0].label} at RS {top:,.0f}"

    return candidates, summary, frozenset(c.key for c in joint)


def _comparison_axis_for(resolved: ResolvedQuery, frame) -> str:
    """
    Which dimension the user is actually comparing.

    Taken from the resolved state, in precedence order: the axis on which this turn named
    two or more peers, else the axis of the comparison still open in the thread. A card
    named during a merchant comparison applies TO that comparison; it does not silently
    turn it into a card comparison (spec sections 8 and 9).
    """
    if len(resolved.merchants) >= 2:
        return "merchant"
    if len(resolved.cards) >= 2:
        return "card"
    if len(resolved.deals) >= 2:
        return "deal"
    if len(resolved.products) >= 2:
        return "product"
    if frame.comparison_axis:
        return frame.comparison_axis
    # Nothing was named as a peer set, but a comparison was asked for: compare the payment
    # cards, which is the only axis that always has more than one member.
    return "card"


def handle_compare(resolved: ResolvedQuery, state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rank the alternatives the user is actually comparing, each with its OWN complete
    financial derivation.

    THE BUG THIS REPLACES: the previous implementation discovered both merchants, then
    either compared their raw shelf prices (ignoring deals, cards and rewards entirely) or
    handed the flattened option list to the optimizer and reported the single global winner
    as though it were a comparison. Both hid the losing alternative rather than computing it.

    Now every entity on the axis becomes its own candidate: its own eligible deals, its own
    eligible cards, its own discount, reward and effective price — and only then are they
    ranked against each other (invariants C and D).
    """
    frame = _frame_for(resolved, state)

    if frame.insufficient == "purchase_unknown":
        return _clarify_purchase(state, frame)

    axis = _comparison_axis_for(resolved, frame)

    # A card/deal comparison with no spend basis cannot produce money, only rates.
    if frame.insufficient == "amount_unknown":
        cards = frame.cards if axis == "card" else (resolved.cards or DataIndex.get_cards())
        return _rate_card_response(state, cards, frame.category or "general")

    options = build_payment_options(frame)
    if not options:
        return handle_abstain(resolved, state)

    candidates = build_candidates(frame, axis, options)
    if not candidates:
        return handle_abstain(resolved, state)

    if len(candidates) == 1:
        # Only one alternative survived eligibility — say so rather than presenting a
        # one-sided list as a comparison.
        best = candidates[0].option
        _record_winner(state, best, None, options)
        state["comparison_axis"] = axis
        state["comparison_candidates"] = candidates
        return state

    # AMBIGUOUS COMPARISON.
    #
    # "A or B for this purchase?" names the alternatives but not what makes one better. The
    # objective then falls back to a default, and the default silently decides the answer —
    # which is only harmless while the metrics agree. They frequently do not: a card can earn
    # LESS cashback and still leave you paying less, because it unlocks a larger merchant
    # discount. Reporting one ranking as "the" answer, with the other metric visible in the
    # same table pointing the other way, is worse than saying nothing.
    #
    # So when the user never stated an objective, rank the same options both ways. If the
    # readings agree there is no ambiguity and the answer proceeds normally; if they disagree,
    # both are reported and neither is presented as the winner. Generic across axes and
    # metrics, and triggered by the DATA disagreeing rather than by any phrasing.
    if getattr(resolved, "objective_source", "explicit") == "default":
        # Rank the SAME options under each meaningful metric. Only the ordering changes:
        # entities, merchant, card, product and amount all come from the one frame, so a
        # different metric can never quietly become a different question.
        views = []
        for metric in ("effective_price", "reward", "discount"):
            cands, summary, winners = _ranking_view(frame, options, axis, metric)
            if cands:
                views.append((metric, cands, summary, winners))

        # Ambiguous exactly when the metrics do not agree on who wins. When they agree there
        # is nothing to ask about and a single verdict is honest.
        if len(views) > 1 and len({w for _m, _c, _s, w in views}) > 1:
            lines = ["You did not say which of these matters, and they do not agree:"]
            for metric, _cands, summary, _w in views:
                lines.append(f"• {_VIEW_LABEL.get(metric, metric)}: {summary}.")
            priorities = ", ".join(_METRIC_PRIORITY[m] for m, _c, _s, _w in views[:-1])
            lines.append(f"There is no single best choice unless you tell me whether you "
                         f"prioritise {priorities} or {_METRIC_PRIORITY[views[-1][0]]}.")
            lines.append("Each option in full:")
            for c in candidates:
                lines.append("• " + _comparison_row_text(c.row()))

            all_cands = [c for _m, cs, _s, _w in views for c in cs]
            citations, trace = merged_evidence(all_cands)

            thread = _thread_of(resolved, state)
            if thread is not None:
                thread.record_comparison(
                    axis=axis,
                    rows=[c.row() for c in candidates],
                    winner_key=None,          # no winner was chosen, and none is invented
                    product_name=frame.product["name"] if frame.product else None,
                    trace=trace,
                    citations=citations,
                )

            _record_winner(state, candidates[0].option, candidates[1].option, options)
            state["comparison_axis"] = axis
            state["comparison_candidates"] = candidates
            state["comparison_ambiguous"] = [
                {"metric": m, "summary": sm, "winners": sorted(w)} for m, _c, sm, w in views
            ]
            state["skip_planning"] = True
            draft = "\n".join(lines) + f"\nCitations: {citations}"
            return _emit(state, draft, citations, trace, validate=True,
                         product_names=[p["name"] for p in frame.products])

    winner = candidates[0]
    runner_up = candidates[1]
    # The verdict is stated in the units the ranking used, not always in rupees-paid.
    _extract, _desc = RewardEngine.METRICS.get(frame.metric, RewardEngine.METRICS["effective_price"])
    winner_value = _extract(winner.option)
    runner_value = _extract(runner_up.option)
    delta = abs(runner_value - winner_value)

    # When the axis IS the product, naming one product as "the subject" would be wrong —
    # the products are the alternatives, not the context.
    subject = "" if axis == "product" else f" for {frame.describe_subject()}"
    heading = _METRIC_HEADING.get(frame.metric, "effective final price")
    lines = [f"Comparison across {axis}s{subject}, ranked by {heading}:"]
    for c in candidates:
        lines.append("• " + _comparison_row_text(c.row()))

    lines.append(_metric_verdict(frame.metric, winner.label, runner_up.label,
                                 winner_value, runner_value))

    citations, trace = merged_evidence(candidates)
    trace = list(trace) + [
        Value(amount=round(delta, 2), provenance=Provenance.DERIVED, record_id="comparison_delta"),
        Value(amount=round(delta, 0), provenance=Provenance.DERIVED, record_id="comparison_delta_rounded"),
    ]

    # Persist the comparison as structured state so follow-ups rank the SAME alternatives
    # instead of starting a fresh global search (spec sections 7 and 13).
    thread = _thread_of(resolved, state)
    if thread is not None:
        thread.record_comparison(
            axis=axis,
            rows=[c.row() for c in candidates],
            winner_key=winner.key,
            product_name=frame.product["name"] if frame.product else None,
            metric=frame.metric,
            trace=trace,
            citations=citations,
        )

    _record_winner(state, winner.option, runner_up.option, options)
    state["comparison_axis"] = axis
    state["comparison_candidates"] = candidates
    state["skip_planning"] = True

    draft = "\n".join(lines) + f"\nCitations: {citations}"
    product_names = [p["name"] for p in frame.products]
    return _emit(state, draft, citations, trace, validate=True, product_names=product_names)


def handle_compute(resolved: ResolvedQuery, state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Price ONE specified combination (this merchant, this card, this amount) rather than
    searching for the best one — the same candidate builder, with a candidate space the
    user has already narrowed.
    """
    frame = _frame_for(resolved, state)

    if frame.insufficient == "purchase_unknown":
        return _clarify_purchase(state, frame)

    # COMPUTE evaluates the combination the user specified: the card named this turn, else
    # the one they already told us to use. Only when neither is known does it fall back to
    # every card (a bare rate question with no card context at all).
    frame.cards = _candidate_cards_for_compute(resolved)

    if frame.insufficient == "amount_unknown":
        if resolved.cards or resolved.category or resolved.constraints.get("preferred_card"):
            return _rate_card_response(state, frame.cards, frame.category or "general")
        return handle_abstain(resolved, state)

    options = build_payment_options(frame)
    if not options:
        return handle_abstain(resolved, state)

    best_opt = options[0]
    runner_up = next((o for o in options if o.card_id != best_opt.card_id), None)
    _record_winner(state, best_opt, runner_up, options)
    return state


# How each ranking metric reads in a sentence, and which entity a tie is reported over.
_TIE_METRIC_PHRASE = {
    "effective_price": "lowest effective price",
    "reward": "highest cashback",
    "discount": "largest discount",
}
_TIE_AXIS_NOUN = {"merchant": "merchants", "deal": "deals", "card": "cards"}
_TIE_COUNT_WORD = {2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight"}


def _detect_tie(options: List[PaymentOption], best: PaymentOption, metric: str,
                pref_card: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Find every option that is joint-best, and say what distinguishes them.

    Ties were previously measured on effective price whatever the question was, and reported
    only when the tied options sat at different MERCHANTS. So three cards paying an identical
    RS 400 of cashback — the tie a reward question is most likely to produce — were resolved
    silently by card-id ordering and one of them presented as the unique winner.

    The comparison now uses the metric the answer is actually ranked by, and the tie is
    reported over whichever dimension genuinely varies. Ordering is alphabetical for stable
    presentation only; nothing here computes or alters a figure.
    """
    extractor, _descending = RewardEngine.METRICS.get(metric, RewardEngine.METRICS["effective_price"])
    best_value = extractor(best)

    tied = [
        o for o in options
        if abs(extractor(o) - best_value) < 0.01 and (not pref_card or o.card_id == pref_card)
    ]
    if len(tied) < 2:
        return None

    # Report over the coarsest dimension that actually differs: a tie across merchants is a
    # choice of where to shop, which matters more than which card settles it.
    for axis, key in (("merchant", lambda o: merchant_display(o.merchant) if o.merchant else None),
                      ("deal", lambda o: o.discount_source_id),
                      ("card", lambda o: o.card_id)):
        labels = sorted({key(o) for o in tied if key(o)})
        if len(labels) > 1:
            return {
                "metric": metric,
                "value": round(float(best_value), 2),
                "axis": axis,
                "labels": labels,
            }
    return None


def handle_optimize(resolved: ResolvedQuery, state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Find the cheapest grounded route for the purchase in scope.

    The candidate space is every (merchant x deal x card) combination the frame admits —
    built and costed before anything is chosen, so the winner is selected from complete
    financial results rather than a deal being picked first and priced afterwards
    (spec section 5).
    """
    frame = _frame_for(resolved, state)

    if frame.insufficient == "purchase_unknown":
        return _clarify_purchase(state, frame)

    if frame.insufficient == "amount_unknown":
        if resolved.cards or resolved.category:
            return _rate_card_response(
                state, resolved.cards or DataIndex.get_cards(), frame.category or "general"
            )
        return handle_abstain(resolved, state)

    options = build_payment_options(frame)
    if not options:
        return handle_abstain(resolved, state)

    # A card the user told us to use constrains WHICH option is recommended, but every
    # other card is still costed so the runner-up is a real computed alternative.
    pref_card = resolved.constraints.get("preferred_card")
    named_opts = [o for o in options if o.card_id == pref_card] if pref_card else []
    if named_opts:
        best_opt = named_opts[0]
        runner_up = next((o for o in options if o.card_id != pref_card), None)
    else:
        best_opt = options[0]
        runner_up = next((o for o in options if o.card_id != best_opt.card_id), None)

    budget = resolved.constraints.get("budget")
    if budget is not None and best_opt.effective_price.amount > float(budget):
        state["final_response"] = f"no reliable deal found; the cheapest grounded option is RS {best_opt.effective_price.amount:,.0f}."
        state["abstained"] = True
        state["skip_planning"] = True
        state["best_option"] = None
        state["citations"] = []
        return state

    # Report a genuine tie between merchants rather than silently preferring one.
    state["tie"] = _detect_tie(options, best_opt, frame.metric, pref_card)
    state["is_tie"] = bool(state["tie"])
    # Back-compatible view for callers that only ever knew about merchant ties.
    state["tied_merchants"] = (state["tie"] or {}).get("labels", [])

    # Explain the visible deals that lost. Read-only: it re-runs the same eligibility
    # engine over the candidate space that was already built, and cannot add to it.
    state["rejected_visible_deals"] = explain_rejected_visible(
        frame=frame,
        retrieved_deal_ids=[
            r.get("deal_id") for r in (state.get("retrieved_records") or []) if r.get("deal_id")
        ],
        applied_discount=best_opt.discount_applied.amount,
        applied_deal_id=best_opt.discount_source_id,
        reference_price=best_opt.base_price.amount,
    )

    _record_winner(state, best_opt, runner_up, options)
    return state
