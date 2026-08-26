from __future__ import annotations

"""
Candidate space construction — the stage that was missing.

THE PROBLEM THIS SOLVES
-----------------------
Retrieved records are not financial candidates. Before this module existed, each handler
built its own ad-hoc `base_prices` list, handed the flattened result to the reward engine,
and read back the single globally best option. That is correct for "find me the cheapest
way to buy X" and silently wrong for everything else: a merchant comparison retrieved both
merchants, priced only the winning one, and reported that as a comparison.

The pipeline is now explicit and shared:

    RESOLVED THREAD STATE  (what the user is talking about)
            |
            v
    PurchaseFrame          (which products / merchants / cards / amount are in scope)
            |
            v
    BASE PRICES            (one per product x merchant actually in the catalog)
            |
            v
    DEAL ELIGIBILITY       (canonical is_deal_eligible, per combination)
            |
            v
    PAYMENT OPTIONS        (deterministic RewardEngine arithmetic, one per combination)
            |
            v
    CANDIDATES             (grouped along the axis being ranked, best option per group)
            |
            v
    RANKING                (by the metric the question actually asks for)

Every operation — optimize, compare, compute, follow-up recalculation — goes through these
same stages. They differ only in which candidate space they declare and which axis they
rank along, never in how a rupee is computed.

FINANCIAL SAFETY INVARIANT
--------------------------
Nothing in this module accepts a number from an LLM, and nothing in it invents one. Prices
come from product records, discounts and caps from deal/card records, and every derived
figure from `RewardEngine`. When the frame cannot be grounded (no product, no category, no
merchant, no card, no deal), it reports `insufficient` so the caller can ask — it never
picks an arbitrary dataset row to make an answer possible.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.core.primitives import BasePrice, enumerate_options, filter_deals, is_deal_eligible
from app.core.provenance import Provenance, Value
from app.core.reward_engine import PaymentOption, RewardEngine
from app.rag.index import DataIndex


# ---------------------------------------------------------------------------- merchants

def merchant_display_map() -> Dict[str, str]:
    """lower-case merchant -> the catalog's own spelling of it."""
    names: Dict[str, str] = {}
    for p in DataIndex.get_products():
        for m in (p.get("prices") or {}).keys():
            names.setdefault(m.lower(), m)
    for d in DataIndex.get_deals():
        m = d.get("merchant")
        if m:
            names.setdefault(m.lower(), m)
    return names


def merchant_display(merchant: Optional[str]) -> str:
    if not merchant:
        return "Partner Merchant"
    return merchant_display_map().get(merchant.lower(), merchant.title())


# ------------------------------------------------------------------------------- frame

@dataclass
class PurchaseFrame:
    """
    The fully-resolved, canonical description of WHAT to compute over this turn.

    Produced once from the conversation thread + this turn's resolved entities, then shared
    by every handler. If a handler needs to know the product, merchant, card or amount, it
    reads it here — it never re-derives it from the raw query or from retrieval.
    """
    products: List[Dict[str, Any]] = field(default_factory=list)
    category: Optional[str] = None
    merchants: List[str] = field(default_factory=list)      # empty => every merchant that prices the product
    cards: List[Dict[str, Any]] = field(default_factory=list)
    amount: Optional[float] = None                          # explicit user figure overriding catalog price
    amount_record_id: str = "user_amount"
    allow_deals: bool = True
    comparison_axis: Optional[str] = None
    comparison_entities: List[str] = field(default_factory=list)
    metric: str = "effective_price"
    spend_to_date: Dict[str, float] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    insufficient: Optional[str] = None                      # set when the purchase cannot be identified

    # Evidence bookkeeping. Retrieval is evidence DISCOVERY: retrieved records may confirm
    # or add to the deals considered, but may never redefine the product/merchant/category
    # the user is talking about (spec section 17).
    evidence_deal_ids: List[str] = field(default_factory=list)

    @property
    def product(self) -> Optional[Dict[str, Any]]:
        return self.products[0] if self.products else None

    def describe_subject(self) -> str:
        if self.products:
            return self.products[0].get("name", "this purchase")
        if self.category:
            return f"{self.category} spend"
        return "this purchase"


def build_purchase_frame(
    resolved: Any,
    state: Optional[Dict[str, Any]] = None,
    retrieved_records: Optional[List[Dict[str, Any]]] = None,
) -> PurchaseFrame:
    """
    Turn one ResolvedQuery (which already carries the merged conversation thread) into the
    candidate space to evaluate.

    Constraint precedence, highest first:
      1. entities the user named THIS turn
      2. an open comparison on the relevant axis
      3. context inherited from the conversation thread (already folded into `resolved`)
      4. the full catalog
    """
    state = state or {}
    cmp_axis = resolved.constraints.get("comparison_axis")
    cmp_entities = [e for e in resolved.constraints.get("comparison_entities", []) if e]

    # ---- products -----------------------------------------------------------------
    products: List[Dict[str, Any]] = list(resolved.products)
    if cmp_axis == "product" and len(cmp_entities) >= 2:
        widened = [DataIndex.get_product_by_id(pid) for pid in cmp_entities]
        widened = [p for p in widened if p]
        if len(widened) >= 2:
            products = widened

    category = resolved.category or (products[0].get("category") if products else None)

    # ---- merchants ----------------------------------------------------------------
    # A merchant comparison is a CONSTRAINT on the candidate space, not a separate mode:
    # it narrows which merchants get priced, and every one of them still gets its own full
    # eligibility + reward calculation below.
    merchants: List[str] = [m.lower() for m in resolved.merchants]
    if not merchants and cmp_axis == "merchant" and len(cmp_entities) >= 2:
        merchants = [m.lower() for m in cmp_entities]

    # A reward question ranks CARDS, so the purchase amount has to be the same for each one.
    # With a product in scope and no merchant named, that amount is the lowest the catalogue
    # lists — otherwise ranking by reward would quietly prefer the dearest merchant, because
    # a bigger bill earns more cashback.
    if (getattr(resolved, "objective", None) == "max_reward"
            and len(products) == 1 and not merchants):
        prices = products[0].get("prices") or {}
        if prices:
            merchants = [min(prices, key=lambda m: float(prices[m])).lower()]

    # ---- cards --------------------------------------------------------------------
    cards: List[Dict[str, Any]] = list(resolved.cards)
    if not cards and cmp_axis == "card" and len(cmp_entities) >= 2:
        narrowed = [DataIndex.get_card_by_id(cid) for cid in cmp_entities]
        cards = [c for c in narrowed if c]
    if not cards:
        cards = DataIndex.get_cards()

    # ---- amount -------------------------------------------------------------------
    # An amount the user states for the current calculation wins over the catalog price
    # (spec section 16), but only when it really is a figure for THIS calculation: a fresh
    # first-mention product query prices from the catalog, so a spec number that happens to
    # appear in the sentence can never become a spend figure.
    amount: Optional[float] = None
    amount_record_id = "user_amount"
    explicit_amount_turn = bool(resolved.is_hypothetical or resolved.is_followup)
    if resolved.purchase_amount is not None and (explicit_amount_turn or not products):
        amount = resolved.purchase_amount
        amount_record_id = "user_purchase_amount"
    elif resolved.reward_spend is not None and not products:
        amount = resolved.reward_spend
        amount_record_id = "user_reward_spend"
    elif not products and resolved.amount is not None:
        amount = resolved.amount
        amount_record_id = "user_amount"
    elif not products and state.get("budget") is not None:
        amount = float(state["budget"])
        amount_record_id = "session_budget"

    # ---- deals allowed ------------------------------------------------------------
    # A pure reward question ("how much cashback would I earn on X groceries with card Y")
    # asks what the CARD pays, not what merchant coupon could be stacked on top. Attaching
    # an unrequested merchant deal would answer a question the user did not ask
    # (spec section 19).
    objective = getattr(resolved, "objective", "min_effective_price")

    # A merchant promotion may only enter a REWARD question when the conversation has
    # actually put a merchant or a deal in play — the user named one, or an open comparison
    # is about one. Asking which card pays best is not a request to go find a coupon, and
    # attaching one silently answers a different question. Price and discount objectives
    # are about the purchase itself, so they always consider the offers.
    merchant_in_play = bool(
        resolved.merchants or resolved.deals or cmp_axis in ("merchant", "deal")
    )
    allow_deals = objective != "max_reward" or merchant_in_play

    # Rank by what the user is actually optimising for (spec: ranking follows the objective).
    metric = {"max_reward": "reward", "max_discount": "discount"}.get(objective, "effective_price")

    # ---- groundedness -------------------------------------------------------------
    # A budget is not a product and an amount is not an intent (spec sections 18, 28).
    insufficient = None
    if not products and not category and not merchants and not resolved.deals and not resolved.cards:
        insufficient = "purchase_unknown"
    elif not products and amount is None and not resolved.deals:
        insufficient = "amount_unknown"

    frame = PurchaseFrame(
        products=products,
        category=category,
        merchants=merchants,
        cards=cards,
        amount=amount,
        amount_record_id=amount_record_id,
        allow_deals=allow_deals,
        comparison_axis=cmp_axis,
        comparison_entities=cmp_entities,
        metric=metric,
        spend_to_date=dict(resolved.constraints.get("spend_to_date") or {}),
        constraints=dict(resolved.constraints),
        insufficient=insufficient,
    )

    if retrieved_records:
        frame.evidence_deal_ids = [
            r.get("deal_id") for r in retrieved_records if isinstance(r, dict) and r.get("deal_id")
        ]

    return frame


# ------------------------------------------------------------------------ base prices

def build_base_prices(frame: PurchaseFrame) -> List[BasePrice]:
    """
    Every priced starting point the frame admits.

    For a product: one per (product, merchant) pair present in the catalog, narrowed to the
    frame's merchants when it has any. For an amount-only calculation: the single figure the
    user actually stated. Nothing here is ever synthesized.
    """
    prices: List[BasePrice] = []

    if frame.products and frame.amount is None:
        allowed = set(frame.merchants) if frame.merchants else None
        for p in frame.products:
            pid = p["product_id"]
            for merchant, price in (p.get("prices") or {}).items():
                if allowed is not None and merchant.lower() not in allowed:
                    continue
                prices.append(BasePrice(
                    value=Value(
                        amount=float(price),
                        provenance=Provenance.SOURCE,
                        record_id=f"{pid}_{merchant.lower()}",
                    ),
                    merchant=merchant.lower(),
                    product_id=pid,
                ))
        return prices

    if frame.amount is not None:
        # A stated figure applies at whichever merchants are in scope; when none are named
        # it is a merchant-less calculation (e.g. "cashback on RS 8,000 of groceries").
        value = Value(
            amount=float(frame.amount),
            provenance=Provenance.SOURCE,
            record_id=frame.amount_record_id,
        )
        product_id = frame.product["product_id"] if frame.product else None
        if frame.merchants:
            for m in frame.merchants:
                prices.append(BasePrice(value=value, merchant=m, product_id=product_id))
        else:
            prices.append(BasePrice(value=value, merchant=None, product_id=product_id))

    return prices


def eligible_deals(frame: PurchaseFrame) -> List[Dict[str, Any]]:
    """
    The deal records the frame admits, filtered through the canonical eligibility engine.

    Deliberately NOT scoped to a single merchant: a merchant comparison needs each
    merchant's own deals, and `enumerate_options` re-checks eligibility per combination
    anyway. Narrowing here is what previously let one merchant's coupon decide the whole
    comparison.
    """
    if not frame.allow_deals:
        return []
    if frame.constraints.get("named_deal_ids"):
        return filter_deals(deal_ids=list(frame.constraints["named_deal_ids"]))
    # An open DEAL comparison keeps constraining the pool on later turns ("which is
    # better?"), so the follow-up ranks the deals under discussion rather than every deal
    # the category happens to contain (spec section 11).
    if frame.comparison_axis == "deal" and len(frame.comparison_entities) >= 2:
        return filter_deals(deal_ids=list(frame.comparison_entities))

    # A product comparison spans products that may sit in different categories, so the deal
    # pool is the union of each product's own eligible deals. `enumerate_options` re-checks
    # every deal against the specific product it is being applied to, so widening here
    # cannot let one product's coupon leak onto another.
    if len(frame.products) > 1:
        pool: List[Dict[str, Any]] = []
        seen = set()
        for p in frame.products:
            for d in filter_deals(category=p.get("category"), product=p):
                if d["deal_id"] not in seen:
                    seen.add(d["deal_id"])
                    pool.append(d)
        return pool

    return filter_deals(category=frame.category, product=frame.product)


def build_payment_options(frame: PurchaseFrame) -> List[PaymentOption]:
    """
    Every valid (base price x deal x card) combination the frame admits, each costed by the
    one deterministic engine and ranked by the frame's metric.
    """
    base_prices = build_base_prices(frame)
    if not base_prices:
        return []

    # `named_merchant`/`preferred_card` in constraints would re-narrow inside
    # enumerate_options; the frame has already made those decisions explicitly, so pass a
    # constraint view without them to keep a single point of narrowing.
    passthrough = {
        k: v for k, v in frame.constraints.items()
        if k not in ("named_merchant", "merchant", "preferred_card", "card")
    }

    return enumerate_options(
        base_prices=base_prices,
        deals=eligible_deals(frame),
        cards=frame.cards,
        category=frame.category or "general",
        spend_to_date=frame.spend_to_date,
        constraints=passthrough,
        product=frame.product,
        metric=frame.metric,
    )


# -------------------------------------------------------------------------- candidates

@dataclass
class RejectedDeal:
    """One visible deal that the candidate space declined, and the reason it declined it."""
    deal_id: str
    headline: str                       # the offer as the deal record states it
    reason: str                         # why it could not be used here
    values: List[Value] = field(default_factory=list)   # SOURCE backing for every figure quoted


# Which rejection to surface when a deal fails for different reasons against different
# cards or merchants. A minimum-spend shortfall is the near miss a shopper can act on, so it
# is reported ahead of the structural mismatches; a card mismatch is reported last because it
# is what every OTHER card reports about a card-locked deal, and is rarely the real blocker.
_REASON_PRECEDENCE = [
    "MIN_SPEND_NOT_MET",
    "PRODUCT_MISMATCH",
    "CATEGORY_MISMATCH",
    "MERCHANT_MISMATCH",
    "CARD_MISMATCH",
]


def _deal_headline(deal: Dict[str, Any]) -> Tuple[str, List[Value]]:
    """The offer restated from the deal record's own structured fields."""
    deal_id = deal["deal_id"]
    values: List[Value] = []
    max_d = deal.get("max_discount")
    if deal.get("discount_type") == "percentage":
        pct = RewardEngine.display_percentage(deal)
        text = f"{pct:g}% off"
        if max_d is not None:
            text += f", capped at RS {float(max_d):,.0f}"
            values.append(Value(amount=float(max_d), provenance=Provenance.SOURCE,
                                record_id=f"{deal_id}_max_discount"))
    else:
        amount = float(deal.get("discount_value", 0))
        text = f"flat RS {amount:,.0f} off"
        values.append(Value(amount=amount, provenance=Provenance.SOURCE,
                            record_id=f"{deal_id}_flat_value"))
    return text, values


def _rejection_sentence(deal: Dict[str, Any], reason: str, frame: PurchaseFrame,
                        reference_price: float) -> Tuple[Optional[str], List[Value]]:
    """The reason in the user's terms, quoting only fields the deal record actually states."""
    deal_id = deal["deal_id"]
    if reason == "MIN_SPEND_NOT_MET":
        min_spend = float(deal.get("min_spend", 0))
        return (
            f"needs a minimum spend of RS {min_spend:,.0f}; this purchase is RS {reference_price:,.0f}",
            [Value(amount=min_spend, provenance=Provenance.SOURCE, record_id=f"{deal_id}_min_spend")],
        )
    if reason == "CARD_MISMATCH" and deal.get("card_specific"):
        return f"is only valid with the {deal['card_specific']} card", []
    if reason == "MERCHANT_MISMATCH" and deal.get("merchant"):
        return f"applies only at {merchant_display(deal['merchant'])}", []
    if reason == "CATEGORY_MISMATCH" and deal.get("category"):
        return f"applies to {deal['category']}, not {frame.category}", []
    if reason == "PRODUCT_MISMATCH":
        return "does not cover this product", []
    return None, []


def explain_rejected_visible(
    frame: PurchaseFrame,
    retrieved_deal_ids: List[str],
    applied_discount: float,
    applied_deal_id: Optional[str],
    reference_price: float,
    limit: int = 3,
) -> List[RejectedDeal]:
    """
    Explain the visible deals that could have looked better but could not be used.

    STRICTLY A REPORT. It reads the frame and re-runs the same canonical eligibility engine
    the candidate space already ran; it constructs no options, admits nothing, and returns no
    figure that is not a field of a deal record or a price already in the trace. Whatever it
    returns has, by construction, been rejected — so nothing here can turn a displayed deal
    into a financial candidate.

    A rejected deal is only worth mentioning when it clears two bars:

      * VISIBLE — it was retrieved, so the user can see it and reasonably wonder about it.
      * MATERIAL — the discount it advertises would have matched or beaten the one actually
        applied. A deal that could never have won explains nothing, so it stays out.
    """
    if not retrieved_deal_ids or not frame.products:
        return []

    base_prices = build_base_prices(frame)
    if not base_prices:
        return []

    out: List[RejectedDeal] = []
    seen = set()
    for deal_id in retrieved_deal_ids:
        if len(out) >= limit or deal_id in seen or deal_id == applied_deal_id:
            continue
        seen.add(deal_id)
        deal = DataIndex.get_deal_by_id(deal_id)
        if not deal:
            continue

        # MATERIAL? Best case this deal could have produced on the purchase we priced.
        potential = RewardEngine.deal_discount(deal, reference_price)
        if potential <= 0 or potential < applied_discount - 0.01:
            continue

        # Re-run the canonical eligibility engine over the same candidate space.
        reasons: List[str] = []
        for bp in base_prices:
            for card in frame.cards:
                ok, why = is_deal_eligible(
                    deal=deal,
                    product=DataIndex.get_product_by_id(bp.product_id) if bp.product_id else frame.product,
                    merchant=bp.merchant,
                    purchase_amount=bp.value.amount,
                    card=card,
                    category=frame.category,
                )
                if ok:
                    reasons = []          # it IS usable somewhere; not a rejection to explain
                    break
                if why:
                    reasons.append(why)
            else:
                continue
            break
        if not reasons:
            continue

        binding = next((r for r in _REASON_PRECEDENCE if r in reasons), reasons[0])
        sentence, reason_values = _rejection_sentence(deal, binding, frame, reference_price)
        if not sentence:
            continue
        headline, headline_values = _deal_headline(deal)
        out.append(RejectedDeal(deal_id=deal_id, headline=headline, reason=sentence,
                                values=headline_values + reason_values))

    return out


@dataclass
class FinancialCandidate:
    """One alternative under comparison, with its own fully-derived financial outcome."""
    key: str
    label: str
    option: PaymentOption

    @property
    def effective_price(self) -> float:
        return self.option.effective_price.amount

    def row(self) -> Dict[str, Any]:
        """Structured memo row — every figure already grounded by the reward engine."""
        o = self.option
        return {
            "key": self.key,
            "label": self.label,
            "merchant": o.merchant,
            "product_id": o.product_id,
            "card_id": o.card_id,
            "deal_id": o.discount_source_id,
            "base_price": o.base_price.amount,
            "discount": o.discount_applied.amount,
            "price_after_discount": o.price_after_discount.amount,
            "reward": o.reward_earned.amount,
            "effective_price": o.effective_price.amount,
            "cap_hit": bool(o.cap_hit),
            "cap_explanation": o.cap_explanation,
        }


def _axis_key(option: PaymentOption, axis: str) -> Optional[str]:
    if axis == "merchant":
        return (option.merchant or "").lower() or None
    if axis == "card":
        return option.card_id
    if axis == "deal":
        return option.discount_source_id or "no_deal"
    if axis == "product":
        return option.product_id
    return None


def _axis_label(key: str, axis: str) -> str:
    if axis == "merchant":
        return merchant_display(key)
    if axis == "card":
        card = DataIndex.get_card_by_id(key) or {}
        return card.get("name", key)
    if axis == "deal":
        if key == "no_deal":
            return "No deal applied"
        deal = DataIndex.get_deal_by_id(key) or {}
        return deal.get("title", key)
    if axis == "product":
        prod = DataIndex.get_product_by_id(key) or {}
        return prod.get("name", key)
    return key


def build_candidates(
    frame: PurchaseFrame,
    axis: str,
    options: Optional[List[PaymentOption]] = None,
) -> List[FinancialCandidate]:
    """
    Group the frame's payment options along `axis` and keep the best option for each
    distinct entity, then rank the survivors by the frame's metric.

    This is what makes a comparison a comparison: every entity on the axis gets its OWN
    best-case financial outcome, so the loser is a real computed alternative rather than an
    entity that was retrieved and then dropped.
    """
    if options is None:
        options = build_payment_options(frame)

    grouped: Dict[str, PaymentOption] = {}
    for opt in options:
        key = _axis_key(opt, axis)
        if key is None:
            continue
        current = grouped.get(key)
        if current is None or _is_better(opt, current, frame.metric):
            grouped[key] = opt

    # When the comparison names entities explicitly, preserve the user's ordering for any
    # that produced a candidate — the ranking below still decides the winner.
    ordered_keys = list(grouped.keys())
    if frame.comparison_axis == axis and frame.comparison_entities:
        wanted = [e.lower() for e in frame.comparison_entities]
        ordered_keys.sort(key=lambda k: wanted.index(k.lower()) if k.lower() in wanted else len(wanted))

    candidates = [
        FinancialCandidate(key=k, label=_axis_label(k, axis), option=grouped[k])
        for k in ordered_keys
    ]
    return rank_candidates(candidates, frame.metric)


def _is_better(a: PaymentOption, b: PaymentOption, metric: str) -> bool:
    extractor, descending = RewardEngine.METRICS.get(metric, RewardEngine.METRICS["effective_price"])
    va, vb = extractor(a), extractor(b)
    if va != vb:
        return va > vb if descending else va < vb
    return (a.card_id or "zzzzz") < (b.card_id or "zzzzz")


def rank_candidates(candidates: List[FinancialCandidate], metric: str = "effective_price") -> List[FinancialCandidate]:
    extractor, descending = RewardEngine.METRICS.get(metric, RewardEngine.METRICS["effective_price"])

    def key(c: FinancialCandidate):
        val = extractor(c.option)
        return (-val if descending else val, c.option.card_id or "zzzzz", c.key)

    return sorted(candidates, key=key)


def evidence_report(frame: PurchaseFrame) -> Dict[str, List[str]]:
    """
    Which retrieved records the candidate space actually admitted, and which it did not.

    Retrieval discovers evidence; the frame decides what is in scope. A retrieved deal is
    admitted only if it survives the canonical eligibility engine against the product,
    merchant and category the CONVERSATION established — never the other way round. This
    report makes that direction visible (invariant B) and is reported, not acted on.
    """
    if not frame.evidence_deal_ids:
        return {"admitted": [], "rejected": []}
    admitted_ids = {d["deal_id"] for d in eligible_deals(frame)}
    admitted = [d for d in frame.evidence_deal_ids if d in admitted_ids]
    rejected = [d for d in frame.evidence_deal_ids if d not in admitted_ids]
    return {"admitted": admitted, "rejected": rejected}


def merged_evidence(candidates: List[FinancialCandidate]) -> Tuple[List[str], List[Value]]:
    """Union of the citations and provenance traces of every candidate that was ranked."""
    citations: List[str] = []
    trace: List[Value] = []
    seen_trace = set()
    for c in candidates:
        for cit in c.option.citations:
            if cit not in citations:
                citations.append(cit)
        for v in c.option.trace:
            sig = (v.record_id, round(float(v.amount), 2))
            if sig not in seen_trace:
                seen_trace.add(sig)
                trace.append(v)
    return citations, trace
