from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from app.rag.index import DataIndex
from app.core.provenance import Value, Provenance
from app.core.reward_engine import RewardEngine, PaymentOption


@dataclass(frozen=True)
class BasePrice:
    """
    One priced starting point for a calculation, with the merchant it belongs to carried
    explicitly.

    Previously the merchant was recovered by splitting `Value.record_id` on "_", which
    silently mangled every multi-word merchant in the catalog ("Reliance Digital" became
    "digital", "Country Delight" became "delight"). Merchant identity is the grouping key
    for merchant comparisons and the label on every answer, so it is now a real field.
    """
    value: Value
    merchant: Optional[str] = None
    product_id: Optional[str] = None


def as_base_prices(items: List[Union[Value, "BasePrice"]]) -> List["BasePrice"]:
    """Accept either raw Values (legacy callers) or BasePrices, normalizing to BasePrice."""
    out: List[BasePrice] = []
    for it in items:
        if isinstance(it, BasePrice):
            out.append(it)
        else:
            out.append(BasePrice(value=it, merchant=None, product_id=None))
    return out

def is_deal_eligible(
    deal: Dict[str, Any],
    product: Optional[Dict[str, Any]] = None,
    merchant: Optional[str] = None,
    purchase_amount: Optional[float] = None,
    card: Optional[Dict[str, Any]] = None,
    category: Optional[str] = None
) -> Tuple[bool, Optional[str]]:
    """
    Deterministic Deal Eligibility Evaluator.
    Evaluates:
      - Merchant compatibility
      - Category compatibility
      - Product / subcategory scope applicability
      - Card compatibility
      - Minimum spend thresholds
    Returns (is_eligible, rejection_reason)
    """
    # 1. Merchant Check
    if merchant:
        d_merch = (deal.get("merchant") or "").lower()
        m_low = merchant.lower()
        if d_merch not in ["", "all", "any"] and d_merch != m_low and m_low not in d_merch and d_merch not in m_low:
            return False, "MERCHANT_MISMATCH"

    # 2. Category Check
    req_cat = (category or (product.get("category") if product else None) or "").lower()
    d_cat = (deal.get("category") or "").lower()
    if req_cat and d_cat and d_cat not in ["", "all", "any"] and d_cat != req_cat:
        return False, "CATEGORY_MISMATCH"

    # 3. Product / Subcategory Scope Check.
    # The `subcategory` field in this dataset is a coarse merchandising bucket, not a
    # strict taxonomic partition — e.g. deal_010's own terms say "Valid across Myntra
    # apparel and shoes" despite being tagged subcategory=apparel. So an EQUAL subcategory
    # is treated as a confirmed match (skips the free-text heuristics below entirely —
    # structured field wins), but an unequal subcategory is NOT a hard reject on its own;
    # the free-text heuristics in 3b (keyed off the deal's actual title) make that call,
    # since they target genuinely distinct product types (e.g. "Home Appliances" vs an
    # e-reader) rather than soft merchandising categories.
    d_subcat = (deal.get("subcategory") or "").lower()
    p_subcat = (product.get("subcategory") or "").lower() if product else ""
    subcat_confirmed_match = bool(
        product and d_subcat and p_subcat and d_subcat == p_subcat
        and d_subcat not in ["", "all", "any", "general", req_cat]
    )

    # 3b. Free-text scope heuristics — fallback ONLY when the structured subcategory
    # fields did not already give a conclusive answer (spec: structured fields first,
    # free-text interpretation only when structured data is insufficient).
    if product and not subcat_confirmed_match:
        p_name = (product.get("name") or "").lower()
        p_desc = (product.get("description") or "").lower()
        p_text = f"{p_name} {p_desc}"
        d_title = (deal.get("title") or "").lower()

        if any(w in d_title for w in ["appliance", "appliances", "home appliance", "refrigerator", "washing machine"]):
            if not any(w in p_text for w in ["appliance", "appliances", "refrigerator", "washing machine"]):
                return False, "PRODUCT_MISMATCH"

        if "laptop" in d_title:
            if not any(w in p_text for w in ["laptop", "macbook", "notebook"]):
                return False, "PRODUCT_MISMATCH"

        if any(w in d_title for w in ["smartphone", "smartphones", "mobile phone"]):
            if not any(w in p_text for w in ["smartphone", "iphone", "mobile"]):
                return False, "PRODUCT_MISMATCH"

        if "apple store student discount" in d_title:
            if not any(w in p_text for w in ["macbook", "ipad"]):
                return False, "PRODUCT_MISMATCH"

        if any(w in d_title for w in ["shoe", "shoes", "footwear"]):
            if not any(w in p_text for w in ["shoe", "shoes", "footwear"]):
                return False, "PRODUCT_MISMATCH"

        if "flight" in d_title:
            if not any(w in p_text for w in ["flight", "ticket", "airline"]):
                return False, "PRODUCT_MISMATCH"

        if any(w in d_title for w in ["hotel", "resort", "stay"]):
            if not any(w in p_text for w in ["hotel", "resort", "stay"]):
                return False, "PRODUCT_MISMATCH"

        if "milk" in d_title:
            if "milk" not in p_text:
                return False, "PRODUCT_MISMATCH"

    # 4. Card Compatibility Check — only enforced when the caller actually supplied a
    # candidate card. A caller that omits `card` (e.g. a deal-discovery listing that isn't
    # scoped to one card yet) is asking "does this deal exist", not "does my card qualify",
    # so a card-restricted deal must still surface rather than being silently dropped.
    card_sp = (deal.get("card_specific") or "").lower()
    if card_sp and card is not None:
        card_id = (card.get("card_id") or "").lower()
        if not card_id or card_id != card_sp:
            return False, "CARD_MISMATCH"

    # 5. Minimum Spend Threshold Check
    if purchase_amount is not None:
        min_s = float(deal.get("min_spend", 0))
        if purchase_amount < min_s:
            return False, "MIN_SPEND_NOT_MET"

    return True, None


def filter_deals(
    category: Optional[str] = None,
    merchant: Optional[str] = None,
    card: Optional[str] = None,
    min_spend_lte: Optional[float] = None,
    discount_type: Optional[str] = None,
    deal_ids: Optional[List[str]] = None,
    product: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Primitive 1: Filter deals from the authoritative dataset.
    """
    deals = DataIndex.get_deals()
    matched = []
    for d in deals:
        if deal_ids and d.get("deal_id") not in deal_ids:
            continue
        eligible, _ = is_deal_eligible(
            deal=d,
            product=product,
            merchant=merchant,
            category=category,
            purchase_amount=min_spend_lte
        )
        if not eligible:
            continue
        if card and d.get("card_specific"):
            d_card = d["card_specific"].lower()
            if d_card != card.lower():
                continue
        if discount_type and d.get("discount_type") != discount_type:
            continue
        matched.append(d)
    return matched


def filter_products(
    category: Optional[str] = None,
    merchant: Optional[str] = None,
    name_tokens: Optional[List[str]] = None,
    product_ids: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Primitive 2: Filter products from the authoritative dataset.
    """
    products = DataIndex.get_products()
    matched = []
    for p in products:
        if product_ids and p.get("product_id") not in product_ids:
            continue
        if category and p.get("category"):
            if p["category"].lower() != category.lower():
                continue
        if merchant and p.get("prices"):
            if merchant.lower() not in [m.lower() for m in p["prices"].keys()]:
                continue
        if name_tokens:
            p_name = p.get("name", "").lower()
            if not all(t.lower() in p_name for t in name_tokens):
                continue
        matched.append(p)
    return matched


def aggregate_records(
    records: List[Dict[str, Any]],
    field: str,
    op: str = "max",
    amount: Optional[float] = None
) -> Tuple[float, Optional[Dict[str, Any]]]:
    """
    Primitive 3: Compute max/min/count/sum over a set of records.
    """
    if not records:
        return 0.0, None

    if op == "count":
        return float(len(records)), records[0] if records else None

    best_val = -float("inf") if op == "max" else float("inf")
    best_record = None

    for r in records:
        val = 0.0
        if field == "discount":
            if r.get("discount_type") == "percentage" and amount is None:
                # No spend to apply the rate to. Ranking by the RATE would compare a
                # percentage against a flat deal's rupees — 5 against 250 — and pick an
                # arbitrary winner. The comparable quantity in the same units is the most
                # the deal can ever pay out, which is its cap.
                cap = r.get("max_discount")
                val = float(cap) if cap is not None else RewardEngine.display_percentage(r)
            else:
                val = RewardEngine.deal_discount(r, amount if amount is not None else 0.0)
        elif field in r:
            try:
                val = float(r[field])
            except (ValueError, TypeError):
                continue
        else:
            continue

        if op == "max" and val > best_val:
            best_val = val
            best_record = r
        elif op == "min" and val < best_val:
            best_val = val
            best_record = r

    if best_record is None:
        return 0.0, records[0] if records else None
    return best_val, best_record


def enumerate_options(
    base_prices: List[Union[Value, BasePrice]],
    deals: List[Dict[str, Any]],
    cards: List[Dict[str, Any]],
    category: str = "general",
    spend_to_date: Optional[Dict[str, float]] = None,
    constraints: Optional[Dict[str, Any]] = None,
    product: Optional[Dict[str, Any]] = None,
    metric: str = "effective_price"
) -> List[PaymentOption]:
    """
    Primitive 4: Exhaustive cross-product evaluation of (base_price, deal, card).

    THE one place where payment options come into existence. Every operation — optimize,
    compare, compute, follow-up recalculation — reaches the arithmetic through here, so no
    handler can grow its own money math (spec section 30). Eligibility is likewise
    delegated to the single canonical `is_deal_eligible`.

    The caller decides WHICH combinations to enumerate (the candidate space); this function
    decides only what each of them is worth.
    """
    if constraints is None:
        constraints = {}

    req_card = constraints.get("preferred_card") or constraints.get("card")
    req_merchant = constraints.get("named_merchant") or constraints.get("merchant")

    options: List[PaymentOption] = []

    for bp in as_base_prices(base_prices):
        bp_merch = bp.merchant.lower() if bp.merchant else None
        # Legacy call sites still pass bare Values; recover the merchant from the record id
        # only in that case, and only for the single-token form it can represent faithfully.
        if bp_merch is None and bp.value.record_id and bp.value.record_id.startswith("prod_"):
            parts = bp.value.record_id.split("_")
            if len(parts) >= 2:
                bp_merch = parts[-1].lower()

        target_merchant = req_merchant.lower() if req_merchant else bp_merch
        if req_merchant and bp_merch and bp_merch != req_merchant.lower():
            continue

        # Candidate cards
        cand_cards = cards
        if req_card:
            filtered_c = [c for c in cards if c.get("card_id", "").lower() == req_card.lower()]
            if filtered_c:
                cand_cards = filtered_c

        pid = bp.product_id or (product or {}).get("product_id")

        # Eligibility and reward rates are properties of the PRODUCT being priced, so a
        # candidate space spanning several products (a product comparison) must evaluate
        # each base price against its own product and category — not against whichever
        # product happened to be named first.
        bp_product = DataIndex.get_product_by_id(pid) if pid else None
        if bp_product is None:
            bp_product = product
        bp_category = (bp_product.get("category") if bp_product else None) or category

        # Evaluate all combinations
        for c in cand_cards:
            for d in deals:
                eligible, _ = is_deal_eligible(
                    deal=d,
                    product=bp_product,
                    merchant=target_merchant,
                    purchase_amount=bp.value.amount,
                    card=c,
                    category=bp_category
                )
                if eligible:
                    options.append(
                        RewardEngine.calculate_payment_option(
                            base_price=bp.value,
                            deal=d,
                            card=c,
                            category=bp_category,
                            spend_to_date=spend_to_date,
                            merchant=bp.merchant,
                            product_id=pid
                        )
                    )
            # Baseline (no deal)
            options.append(
                RewardEngine.calculate_payment_option(
                    base_price=bp.value,
                    deal=None,
                    card=c,
                    category=bp_category,
                    spend_to_date=spend_to_date,
                    merchant=bp.merchant,
                    product_id=pid
                )
            )

    return RewardEngine.rank_options(options, metric=metric)


def explain_comparison(
    option_a: PaymentOption,
    option_b: PaymentOption
) -> str:
    """
    Primitive 5: Explains why option_a differs from option_b in terms of deltas.
    """
    eff_a = option_a.effective_price.amount
    eff_b = option_b.effective_price.amount
    diff = abs(eff_a - eff_b)
    
    card_a = option_a.card_id or "Card A"
    card_b = option_b.card_id or "Card B"
    
    if abs(eff_a - eff_b) < 0.01:
        return f"Both {card_a} and {card_b} result in the same effective price of RS {eff_a:.0f}."
    elif eff_a < eff_b:
        return f"{card_a} is cheaper than {card_b} by RS {diff:.0f} (RS {eff_a:.0f} vs RS {eff_b:.0f})."
    else:
        return f"{card_b} is cheaper than {card_a} by RS {diff:.0f} (RS {eff_b:.0f} vs RS {eff_a:.0f})."
