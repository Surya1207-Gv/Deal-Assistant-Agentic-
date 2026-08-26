from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from app.core.provenance import Value, Provenance

@dataclass
class PaymentOption:
    base_price: Value
    discount_applied: Value
    discount_source_id: Optional[str]
    price_after_discount: Value
    card_id: Optional[str]
    reward_earned: Value
    reward_rate_applied: float
    raw_reward: Value
    cap_hit: bool
    cap_explanation: Optional[str]
    reward_lost_to_cap: Value
    effective_price: Value
    citations: List[str]
    trace: List[Value] = field(default_factory=list)
    # The merchant this option is priced at, carried explicitly rather than re-derived by
    # parsing `base_price.record_id`. Multi-word merchants ("Reliance Digital",
    # "Country Delight", "Booking.com") made that parsing lossy, and merchant identity is
    # the grouping key for merchant comparisons — it has to be exact.
    merchant: Optional[str] = None
    product_id: Optional[str] = None

class RewardEngine:
    """
    Deterministic financial calculations engine.
    Computes discounts, reward points, card cashback, caps, headroom, and effective price.
    Appends EVERY monetary number produced to PaymentOption.trace as a tagged Value object.
    """

    @staticmethod
    def deal_discount(deal: Dict[str, Any], amount: float) -> float:
        """
        The discount a deal yields on `amount`, from the deal record's own fields.

        THE single implementation of this formula. It previously existed three times — here,
        in the aggregate primitive, and in the lookup handler — which is exactly the kind of
        drift spec section 30 forbids: three copies can disagree, and two of them applied a
        different `max_discount` default than the engine that produced the actual answer.

        Purely arithmetic: whether the deal APPLIES at all (merchant, card, category,
        product scope, minimum spend) is decided by `is_deal_eligible`, never here.
        """
        if not deal:
            return 0.0
        disc_val = float(deal.get("discount_value", 0.0) or 0.0)
        max_disc = deal.get("max_discount")
        max_disc = float(max_disc) if max_disc is not None else float("inf")

        if deal.get("discount_type") == "percentage":
            # Rates are stored as fractions (0.05); tolerate a whole-percent spelling (5).
            rate = disc_val if disc_val <= 1.0 else disc_val / 100.0
            return min(amount * rate, max_disc)
        return min(disc_val, max_disc)

    @staticmethod
    def display_percentage(deal: Dict[str, Any]) -> float:
        """A percentage deal's rate rendered for display, from the same normalization."""
        d_val = float(deal.get("discount_value", 0.0) or 0.0)
        return d_val * 100 if d_val <= 1.0 else d_val

    @staticmethod
    def calculate_payment_option(
        base_price: Value,
        deal: Optional[Dict[str, Any]] = None,
        card: Optional[Dict[str, Any]] = None,
        category: str = "general",
        spend_to_date: Optional[Dict[str, float]] = None,
        merchant: Optional[str] = None,
        product_id: Optional[str] = None
    ) -> PaymentOption:
        if spend_to_date is None:
            spend_to_date = {}

        trace: List[Value] = [base_price]
        citations: List[str] = []

        amount = base_price.amount
        card_id = card.get("card_id", "default_card") if card else None

        # 1. Coupon Discount Stacking
        discount_amount = 0.0
        discount_source_id = None

        if deal:
            deal_id = deal.get("deal_id", "deal_unknown")
            min_spend = deal.get("min_spend", 0)
            card_sp = deal.get("card_specific")
            is_card_eligible = (not card_sp) or (card_id and card_sp.lower() == card_id.lower())

            if is_card_eligible and amount >= min_spend:
                discount_amount = RewardEngine.deal_discount(deal, amount)

                if discount_amount > 0:
                    discount_source_id = deal_id
                    citations.append(deal_id)
                    max_d = deal.get("max_discount")
                    if max_d is not None and max_d < float("inf"):
                        trace.append(Value(amount=float(max_d), provenance=Provenance.SOURCE, record_id=f"{deal_id}_max_discount"))
                    if min_spend > 0:
                        trace.append(Value(amount=float(min_spend), provenance=Provenance.SOURCE, record_id=f"{deal_id}_min_spend"))

        discount_applied = Value(
            amount=round(discount_amount, 2),
            provenance=Provenance.DERIVED,
            record_id="discount_applied",
            formula=f"Discount of {discount_amount} applied",
            inputs=[base_price.record_id or "base_price"]
        )
        trace.append(discount_applied)

        # 2. Post-discount Price
        price_after_disc = max(0.0, amount - discount_amount)
        price_after_discount_val = Value(
            amount=round(price_after_disc, 2),
            provenance=Provenance.DERIVED,
            record_id="price_after_discount",
            formula=f"{amount} - {discount_amount}",
            inputs=["base_price", "discount_applied"]
        )
        trace.append(price_after_discount_val)

        # 3. Card Reward & Cap Calculation
        reward_amount = 0.0
        raw_reward_amount = 0.0
        reward_lost_amount = 0.0
        reward_rate_applied = 0.0
        cap_hit = False
        cap_explanation: Optional[str] = None
        card_id = None

        if card:
            card_id = card.get("card_id", "card_unknown")
            citations.append(card_id)

            card_min_spend = card.get("min_spend", 0)
            if price_after_disc < card_min_spend:
                raw_reward_val = Value(amount=0.0, provenance=Provenance.DERIVED, record_id="raw_reward")
                reward_earned = Value(amount=0.0, provenance=Provenance.DERIVED, record_id="reward_earned")
                reward_lost_val = Value(amount=0.0, provenance=Provenance.DERIVED, record_id="reward_lost_to_cap")
                trace.extend([raw_reward_val, reward_earned, reward_lost_val])
                cap_hit = False
                cap_explanation = f"Minimum spend requirement of RS {card_min_spend} not met for {card.get('name', card_id)}"
            else:
                base_rate = card.get("base_rate", 0.01)
                category_multipliers = card.get("category_multipliers", {})

                multiplier = category_multipliers.get(category, 1.0)
                reward_rate_applied = base_rate * multiplier
                raw_reward_amount = price_after_disc * reward_rate_applied

                caps = card.get("caps", {})
                monthly_cap = caps.get("monthly_cashback_cap", float("inf"))
                category_caps = caps.get("category_caps", {})
                cat_cap = category_caps.get(category, float("inf"))

                prior_monthly = spend_to_date.get(f"{card_id}_monthly", 0.0)
                prior_cat = spend_to_date.get(f"{card_id}_{category}", 0.0)

                monthly_headroom = max(0.0, monthly_cap - prior_monthly)
                cat_headroom = max(0.0, cat_cap - prior_cat)

                effective_cap_headroom = min(monthly_headroom, cat_headroom)

                # Append all intermediate cap & rate values into trace
                raw_reward_val = Value(amount=round(raw_reward_amount, 2), provenance=Provenance.DERIVED, record_id="raw_reward")
                raw_reward_rounded = Value(amount=round(raw_reward_amount, 0), provenance=Provenance.DERIVED, record_id="raw_reward_rounded")
                trace.extend([raw_reward_val, raw_reward_rounded])

                if cat_cap < float("inf"):
                    trace.append(Value(amount=float(cat_cap), provenance=Provenance.SOURCE, record_id=f"{card_id}_{category}_cap"))
                if monthly_cap < float("inf") and monthly_cap != 999999:
                    trace.append(Value(amount=float(monthly_cap), provenance=Provenance.SOURCE, record_id=f"{card_id}_monthly_cap"))
                if prior_cat > 0:
                    trace.append(Value(amount=float(prior_cat), provenance=Provenance.SOURCE, record_id=f"{card_id}_{category}_prior_spend"))
                if prior_monthly > 0:
                    trace.append(Value(amount=float(prior_monthly), provenance=Provenance.SOURCE, record_id=f"{card_id}_monthly_prior_spend"))
                if cat_headroom < float("inf"):
                    trace.append(Value(amount=round(cat_headroom, 2), provenance=Provenance.DERIVED, record_id=f"{card_id}_{category}_cat_headroom"))
                if monthly_headroom < float("inf"):
                    trace.append(Value(amount=round(monthly_headroom, 2), provenance=Provenance.DERIVED, record_id=f"{card_id}_monthly_headroom"))

                if raw_reward_amount > effective_cap_headroom:
                    reward_amount = effective_cap_headroom
                    reward_lost_amount = raw_reward_amount - effective_cap_headroom
                    cap_hit = True

                    reasons = []
                    if cat_headroom < raw_reward_amount:
                        reasons.append(f"category cap RS {cat_cap:.0f} (RS {prior_cat:.0f} prior spend, headroom RS {cat_headroom:.0f})")
                    if monthly_headroom < raw_reward_amount:
                        reasons.append(f"monthly cashback cap RS {monthly_cap:.0f} (RS {prior_monthly:.0f} prior spend, headroom RS {monthly_headroom:.0f})")

                    cap_explanation = f"Reward capped at RS {reward_amount:.0f} due to " + " and ".join(reasons)
                else:
                    reward_amount = raw_reward_amount
                    reward_lost_amount = 0.0
                    cap_hit = False

                reward_earned = Value(
                    amount=round(reward_amount, 2),
                    provenance=Provenance.DERIVED,
                    record_id="reward_earned",
                    formula=f"{price_after_disc} * {reward_rate_applied:.4f}",
                    inputs=["price_after_discount"]
                )
                trace.append(reward_earned)

                reward_lost_val = Value(
                    amount=round(reward_lost_amount, 2),
                    provenance=Provenance.DERIVED,
                    record_id="reward_lost_to_cap",
                    formula=f"{raw_reward_amount} - {reward_amount}",
                    inputs=["raw_reward", "reward_earned"]
                )
                reward_lost_rounded = Value(
                    amount=round(reward_lost_amount, 0),
                    provenance=Provenance.DERIVED,
                    record_id="reward_lost_rounded"
                )
                trace.extend([reward_lost_val, reward_lost_rounded])
        else:
            raw_reward_val = Value(amount=0.0, provenance=Provenance.DERIVED, record_id="raw_reward")
            reward_earned = Value(amount=0.0, provenance=Provenance.DERIVED, record_id="reward_earned")
            reward_lost_val = Value(amount=0.0, provenance=Provenance.DERIVED, record_id="reward_lost_to_cap")
            trace.extend([raw_reward_val, reward_earned, reward_lost_val])

        eff_price = max(0.0, price_after_disc - reward_amount)
        effective_price_val = Value(
            amount=round(eff_price, 2),
            provenance=Provenance.DERIVED,
            record_id="effective_price",
            formula=f"{price_after_disc} - {reward_amount}",
            inputs=["price_after_discount", "reward_earned"]
        )
        effective_price_rounded = Value(
            amount=round(eff_price, 0),
            provenance=Provenance.DERIVED,
            record_id="effective_price_rounded"
        )
        trace.extend([effective_price_val, effective_price_rounded])

        return PaymentOption(
            base_price=base_price,
            discount_applied=discount_applied,
            discount_source_id=discount_source_id,
            price_after_discount=price_after_discount_val,
            card_id=card_id,
            reward_earned=reward_earned,
            reward_rate_applied=reward_rate_applied,
            raw_reward=raw_reward_val,
            cap_hit=cap_hit,
            cap_explanation=cap_explanation,
            reward_lost_to_cap=reward_lost_val,
            effective_price=effective_price_val,
            citations=citations,
            trace=trace,
            merchant=merchant,
            product_id=product_id
        )

    # Ranking metrics. Not every question is "cheapest" (spec section 31): a cashback
    # question ranks by reward earned, a price question by effective final price. The
    # metric is chosen by the caller from the resolved operation; the arithmetic behind
    # every metric is the same PaymentOption produced above.
    METRICS = {
        # (extractor, descending?) — ties always broken by card_id for determinism.
        "effective_price": (lambda o: o.effective_price.amount, False),
        "reward": (lambda o: o.reward_earned.amount, True),
        "discount": (lambda o: o.discount_applied.amount, True),
    }

    @classmethod
    def rank_options(cls, options: List[PaymentOption], metric: str = "effective_price") -> List[PaymentOption]:
        extractor, descending = cls.METRICS.get(metric, cls.METRICS["effective_price"])

        def sort_key(opt: PaymentOption):
            val = extractor(opt)
            cid = opt.card_id or "zzzzz"
            return (-val if descending else val, cid)

        return sorted(options, key=sort_key)
