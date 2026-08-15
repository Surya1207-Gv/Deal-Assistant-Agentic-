from __future__ import annotations
from typing import List, Dict, Any, Optional
from app.rag.index import DataIndex
from app.core.provenance import Value, Provenance
from app.core.reward_engine import RewardEngine, PaymentOption

def best_card(
    amount: Optional[float] = None,
    category: str = "general",
    available_cards: Optional[List[str]] = None,
    spend_to_date: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Tool 3: Ranks cards for spend amount and category using RewardEngine.
    Returns payment options containing tagged Value objects.
    Safely handles None amount by checking requirements.
    """
    if amount is None:
        return {
            "amount": None,
            "category": category,
            "ranked_options": [],
            "best_option": None,
            "top_card_id": None,
            "lowest_effective_price": None,
            "note": "Amount required for card ranking"
        }

    cards = DataIndex.get_cards()
    if available_cards:
        cards = [c for c in cards if c["card_id"].lower() in [ac.lower() for ac in available_cards]]

    base_val = Value(
        amount=float(amount),
        provenance=Provenance.SOURCE,
        record_id="spend_amount"
    )

    options: List[PaymentOption] = []
    for card in cards:
        opt = RewardEngine.calculate_payment_option(
            base_price=base_val,
            card=card,
            category=category,
            spend_to_date=spend_to_date
        )
        options.append(opt)

    ranked = RewardEngine.rank_options(options)
    best_opt = ranked[0] if ranked else None

    return {
        "amount": base_val,
        "category": category,
        "ranked_options": ranked,
        "best_option": best_opt,
        "top_card_id": best_opt.card_id if best_opt else None,
        "lowest_effective_price": best_opt.effective_price if best_opt else None
    }
