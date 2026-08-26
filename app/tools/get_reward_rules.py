from __future__ import annotations
from typing import Dict, Any, Optional
from app.rag.index import DataIndex

def get_reward_rules(card_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Tool 4: Fetches reward policy, category multipliers, and caps for a card.
    Safely handles None card_id.
    """
    if not card_id:
        return {"found": False, "error": "No card_id provided"}
    clean_card_id = str(card_id).strip().lower()
    card = DataIndex.get_card_by_id(clean_card_id)
    if not card:
        return {"found": False, "card_id": clean_card_id, "error": "Card not found"}

    return {
        "found": True,
        "card_id": card["card_id"],
        "name": card["name"],
        "base_rate": card["base_rate"],
        "category_multipliers": card["category_multipliers"],
        "caps": card["caps"],
        "min_spend": card.get("min_spend", 0)
    }
