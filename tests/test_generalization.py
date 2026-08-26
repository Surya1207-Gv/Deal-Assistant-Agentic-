from __future__ import annotations
import json
import pytest
from pathlib import Path
from typing import Dict, Any, List, Optional

from app.agent.graph import graph_app
from app.rag.index import DataIndex
from app.core.provenance import ProvenanceValidator
from app.core.primitives import is_deal_eligible

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_data():
    with open(PROJECT_ROOT / "data" / "products.json", "r", encoding="utf-8") as f:
        products = json.load(f)
    with open(PROJECT_ROOT / "data" / "deals.json", "r", encoding="utf-8") as f:
        deals = json.load(f)
    with open(PROJECT_ROOT / "data" / "cards.json", "r", encoding="utf-8") as f:
        cards = json.load(f)
    return products, deals, cards

# Independent Reference Brute-Force Calculator (NOT importing RewardEngine)
def reference_calculate_effective(base_amount: float, deal: Optional[Dict[str, Any]], card: Dict[str, Any], category: str) -> float:
    discount = 0.0
    if deal:
        min_s = deal.get("min_spend", 0)
        card_sp = deal.get("card_specific")
        if (not card_sp or card_sp.lower() == card["card_id"].lower()) and base_amount >= min_s:
            d_type = deal.get("discount_type", "flat")
            d_val = float(deal.get("discount_value", 0))
            max_d = float(deal.get("max_discount", float("inf")))
            if d_type == "percentage":
                discount = min(base_amount * (d_val if d_val <= 1.0 else d_val / 100.0), max_d)
            else:
                discount = min(d_val, max_d)
    
    post_price = max(0.0, base_amount - discount)
    
    base_rate = card.get("base_rate", 0.01)
    mult = card.get("category_multipliers", {}).get(category, 1.0)
    applied_rate = base_rate * mult
    raw_reward = post_price * applied_rate
    
    caps = card.get("caps", {})
    m_cap = caps.get("monthly_cashback_cap", float("inf"))
    cat_caps = caps.get("category_caps", {})
    c_cap = cat_caps.get(category, float("inf"))
    eff_cap = min(m_cap, c_cap)
    
    reward = min(raw_reward, eff_cap)
    return round(post_price - reward, 2)


def test_generalization_products_cheapest():
    products, deals, cards = load_data()
    for p in products:
        q = f"What is the cheapest way to buy {p['name']}?"
        state = graph_app.invoke({"query": q, "spend_to_date": {}})
        assert state.get("provenance_valid", False) is True, f"Failed provenance for {p['name']}"
        assert state.get("operation") in ["OPTIMIZE", "COMPUTE"], f"Wrong op for {p['name']}: {state.get('operation')}"
        
        # Verify ground truth. Candidate deals are filtered through the app's own
        # canonical eligibility engine (is_deal_eligible) — merchant/category alone is not
        # sufficient (section 10: category match is not enough; e.g. an Ajio "Shoe Fair"
        # deal must not discount jeans just because both are merchant=Ajio, category=shopping).
        # The effective-price arithmetic below stays independent of RewardEngine.
        best_ref = float("inf")
        cat = p["category"]
        for m, price in p["prices"].items():
            m_deals = [
                d for d in deals
                if is_deal_eligible(deal=d, product=p, merchant=m, purchase_amount=float(price), category=cat)[0]
            ]
            for c in cards:
                for d in m_deals:
                    eff = reference_calculate_effective(float(price), d, c, cat)
                    if eff < best_ref:
                        best_ref = eff
                eff_base = reference_calculate_effective(float(price), None, c, cat)
                if eff_base < best_ref:
                    best_ref = eff_base
        
        resp = state.get("final_response", "")
        # Number should be present in final response
        ref_int = round(best_ref)
        assert f"{ref_int}" in resp or f"{ref_int:,.0f}" in resp or f"{best_ref:.0f}" in resp, f"Expected ~{best_ref} in response for {p['name']}, got: {resp}"


def test_generalization_deals_listing():
    products, deals, cards = load_data()
    merchants = list({d["merchant"] for d in deals if d.get("merchant")})
    for m in merchants:
        q = f"What deals does {m} have?"
        state = graph_app.invoke({"query": q, "spend_to_date": {}})
        assert state.get("provenance_valid", False) is True
        assert state.get("operation") in ["LIST", "DEAL_INFO", "LOOKUP"]
        resp = state.get("final_response", "")
        # Should mention at least one deal ID for that merchant
        expected_deals = [d["deal_id"] for d in deals if d.get("merchant", "").lower() == m.lower()]
        if expected_deals:
            assert any(did in resp for did in expected_deals), f"Expected deal ID in response for {m}"


def test_generalization_cards_caps_lookup():
    products, deals, cards = load_data()
    for c in cards:
        for cat in ["groceries", "electronics", "travel"]:
            q = f"What is the reward cap on {c['name']} for {cat}?"
            state = graph_app.invoke({"query": q, "spend_to_date": {}})
            assert state.get("provenance_valid", False) is True
            assert state.get("operation") in ["LOOKUP", "CARD_INFO"]


def test_generalization_eligibility():
    products, deals, cards = load_data()
    for d in deals[:10]:
        c = cards[0]
        q = f"Can I use {c['name']} with {d['deal_id']}?"
        state = graph_app.invoke({"query": q, "spend_to_date": {}})
        assert state.get("provenance_valid", False) is True
        assert state.get("operation") in ["ELIGIBILITY", "LOOKUP"]


def test_generalization_unresolved_abstain():
    state = graph_app.invoke({"query": "What is the cheapest way to buy Dyson vacuum cleaner?", "spend_to_date": {}})
    assert state.get("abstained", False) is True
    assert "no reliable deal found" in state.get("final_response", "").lower()

    state2 = graph_app.invoke({"query": "Deals on Tesla Cybertruck", "spend_to_date": {}})
    assert state2.get("abstained", False) is True
    assert "no reliable deal found" in state2.get("final_response", "").lower()
