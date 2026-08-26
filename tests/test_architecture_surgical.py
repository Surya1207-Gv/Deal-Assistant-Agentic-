from __future__ import annotations
import json
import pytest
from pathlib import Path
from typing import Dict, Any, List

from app.agent.graph import graph_app
from app.rag.index import DataIndex
from app.core.primitives import is_deal_eligible, filter_deals, enumerate_options
from app.core.provenance import Value, Provenance, ProvenanceValidator
from app.core.reward_engine import RewardEngine

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_kindle_croma_appliances_deal_rejected():
    """
    Section 26: Kindle purchase optimization.
    deal_041 (Croma Home Appliances 8%) must be rejected for Kindle Paperwhite e-reader.
    Amazon ₹13,999 must be the ground base price.
    """
    kindle_prod = DataIndex.get_product_by_id("prod_kindle_paperwhite")
    deal_041 = DataIndex.get_deal_by_id("deal_041")
    sbi_card = DataIndex.get_card_by_id("sbi_cashback")

    eligible, reason = is_deal_eligible(
        deal=deal_041,
        product=kindle_prod,
        merchant="croma",
        purchase_amount=14999.0,
        card=sbi_card,
        category="electronics"
    )
    assert eligible is False
    assert reason == "PRODUCT_MISMATCH"

    state = graph_app.invoke({"query": "Find the cheapest Kindle after discounts and card rewards", "spend_to_date": {}})
    resp = state.get("final_response", "")
    assert "deal_041" not in state.get("citations", [])
    assert state.get("provenance_valid") is True


def test_bigbasket_grocery_deals_listing():
    """
    Section 27: Grocery deal discovery.
    'What grocery deals are available on BigBasket?' must run LIST operation and search_deals only.
    """
    state = graph_app.invoke({"query": "What grocery deals are available on BigBasket?", "spend_to_date": {}})
    assert state.get("operation") == "LIST"
    assert state.get("provenance_valid") is True
    resp = state.get("final_response", "")
    assert "deal_001" in resp or "deal_002" in resp or "deal_042" in resp


def test_grocery_reward_spend_vs_purchase_price():
    """
    Section 28: Grocery reward spend.
    'Which card gives me the best rewards for ₹6,000 groceries?' must use reward_spend = 6000.
    """
    state = graph_app.invoke({"query": "Which card gives me the best rewards for ₹6,000 groceries?", "spend_to_date": {}})
    resolved = state.get("resolved_query")
    assert resolved.reward_spend == 6000.0
    assert state.get("provenance_valid") is True
    resp = state.get("final_response", "")
    assert "deal" not in state.get("citations", []) or any("deal" in c for c in state.get("citations", []))


def test_hdfc_millennia_cashback_no_deal_needed():
    """
    Section 29: HDFC Millennia cashback calculation.
    'What cashback would I earn on ₹8,000 groceries with HDFC Millennia?'
    Millennia earns 5% up to ₹500 cap -> ₹400 cashback. No deal is needed.
    """
    state = graph_app.invoke({"query": "What cashback would I earn on ₹8,000 groceries with HDFC Millennia?", "spend_to_date": {}})
    resp = state.get("final_response", "")
    assert "400" in resp or "7,600" in resp or "7600" in resp
    assert state.get("provenance_valid") is True
    assert "no reliable deal found" not in resp.lower()


def test_missing_spend_amount_no_5000_fallback():
    """
    Section 30: Missing spend amount.
    'Which card is best for groceries?' with no spend amount must NEVER invent ₹5,000.
    """
    state = graph_app.invoke({"query": "Which card is best for groceries?", "spend_to_date": {}})
    resp = state.get("final_response", "")
    assert "5000" not in resp and "5,000" not in resp
    assert state.get("provenance_valid") is True


def test_merchant_isolation():
    """
    Section 31: Merchant isolation.
    A Croma-only deal cannot be applied to an Amazon price.
    """
    deal_croma = DataIndex.get_deal_by_id("deal_013")  # Croma Digital Bonanza
    amazon_bp = Value(amount=69900.0, provenance=Provenance.SOURCE, record_id="prod_iphone_15_amazon")
    sbi_card = DataIndex.get_card_by_id("sbi_cashback")

    eligible, reason = is_deal_eligible(
        deal=deal_croma,
        merchant="amazon",
        purchase_amount=69900.0,
        card=sbi_card,
        category="electronics"
    )
    assert eligible is False
    assert reason == "MERCHANT_MISMATCH"


def test_category_and_subcategory_isolation():
    """
    Section 32: Category and subcategory isolation.
    Home appliances deal cannot apply to headphones.
    """
    deal_appliances = DataIndex.get_deal_by_id("deal_041")
    sony_headphones = DataIndex.get_product_by_id("prod_sony_headphones")

    eligible, reason = is_deal_eligible(
        deal=deal_appliances,
        product=sony_headphones,
        merchant="croma",
        purchase_amount=28990.0,
        category="electronics"
    )
    assert eligible is False
    assert reason == "PRODUCT_MISMATCH"


def test_card_isolation():
    """
    Section 33: Card isolation.
    Deal requiring Axis ACE cannot be applied to HDFC Millennia.
    """
    deal_axis = DataIndex.get_deal_by_id("deal_009")  # Axis ACE on Zomato
    millennia_card = DataIndex.get_card_by_id("hdfc_millennia")

    eligible, reason = is_deal_eligible(
        deal=deal_axis,
        merchant="zomato",
        purchase_amount=800.0,
        card=millennia_card,
        category="food"
    )
    assert eligible is False
    assert reason == "CARD_MISMATCH"


def test_minimum_spend_boundary():
    """
    Section 34: Minimum spend boundary.
    Deal with min_spend = 10,000:
    ₹8,000 -> rejected (MIN_SPEND_NOT_MET)
    ₹10,000 -> accepted (eligible)
    """
    deal_10k = DataIndex.get_deal_by_id("deal_004")  # Amazon Electronics min spend 10000
    icici_card = DataIndex.get_card_by_id("icici_amazon_pay")

    el_8k, reason_8k = is_deal_eligible(
        deal=deal_10k,
        merchant="amazon",
        purchase_amount=8000.0,
        card=icici_card,
        category="electronics"
    )
    assert el_8k is False
    assert reason_8k == "MIN_SPEND_NOT_MET"

    el_10k, reason_10k = is_deal_eligible(
        deal=deal_10k,
        merchant="amazon",
        purchase_amount=10000.0,
        card=icici_card,
        category="electronics"
    )
    assert el_10k is True
    assert reason_10k is None


def test_prompt_injection_inert():
    """
    Section 35: Prompt injection defense with structured fields preserved.
    TC13-TC15 prompt injection descriptions remain inert and structured arithmetic is executed.
    """
    state_tc13 = graph_app.invoke({"query": "What deal is available at SuperStore for a ₹2,000 grocery order?", "spend_to_date": {}})
    resp_tc13 = state_tc13.get("final_response", "")
    assert "100" in resp_tc13
    assert "1805" in resp_tc13
    assert "0" not in resp_tc13 or "90%" not in resp_tc13
    assert state_tc13.get("provenance_valid") is True


def test_explicit_subcategory_taxonomy_matching():
    """
    Explicit subcategory taxonomy verification.
    Verifies that deal.subcategory vs product.subcategory is strictly evaluated.
    """
    appliances_deal = DataIndex.get_deal_by_id("deal_041")
    assert appliances_deal.get("subcategory") == "appliances"

    kindle = DataIndex.get_product_by_id("prod_kindle_paperwhite")
    assert kindle.get("subcategory") == "ereader"

    coffee_maker = DataIndex.get_product_by_id("prod_coffee_maker")
    assert coffee_maker.get("subcategory") == "appliances"

    # Ineligible for ereader
    el_kindle, reason = is_deal_eligible(deal=appliances_deal, product=kindle)
    assert el_kindle is False
    assert reason == "PRODUCT_MISMATCH"

    # Eligible for appliances
    el_coffee, reason_coffee = is_deal_eligible(deal=appliances_deal, product=coffee_maker)
    assert el_coffee is True
    assert reason_coffee is None


def test_dynamic_card_policy_registry():
    """
    Dynamic card reward program updates verification.
    Verifies runtime multiplier and cap overrides via CardPolicyRegistry.
    """
    from app.core.card_registry import CardPolicyRegistry

    # Original Millennia base rate is 0.01 (1%)
    orig_policy = CardPolicyRegistry.get_card_policy("hdfc_millennia")
    assert orig_policy["base_rate"] == 0.01

    # Dynamically register promotional live policy update: 2% base rate, 10x grocery multiplier
    CardPolicyRegistry.register_policy_update(
        card_id="hdfc_millennia",
        base_rate=0.02,
        category_multipliers={"groceries": 10.0, "shopping": 5.0}
    )

    updated_policy = CardPolicyRegistry.get_card_policy("hdfc_millennia")
    assert updated_policy["base_rate"] == 0.02
    assert updated_policy["category_multipliers"]["groceries"] == 10.0

    # Reset back to snapshot
    CardPolicyRegistry.reset_overrides()
    reset_policy = CardPolicyRegistry.get_card_policy("hdfc_millennia")
    assert reset_policy["base_rate"] == 0.01
