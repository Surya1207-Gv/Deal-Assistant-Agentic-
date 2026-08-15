import pytest
from app.core.provenance import Value, Provenance
from app.core.reward_engine import RewardEngine, PaymentOption

@pytest.fixture
def base_card():
    return {
        "card_id": "hdfc_millennia",
        "name": "HDFC Millennia",
        "base_rate": 0.01,
        "category_multipliers": {"groceries": 5.0, "travel": 2.5},
        "caps": {
            "monthly_cashback_cap": 1000,
            "category_caps": {"groceries": 500}
        },
        "min_spend": 0,
        "cap_period": "monthly"
    }

def test_1_flat_base_rate_no_category_match(base_card):
    # Purchase 5000 in electronics category (no multiplier in card) -> 1% base rate = 50 reward
    base_val = Value(amount=5000.0, provenance=Provenance.SOURCE, record_id="prod_tv")
    option = RewardEngine.calculate_payment_option(
        base_price=base_val,
        card=base_card,
        category="electronics"
    )
    assert option.price_after_discount.amount == 5000.0
    assert option.reward_earned.amount == 50.0
    assert option.effective_price.amount == 4950.0
    assert option.cap_hit is False

def test_2_category_multiplier_applies(base_card):
    # Purchase 4000 in groceries category -> 5x multiplier (5% rate) -> 200 reward
    base_val = Value(amount=4000.0, provenance=Provenance.SOURCE, record_id="prod_groceries")
    option = RewardEngine.calculate_payment_option(
        base_price=base_val,
        card=base_card,
        category="groceries"
    )
    assert option.price_after_discount.amount == 4000.0
    assert option.reward_earned.amount == 200.0
    assert option.effective_price.amount == 3800.0
    assert option.cap_hit is False

def test_3_cap_binds_partially(base_card):
    # Spend earns 800 (groceries at 16000 * 5% = 800), category cap is 500 -> reward 500, cap_hit=True
    base_val = Value(amount=16000.0, provenance=Provenance.SOURCE, record_id="prod_bulk_groceries")
    option = RewardEngine.calculate_payment_option(
        base_price=base_val,
        card=base_card,
        category="groceries"
    )
    assert option.reward_earned.amount == 500.0
    assert option.cap_hit is True
    assert "capped at ₹500" in option.cap_explanation or "category cap" in option.cap_explanation

def test_4_cap_already_partly_consumed(base_card):
    # 300 of a 500 cap used this month -> 200 headroom. Purchase of 6000 @ 5% = 300 reward -> capped at 200
    base_val = Value(amount=6000.0, provenance=Provenance.SOURCE, record_id="prod_groceries")
    spend_to_date = {"hdfc_millennia_groceries": 300.0}
    option = RewardEngine.calculate_payment_option(
        base_price=base_val,
        card=base_card,
        category="groceries",
        spend_to_date=spend_to_date
    )
    assert option.reward_earned.amount == 200.0
    assert option.cap_hit is True
    assert option.effective_price.amount == 5800.0

def test_5_two_caps_interact(base_card):
    # Category cap 500 (300 used -> 200 cat headroom). Monthly cap 1000 (900 used -> 100 monthly headroom).
    # Effective headroom = min(200, 100) = 100.
    base_val = Value(amount=4000.0, provenance=Provenance.SOURCE, record_id="prod_groceries")
    spend_to_date = {
        "hdfc_millennia_groceries": 300.0,
        "hdfc_millennia_monthly": 900.0
    }
    option = RewardEngine.calculate_payment_option(
        base_price=base_val,
        card=base_card,
        category="groceries",
        spend_to_date=spend_to_date
    )
    assert option.reward_earned.amount == 100.0
    assert option.cap_hit is True

def test_6_stacking_order_coupon_then_reward(base_card):
    # Base 4000, 5% coupon -> 3800 post-discount. 5% reward calculated on 3800 = 190. Effective = 3610.
    base_val = Value(amount=4000.0, provenance=Provenance.SOURCE, record_id="prod_groceries")
    deal = {
        "deal_id": "deal_042",
        "discount_type": "percentage",
        "discount_value": 0.05,
        "min_spend": 1500,
        "max_discount": 250
    }
    option = RewardEngine.calculate_payment_option(
        base_price=base_val,
        deal=deal,
        card=base_card,
        category="groceries"
    )
    assert option.discount_applied.amount == 200.0
    assert option.price_after_discount.amount == 3800.0
    assert option.reward_earned.amount == 190.0
    assert option.effective_price.amount == 3610.0

def test_7_minimum_spend_not_met(base_card):
    # Card with min_spend = 1000, purchase is 500 -> 0 reward
    min_spend_card = dict(base_card)
    min_spend_card["min_spend"] = 1000

    base_val = Value(amount=500.0, provenance=Provenance.SOURCE, record_id="prod_small")
    option = RewardEngine.calculate_payment_option(
        base_price=base_val,
        card=min_spend_card,
        category="groceries"
    )
    assert option.reward_earned.amount == 0.0
    assert "Minimum spend requirement" in option.cap_explanation

def test_8_tiebreak_between_cards(base_card):
    # Card A and Card B yield same effective price. Tiebreak by card_id ascending.
    base_val = Value(amount=1000.0, provenance=Provenance.SOURCE, record_id="prod_test")
    opt1 = RewardEngine.calculate_payment_option(base_price=base_val, card={"card_id": "card_b", "base_rate": 0.01})
    opt2 = RewardEngine.calculate_payment_option(base_price=base_val, card={"card_id": "card_a", "base_rate": 0.01})

    ranked = RewardEngine.rank_options([opt1, opt2])
    assert ranked[0].card_id == "card_a"
    assert ranked[1].card_id == "card_b"
