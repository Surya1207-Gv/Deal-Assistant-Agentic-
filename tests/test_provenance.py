import pytest
from app.core.provenance import Value, Provenance, ProvenanceValidator
from app.agent.graph import validate_provenance_node
from app.core.reward_engine import PaymentOption, RewardEngine

def test_provenance_tagging():
    v1 = Value(amount=4000.0, provenance=Provenance.SOURCE, record_id="deal_042")
    assert v1.amount == 4000.0
    assert v1.provenance == Provenance.SOURCE
    assert v1.record_id == "deal_042"

    v2 = Value(amount=3800.0, provenance=Provenance.DERIVED, formula="4000 * 0.95", inputs=["deal_042"])
    assert v2.provenance == Provenance.DERIVED
    assert v2.formula == "4000 * 0.95"

def test_provenance_validator_valid_text():
    trace = [
        Value(amount=4000.0, provenance=Provenance.SOURCE, record_id="deal_042"),
        Value(amount=3800.0, provenance=Provenance.DERIVED, formula="4000 - 200"),
        Value(amount=190.0, provenance=Provenance.DERIVED, formula="3800 * 0.05"),
        Value(amount=3610.0, provenance=Provenance.DERIVED, formula="3800 - 190"),
        Value(amount=5.0, provenance=Provenance.DERIVED, formula="5%")
    ]
    draft_text = "The base price is ₹4,000. After a 5% discount, it becomes 3800.00. You earn ₹190 reward, so effective price is 3610."
    is_valid, validated, unverified = ProvenanceValidator.validate(draft_text, trace)
    assert is_valid is True
    assert len(unverified) == 0

def test_provenance_validator_rejects_hallucination():
    trace = [
        Value(amount=4000.0, provenance=Provenance.SOURCE, record_id="deal_042"),
        Value(amount=3800.0, provenance=Provenance.DERIVED, formula="4000 - 200"),
        Value(amount=190.0, provenance=Provenance.DERIVED, formula="3800 * 0.05"),
        Value(amount=3610.0, provenance=Provenance.DERIVED, formula="3800 - 190")
    ]
    # 9999.0 is an unverified hallucinated figure
    draft_text = "You save money and pay effective price ₹9999."
    is_valid, validated, unverified = ProvenanceValidator.validate(draft_text, trace)
    assert is_valid is False
    assert any("9999" in u for u in unverified)

def test_provenance_gating_blocks_draft():
    # Force a PaymentOption with empty trace (so any number in draft is unverified)
    bad_opt = PaymentOption(
        base_price=Value(amount=4000.0, provenance=Provenance.SOURCE),
        discount_applied=Value(amount=200.0, provenance=Provenance.DERIVED),
        discount_source_id="deal_042",
        price_after_discount=Value(amount=3800.0, provenance=Provenance.DERIVED),
        card_id="hdfc_millennia",
        reward_earned=Value(amount=190.0, provenance=Provenance.DERIVED),
        reward_rate_applied=0.05,
        raw_reward=Value(amount=190.0, provenance=Provenance.DERIVED),
        cap_hit=True,
        cap_explanation="Reward capped at 190 with fake extra number 987654",
        reward_lost_to_cap=Value(amount=0.0, provenance=Provenance.DERIVED),
        effective_price=Value(amount=3610.0, provenance=Provenance.DERIVED),
        citations=["deal_042", "hdfc_millennia"],
        trace=[]  # Empty trace forces provenance failure
    )

    state = {
        "best_option": bad_opt,
        "runner_up_option": None,
        "trace": [],
        "citations": ["deal_042", "hdfc_millennia"]
    }

    res_state = validate_provenance_node(state)
    assert res_state["provenance_valid"] is False
    assert "Validation blocked response output" in res_state["final_response"]

def test_cap_binding_trace_non_empty():
    base_price = Value(amount=6000.0, provenance=Provenance.SOURCE, record_id="user_query_spend")
    card = {
        "card_id": "hdfc_millennia",
        "name": "HDFC Millennia",
        "base_rate": 0.01,
        "category_multipliers": {"groceries": 5.0},
        "caps": {"monthly_cashback_cap": 1000, "category_caps": {"groceries": 500}}
    }
    deal = {"deal_id": "deal_002", "min_spend": 5000, "discount_type": "flat", "discount_value": 250}
    spend_to_date = {"hdfc_millennia_groceries": 300.0}

    opt = RewardEngine.calculate_payment_option(
        base_price=base_price,
        deal=deal,
        card=card,
        category="groceries",
        spend_to_date=spend_to_date
    )

    assert len(opt.trace) > 0
    trace_amounts = [v.rounded_amount() for v in opt.trace]
    assert 6000.0 in trace_amounts
    assert 250.0 in trace_amounts
    assert 5750.0 in trace_amounts
    assert 500.0 in trace_amounts  # Category cap
    assert 300.0 in trace_amounts  # Prior spend
    assert 200.0 in trace_amounts  # Headroom & capped reward
    assert opt.cap_hit is True
    assert opt.reward_earned.amount == 200.0
