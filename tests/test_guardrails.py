"""
Grounding guardrails.

Each test here pins a class of failure found by sweeping questions generated from the
dataset itself. They are about what the system must REFUSE to do, or must not quietly
assume, rather than about any particular sentence.
"""
from __future__ import annotations

import pytest

from app.agent.graph import graph_app
from app.core.memory import MemoryManager
from app.rag.index import DataIndex


def ask(query: str, session_id: str) -> dict:
    MemoryManager.reset_session(session_id)
    return graph_app.invoke({"session_id": session_id, "query": query, "spend_to_date": {}})


def abstained(state: dict) -> bool:
    return bool(state.get("abstained")) or "no reliable deal found" in state.get("final_response", "").lower()


def test_out_of_domain_query_does_not_capture_a_product():
    """
    "Mumbai" occurs in exactly one product title (the Delhi-Mumbai flight), which was enough
    for the weakest matcher to turn an out-of-domain question into a flight recommendation.
    A lone title word only counts as a product reference in a turn that is otherwise about
    shopping.
    """
    state = ask("What's the weather in Mumbai?", "guard_domain")
    resp = state.get("final_response", "")
    assert "Recommendation:" not in resp
    assert not state.get("best_option")


def test_named_entity_absent_from_catalogue_abstains():
    """An out-of-catalogue product named WITHOUT a purchase verb must still be refused."""
    state = ask("SpaceX Starlink Satellite Receiver subscription promo code", "guard_ungrounded")
    assert abstained(state)


def test_unknown_card_of_known_issuer_abstains_rather_than_asking():
    """
    "HDFC Infinia" names a card the catalogue does not hold. Offering a choice between the
    two HDFC cards it *does* hold answers a question nobody asked.
    """
    state = ask("Does HDFC Infinia card give 5x reward points?", "guard_unknown_card")
    assert abstained(state)
    assert "Millennia" not in state.get("final_response", "")


def test_ambiguous_issuer_alone_still_asks():
    """The genuine ambiguity is preserved: two catalogue cards share the issuer name."""
    state = ask("How much cashback with HDFC?", "guard_ambiguous_card")
    assert state.get("operation") == "CLARIFY"
    resp = state.get("final_response", "")
    assert "Millennia" in resp and "Regalia" in resp


def test_price_watch_requires_a_real_product():
    """
    A watch is a promise about a real price, so one must never be confirmed for a product
    the catalogue does not carry.

    Asserted on the OUTCOME rather than on the operation: which handler this query reaches
    is partly an LLM classification decision (it routes to WATCH only because the product's
    own name contains the word "watch"), so asserting the route made the test depend on
    provider availability. The guardrail itself holds either way.
    """
    state = ask("Rolex Submariner Gold Watch discount on Chrono24", "guard_watch_unknown")
    resp = state.get("final_response", "").lower()
    assert "price watch registered" not in resp
    assert "rolex" not in resp and "chrono24" not in resp


def test_watch_handler_refuses_a_product_the_catalogue_lacks():
    """The deterministic half of the guardrail above, exercised directly."""
    from app.agent.operations import handle_watch
    from app.agent.resolver import Operation, ResolvedQuery

    resolved = ResolvedQuery(operation=Operation.WATCH, target=500000.0)
    state = handle_watch(resolved, {"query": "Rolex Submariner", "session_id": "guard_watch_unit"})
    assert "no reliable deal found" in state.get("final_response", "").lower()


def test_price_watch_still_works_for_a_catalogue_product():
    state = ask("Tell me if the MacBook Air M2 drops below RS 85000", "guard_watch_known")
    assert not abstained(state)
    resp = state.get("final_response", "")
    assert "85000" in resp or "85,000" in resp


def test_descriptor_word_does_not_trigger_product_clarification():
    """
    "monthly" appears in three product titles across two categories, which makes it a
    descriptor rather than a product family. Asking which one is meant derails a perfectly
    answerable category question.
    """
    state = ask("Best way to stock up on monthly groceries?", "guard_descriptor")
    assert state.get("operation") != "CLARIFY"


def test_shared_model_word_still_asks():
    """A word naming two products in ONE category is a real ambiguity worth asking about."""
    state = ask("Deals on iPhone?", "guard_family")
    assert state.get("operation") == "CLARIFY"
    assert "iPhone 15" in state.get("final_response", "")


def test_purchase_without_a_named_product_still_considers_deals():
    """
    A product-less amount is not automatically a reward question. "Buy groceries worth
    RS 4,000" names no catalogue product yet is plainly a purchase, and suppressing merchant
    deals for it hid every grocery offer.
    """
    state = ask("I want to buy groceries worth 4000", "guard_purchase")
    best = state.get("best_option")
    assert best is not None
    assert best.discount_source_id, "a merchant deal should have been considered"


def test_pure_reward_question_does_not_attach_a_deal():
    """The mirror image: a cashback question is answered from card rules alone."""
    state = ask("How much cashback would I earn on 8,000 groceries with HDFC Millennia?",
                "guard_reward")
    best = state.get("best_option")
    assert best is not None
    assert best.discount_source_id is None
    assert best.reward_earned.amount == pytest.approx(400.0)


def test_injection_records_are_reported_as_excluded():
    """
    The injection filter's verdict has to reach the caller. It was written into graph state
    that AgentState never declared, so LangGraph dropped it silently.
    """
    state = ask("Check SuperStore grocery deal 1 details", "guard_injection")
    assert "deal_031" in (state.get("excluded_injection_records") or [])
    assert state.get("provenance_valid") is True


def test_merchant_implies_category_from_the_data():
    """
    A merchant whose deals all sit in one category identifies that category. Without this,
    a stale session category filtered the merchant's own deals out of retrieval.
    """
    state = ask("What is TravelFly deal 3 discount?", "guard_merchant_cat")
    assert "deal_033" in state.get("final_response", "") or "deal_033" in state.get("citations", [])


@pytest.mark.parametrize("product", [p["name"] for p in DataIndex.get_products()[:6]])
def test_planned_tools_corroborate_the_derivation(product):
    """
    The planner's tool calls are load-bearing: whatever they independently report about
    price, reward policy and deal provenance is checked against the derivation.
    """
    state = ask(f"What's the cheapest way to buy the {product}?", "guard_corrob")
    report = state.get("tool_corroboration") or {}
    assert not report.get("disagreements"), report.get("disagreements")
    assert state.get("best_option") is not None


# ---------------------------------------------------------------- rejected-deal explanations

def test_material_rejected_deals_are_explained():
    """
    A deal the user can see in the retrieved list, whose headline discount would have matched
    or beaten the one actually applied, is explained rather than silently dropped.

    Both Amazon electronics deals below advertise more than the RS 1,399.90 that was applied,
    and both fail on minimum spend against a RS 13,999 Kindle.
    """
    state = ask("I'm trying to get a Kindle as cheaply as possible. "
                "I don't care which card I use. What would you recommend?", "wn_kindle")
    resp = state.get("final_response", "")
    assert "Why not the other visible deals?" in resp
    assert "deal_035" in resp and "25,000" in resp
    assert "deal_018" in resp and "15,000" in resp
    assert "13,999" in resp
    assert state.get("provenance_valid") is True


def test_immaterial_rejected_deals_are_not_mentioned():
    """
    Deals that could never have won explain nothing. deal_013 (flat RS 1,000 at Croma) and
    deal_005 (7.5% -> RS 1,049.93 here) both fall short of the RS 1,399.90 applied, so neither
    belongs in the note even though both were retrieved.
    """
    state = ask("I'm trying to get a Kindle as cheaply as possible. "
                "I don't care which card I use. What would you recommend?", "wn_kindle2")
    shown = {r.deal_id for r in (state.get("rejected_visible_deals") or [])}
    assert "deal_013" not in shown
    assert "deal_005" not in shown


def test_explained_deals_never_become_financial_candidates():
    """
    The explanation is a REPORT over the candidate space, never an input to it. A surfaced
    deal must not appear as a payment option, as the applied deal, or in the citations.
    """
    from app.core.primitives import is_deal_eligible

    for product in DataIndex.get_products():
        state = ask(f"What's the cheapest way to buy the {product['name']}?",
                    "wn_" + product["product_id"])
        rejected = state.get("rejected_visible_deals") or []
        options = state.get("payment_options") or []
        best = state.get("best_option")
        for r in rejected:
            assert r.deal_id not in (state.get("citations") or []), r.deal_id
            assert best is None or best.discount_source_id != r.deal_id, r.deal_id
            assert all(o.discount_source_id != r.deal_id for o in options), r.deal_id
            # and it really is ineligible everywhere in the space
            deal = DataIndex.get_deal_by_id(r.deal_id)
            assert not any(
                is_deal_eligible(deal=deal, product=product, merchant=m.lower(),
                                 purchase_amount=float(price), card=c,
                                 category=product["category"])[0]
                for m, price in product["prices"].items()
                for c in DataIndex.get_cards()
            ), r.deal_id


def test_explanations_do_not_leak_retrieval_internals():
    state = ask("I'm trying to get a Kindle as cheaply as possible. "
                "I don't care which card I use. What would you recommend?", "wn_kindle3")
    resp = state.get("final_response", "").lower()
    for leak in ("retrieval_score", "score=", "min_spend_not_met", "card_mismatch",
                 "merchant_mismatch", "product_mismatch", "category_mismatch"):
        assert leak not in resp, leak


# ---------------------------------------------------------------- objective semantics

OBJECTIVE_PARAPHRASES = [
    "What's the best card for an 8,000 grocery purchase?",
    "Which card gives me the most cashback on 8k groceries?",
    "I'll spend 8000 on groceries - which card should I use?",
    "How much cashback can I get on groceries if I pay 8,000?",
    "Which card is most rewarding for an 8k grocery transaction?",
    "Best card to use for 8,000 of groceries?",
]


def test_paraphrases_of_one_intent_resolve_identically():
    """
    The same question asked six ways must produce the same semantics. Previously the
    objective was a by-product of five reward keywords, so half of these attached a merchant
    coupon and half did not, and two failed to parse the amount at all.
    """
    seen = set()
    for i, q in enumerate(OBJECTIVE_PARAPHRASES):
        state = ask(q, f"obj_para_{i}")
        resolved = state["resolved_query"]
        best = state.get("best_option")
        assert resolved.amount == 8000.0, (q, resolved.amount)
        assert resolved.objective == "max_reward", (q, resolved.objective)
        assert best is not None and best.discount_source_id is None, q
        seen.add((resolved.objective, best.card_id, round(best.reward_earned.amount, 2)))
    assert len(seen) == 1, seen


def test_objectives_are_distinguished():
    """Reward, price and discount objectives must not collapse into one another."""
    reward = ask("What is the best card for 8,000 groceries?", "obj_r")
    price = ask("What is the cheapest way to buy 8,000 of groceries?", "obj_p")
    discount = ask("Which grocery deal gives the biggest discount?", "obj_d")

    assert reward["resolved_query"].objective == "max_reward"
    assert reward.get("best_option").discount_source_id is None, "reward question attached a deal"

    assert price["resolved_query"].objective == "min_effective_price"
    assert price.get("best_option").discount_source_id is not None, "price question ignored deals"

    assert discount["resolved_query"].objective == "max_discount"
    assert discount.get("operation") == "AGGREGATE"


def test_objective_persists_across_a_follow_up():
    MemoryManager.reset_session("obj_follow")
    graph_app.invoke({"session_id": "obj_follow",
                      "query": "Which card gives the most cashback on 8,000 groceries?",
                      "spend_to_date": {}})
    state = graph_app.invoke({"session_id": "obj_follow", "query": "What if I spend 12,000?",
                              "spend_to_date": {}})
    assert state["resolved_query"].objective == "max_reward"
    assert state["resolved_query"].amount == 12000.0
    assert state.get("best_option").discount_source_id is None


def test_amount_parsing_is_phrasing_independent():
    """The same figure, written six ways, is the same amount; specs are not amounts."""
    from app.agent.resolver import QueryResolver

    for phrasing in ["for an 8,000 grocery purchase", "8k groceries", "if I pay 8,000",
                     "8,000 worth of groceries", "on an 8,000 grocery bill",
                     "spend 8000 on groceries"]:
        amount, _ = QueryResolver._parse_amount(phrasing, phrasing.lower())
        assert amount == 8000.0, (phrasing, amount)

    for spec in ["4K display models only?", "Apple iPhone 15 128GB",
                 "Samsung 55-inch 4K Smart TV", "Gourmet Dinner Meal Box for 4"]:
        amount, _ = QueryResolver._parse_amount(spec, spec.lower())
        assert amount is None, (spec, amount)


def test_indifference_does_not_read_as_an_objective():
    """"I don't care which card" waives the card dimension; it does not request one."""
    state = ask("I'm trying to get a Kindle as cheaply as possible. "
                "I don't care which card I use. What would you recommend?", "obj_indiff")
    assert state["resolved_query"].objective == "min_effective_price"


def test_rejected_deal_can_be_explained_by_the_figure_it_advertised():
    """
    A number that matches a figure the previous answer quoted is a reference back to it, not
    a new spend. Reading it as a purchase amount priced a RS 1,500 basket.
    """
    MemoryManager.reset_session("obj_rej")
    graph_app.invoke({"session_id": "obj_rej",
                      "query": "I want the cheapest Kindle. Any card is fine.",
                      "spend_to_date": {}})
    state = graph_app.invoke({"session_id": "obj_rej",
                              "query": "Why did not you use the 1,500 off deal?",
                              "spend_to_date": {}})
    assert state.get("operation") == "EXPLAIN"
    resp = state.get("final_response", "")
    assert "25,000" in resp and "13,999" in resp
    assert "Base Price: RS 1500" not in resp
    assert state.get("provenance_valid") is True


# ---------------------------------------------------------------- tie handling

def test_card_tie_is_reported_not_silently_broken():
    """
    Three cards pay an identical RS 400 on this spend. Ties were measured on effective price
    whatever the question was, and reported only across merchants, so a card tie was resolved
    by card-id ordering and one card presented as the unique winner.
    """
    state = ask("What is the best card for 8,000 groceries?", "tie_card")
    tie = state.get("tie")
    assert tie is not None, "tie not detected"
    assert tie["axis"] == "card" and tie["metric"] == "reward"
    assert tie["value"] == pytest.approx(400.0)
    assert len(tie["labels"]) == 3
    assert tie["labels"] == sorted(tie["labels"]), "presentation order must be deterministic"
    resp = state.get("final_response", "")
    assert "Three cards tie for the highest cashback of RS 400" in resp
    for card in tie["labels"]:
        assert card in resp
    assert state.get("provenance_valid") is True


def test_merchant_tie_is_reported_over_merchants():
    state = ask("Cheapest way to order Gourmet Dinner Meal Box for 4 with Amex SmartEarn",
                "tie_merchant")
    tie = state.get("tie")
    assert tie is not None and tie["axis"] == "merchant"
    assert tie["metric"] == "effective_price"
    assert len(tie["labels"]) >= 2
    assert "tie for the lowest effective price" in state.get("final_response", "")


def test_no_tie_is_invented_when_there_is_a_unique_winner():
    state = ask("What is the cheapest way to buy the Kindle Paperwhite?", "tie_none")
    assert state.get("tie") is None
    assert "tie for the" not in state.get("final_response", "")


def test_tie_detection_does_not_change_the_chosen_figures():
    """Reporting a tie must not alter what was computed."""
    state = ask("What is the best card for 8,000 groceries?", "tie_figures")
    best = state.get("best_option")
    assert best.reward_earned.amount == pytest.approx(400.0)
    assert best.discount_applied.amount == pytest.approx(0.0)
    assert best.effective_price.amount == pytest.approx(7600.0)
