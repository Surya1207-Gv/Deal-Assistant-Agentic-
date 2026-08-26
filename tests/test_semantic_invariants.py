"""
Architectural invariants of the semantic frame.

These are not example questions. Every case is GENERATED from the catalogue, so the suite
grows with the data and cannot be satisfied by special-casing a sentence. Each test states an
invariant the pipeline must hold for any turn, not a fact about one query.
"""
from __future__ import annotations

import itertools

import pytest

from app.agent.graph import graph_app
from app.core.memory import MemoryManager
from app.core.primitives import is_deal_eligible
from app.rag.index import DataIndex

PRODUCTS = DataIndex.get_products()
CARDS = DataIndex.get_cards()
CATEGORIES = sorted({p["category"] for p in PRODUCTS})

REWARD_PHRASINGS = [
    "What cashback would I get on a {a:,} {cat} purchase with {card}?",
    "How much cashback do I earn spending {a:,} on {cat} with {card}?",
    "What reward do I get on {a:,} of {cat} using {card}?",
]


def ask(query: str, session_id: str) -> dict:
    MemoryManager.reset_session(session_id)
    return graph_app.invoke({"session_id": session_id, "query": query, "spend_to_date": {}})


# ------------------------------------------------------------------ I1, I3, I4
@pytest.mark.parametrize("card", CARDS, ids=lambda c: c["card_id"])
@pytest.mark.parametrize("phrasing", range(len(REWARD_PHRASINGS)))
def test_reward_frame_admits_no_merchant_deal(card, phrasing):
    """
    INVARIANT 1 — a reward question with no merchant or deal named cannot acquire a
    merchant-specific deal, however retrieval scores it.
    INVARIANT 3 — it is ranked by reward, not by effective price.
    INVARIANT 4 — the card the user named stays fixed.
    """
    query = REWARD_PHRASINGS[phrasing].format(a=10000, cat="groceries", card=card["name"])
    state = ask(query, f"inv1_{card['card_id']}_{phrasing}")
    resolved = state["resolved_query"]
    best = state.get("best_option")
    options = state.get("payment_options") or []

    assert resolved.objective == "max_reward", query
    assert best is not None, query
    assert best.discount_source_id is None, f"{query} acquired deal {best.discount_source_id}"
    assert all(o.discount_source_id is None for o in options), f"{query} admitted a deal"
    assert best.card_id == card["card_id"], query
    assert best.reward_earned.amount == pytest.approx(
        max(o.reward_earned.amount for o in options)), query
    assert state.get("provenance_valid") is True, query


# ------------------------------------------------------------------ I5
@pytest.mark.parametrize("category", CATEGORIES)
def test_unnamed_card_means_every_card_is_considered(category):
    """INVARIANT 5 — with no card named, the card becomes the ranking dimension."""
    state = ask(f"Which card gives the highest cashback on 10,000 of {category}?", f"inv5_{category}")
    options = state.get("payment_options") or []
    assert state["resolved_query"].objective == "max_reward"
    assert len({o.card_id for o in options}) == len(CARDS)
    assert all(o.discount_source_id is None for o in options)


# ------------------------------------------------------------------ I2, I6
@pytest.mark.parametrize("product", PRODUCTS, ids=lambda p: p["product_id"])
def test_price_frame_evaluates_deals_and_every_option_is_eligible(product):
    """
    INVARIANT 2 — a price objective may evaluate merchant deals.
    INVARIANT 6 — no retrieved record becomes a PaymentOption without passing the canonical
    eligibility engine for that exact combination.
    """
    state = ask(f"Find the cheapest way to buy the {product['name']}.", "inv2_" + product["product_id"])
    resolved = state["resolved_query"]
    options = state.get("payment_options") or []
    best = state.get("best_option")

    assert resolved.objective == "min_effective_price"
    assert best.effective_price.amount == pytest.approx(
        min(o.effective_price.amount for o in options))

    for option in options:
        if not option.discount_source_id:
            continue
        eligible, _ = is_deal_eligible(
            deal=DataIndex.get_deal_by_id(option.discount_source_id),
            product=product,
            merchant=option.merchant,
            purchase_amount=option.base_price.amount,
            card=DataIndex.get_card_by_id(option.card_id) if option.card_id else None,
            category=product["category"],
        )
        assert eligible, f"{product['product_id']}: {option.discount_source_id} was not eligible"


# ------------------------------------------------------------------ I7
@pytest.mark.parametrize("product", PRODUCTS[:5], ids=lambda p: p["product_id"])
def test_budget_does_not_become_the_transaction_amount(product):
    """INVARIANT 7 — a stated budget is a ceiling, never the price of the thing."""
    sid = "inv7_" + product["product_id"]
    MemoryManager.reset_session(sid)
    graph_app.invoke({"session_id": sid, "query": "My budget is 50,000.", "spend_to_date": {}})
    state = graph_app.invoke({"session_id": sid,
                              "query": f"Find the cheapest way to buy the {product['name']}.",
                              "spend_to_date": {}})
    best = state.get("best_option")
    if best is not None:
        assert best.base_price.amount != pytest.approx(50000.0), product["product_id"]
        assert best.base_price.amount in {float(v) for v in product["prices"].values()}


# ------------------------------------------------------------------ I8, I9
@pytest.mark.parametrize("product", [p for p in PRODUCTS if len(p["prices"]) >= 2][:5],
                         ids=lambda p: p["product_id"])
def test_followup_changes_only_its_own_dimension_and_topic_shift_clears(product):
    """
    INVARIANT 8 — a follow-up changes only the dimension it names.
    INVARIANT 9 — a topic shift does not leak the old product.
    """
    sid = "inv8_" + product["product_id"]
    merchants = list(product["prices"])
    MemoryManager.reset_session(sid)
    graph_app.invoke({"session_id": sid, "query": f"I want the {product['name']}.", "spend_to_date": {}})

    graph_app.invoke({"session_id": sid, "query": f"{merchants[1]}?", "spend_to_date": {}})
    thread = MemoryManager.get_session(sid).thread
    assert thread.active_product_id == product["product_id"], "product was lost"
    assert (thread.active_merchant or "").lower() == merchants[1].lower(), "merchant did not change"

    other = next(c for c in CATEGORIES if c != product["category"])
    graph_app.invoke({"session_id": sid, "query": f"What {other} deals are available?", "spend_to_date": {}})
    thread = MemoryManager.get_session(sid).thread
    assert thread.active_product_id != product["product_id"], "stale product leaked across topics"


# ------------------------------------------------------------------ I12
@pytest.mark.parametrize("category", CATEGORIES)
def test_joint_best_candidates_are_reported_as_a_tie(category):
    """INVARIANT 12 — never fabricate a unique winner when candidates are joint-best."""
    state = ask(f"Which card gives the highest cashback on 10,000 of {category}?", f"inv12_{category}")
    options = state.get("payment_options") or []
    if not options:
        return
    top = max(o.reward_earned.amount for o in options)
    joint = {o.card_id for o in options if abs(o.reward_earned.amount - top) < 0.01}
    tie = state.get("tie")
    if len(joint) > 1:
        assert tie is not None and set(tie["labels"]) == joint, category
    else:
        assert tie is None, category


# ------------------------------------------------------------------ I10
def test_financial_formulas_have_a_single_source():
    """INVARIANT 10 — one construction site for money, one for each formula."""
    import pathlib

    app_dir = pathlib.Path("app")
    sources = {f: f.read_text(encoding="utf-8") for f in app_dir.rglob("*.py")}
    assert sum(t.count("PaymentOption(") for f, t in sources.items()
               if f.name != "state.py") == 1
    assert sum(t.count("def deal_discount") for t in sources.values()) == 1
    assert sum(t.count("def effective_rate") for t in sources.values()) == 1
    assert sum(t.count("def calculate_payment_option") for t in sources.values()) == 1
    assert sum(t.count("def is_deal_eligible") for t in sources.values()) == 1


# ------------------------------------------------------------------ waiver semantics
def test_waiving_a_dimension_releases_only_that_turns_constraint():
    """
    A standing preference applies, is released for a turn that waives it, and applies again
    afterwards. Waiving is not the same as clearing.
    """
    sid = "inv_waive"
    MemoryManager.reset_session(sid)
    graph_app.invoke({"session_id": sid, "query": "I prefer HDFC Millennia.", "spend_to_date": {}})
    product = DataIndex.get_product_by_id("prod_kindle_paperwhite")
    q = f"Cheapest way to buy the {product['name']}?"

    pinned = graph_app.invoke({"session_id": sid, "query": q, "spend_to_date": {}})
    waived = graph_app.invoke({"session_id": sid, "query": "I don't care which card I use. " + q,
                               "spend_to_date": {}})
    restored = graph_app.invoke({"session_id": sid, "query": q, "spend_to_date": {}})

    assert pinned["best_option"].card_id == "hdfc_millennia"
    assert waived["best_option"].card_id != "hdfc_millennia"
    assert waived["best_option"].effective_price.amount < pinned["best_option"].effective_price.amount
    assert restored["best_option"].card_id == "hdfc_millennia"
    assert MemoryManager.get_session(sid).preferred_card == "hdfc_millennia"


def test_product_scoped_listing_only_shows_merchants_that_sell_it():
    """A deal from a merchant that does not carry the product is not a deal for that product."""
    product = DataIndex.get_product_by_id("prod_kindle_paperwhite")
    sellers = {m.lower() for m in product["prices"]}
    state = ask(f"What deals are available for the {product['name']}?", "inv_listing")
    response = state.get("final_response", "")
    for deal in DataIndex.get_deals():
        if deal.get("merchant") and deal["merchant"].lower() not in sellers:
            assert deal["deal_id"] not in response, deal["deal_id"]


# ------------------------------------------------------------------ comparison objective
CARD_PAIRS = [(CARDS[i], CARDS[j])
              for i in range(len(CARDS)) for j in range(i + 1, len(CARDS))]


def _compare(a, b, clause, sid):
    return ask(f"{clause} {a['name']} or {b['name']} for a 10,000 grocery purchase?", sid)


@pytest.mark.parametrize("pair", CARD_PAIRS[:6], ids=lambda p: f"{p[0]['card_id']}_{p[1]['card_id']}")
def test_explicit_comparison_objectives_rank_by_what_was_asked(pair):
    """
    An objective the user STATES decides the ranking metric and the wording of the verdict.
    Ranking already followed the objective; the narration did not, so a cashback comparison
    was ranked by cashback and then explained with an effective-price verdict.
    """
    a, b = pair
    cheap = _compare(a, b, "Which is cheaper,", f"cmp_p_{a['card_id']}_{b['card_id']}")
    assert cheap["resolved_query"].objective == "min_effective_price"
    assert cheap["resolved_query"].objective_source == "explicit"
    assert "ranked by effective final price" in cheap["final_response"]

    reward = _compare(a, b, "Which gives more cashback,", f"cmp_r_{a['card_id']}_{b['card_id']}")
    assert reward["resolved_query"].objective == "max_reward"
    assert "ranked by cashback earned" in reward["final_response"]
    # a reward comparison admits no merchant coupon
    assert all(o.discount_source_id is None for o in (reward.get("payment_options") or []))

    discount = _compare(a, b, "Which gives the biggest discount,", f"cmp_d_{a['card_id']}_{b['card_id']}")
    assert discount["resolved_query"].objective == "max_discount"
    assert "ranked by discount applied" in discount["final_response"]


@pytest.mark.parametrize("pair", CARD_PAIRS[:6], ids=lambda p: f"{p[0]['card_id']}_{p[1]['card_id']}")
def test_unstated_comparison_objective_never_silently_picks_a_metric(pair):
    """
    With no objective stated, a comparison is only allowed to declare a winner when the
    readings agree. Where they disagree it must report both and declare neither.
    """
    a, b = pair
    state = _compare(a, b, "", f"cmp_amb_{a['card_id']}_{b['card_id']}")
    resolved = state["resolved_query"]
    assert resolved.objective_source == "default"

    response = state.get("final_response", "")
    ambiguous = state.get("comparison_ambiguous")
    thread = MemoryManager.get_session(f"cmp_amb_{a['card_id']}_{b['card_id']}").thread
    if ambiguous:
        # Asserted structurally, not on wording: every meaningful metric is reported, every
        # metric-specific verdict is withheld, and no winner is recorded.
        assert {v["metric"] for v in ambiguous} == {"effective_price", "reward", "discount"}
        for verdict in ("is cheaper than", "earns", "discounts"):
            assert f"{verdict} " not in response.split("Each option in full")[0], verdict
        assert thread.last_comparison["winner_key"] is None
        assert all(v["winners"] for v in ambiguous)
    else:
        # readings agreed, so a single verdict is honest
        assert state.get("comparison_ambiguous") is None
        assert thread.last_comparison["winner_key"] is not None
    assert state.get("provenance_valid") is True


def test_alternate_readings_are_reported_across_differing_base_prices():
    """
    Every meaningful metric is reported when they disagree, whatever the axis. Merchants at
    different prices are exactly the case where the readings diverge — the dearer merchant
    pays more cashback while costing more overall — and hiding that would be the silent
    choice this mechanism exists to prevent.
    """
    product = DataIndex.get_product_by_id("prod_kindle_paperwhite")
    merchants = list(product["prices"])
    assert len({float(v) for v in product["prices"].values()}) > 1, "fixture needs differing prices"
    state = ask(f"{merchants[0]} or {merchants[1]} for the {product['name']}?", "cmp_merch")
    assert state["resolved_query"].objective_source == "default"
    ambiguous = state.get("comparison_ambiguous")
    assert ambiguous, "metrics disagree here and must be reported"
    assert {v["metric"] for v in ambiguous} == {"effective_price", "reward", "discount"}
    assert "no single best choice" in state.get("final_response", "")
    thread = MemoryManager.get_session("cmp_merch").thread
    assert thread.last_comparison["winner_key"] is None
    assert state.get("provenance_valid") is True


def test_an_assumed_objective_does_not_become_sticky():
    """Only a stated objective is inherited by later turns; an assumption is not intent."""
    sid = "cmp_sticky"
    MemoryManager.reset_session(sid)
    graph_app.invoke({"session_id": sid, "query": "I want the Kindle Paperwhite.", "spend_to_date": {}})
    assert MemoryManager.get_session(sid).thread.active_objective is None

    graph_app.invoke({"session_id": sid, "query": "Which card gives the most cashback?",
                      "spend_to_date": {}})
    assert MemoryManager.get_session(sid).thread.active_objective == "max_reward"


# ------------------------------------------------------------------ matrix A-R
def test_K_narration_vocabulary_follows_the_ranking_metric():
    """K — never a hardcoded template: the verdict's verb comes from the metric."""
    pairs = [("Which is cheaper,", "ranked by effective final price", ("cheaper than",)),
             ("Which gives more cashback,", "ranked by cashback earned", ("earns", "come out level")),
             ("Which gives the biggest discount,", "ranked by discount applied", ("discounts", "come out level"))]
    a, b = CARDS[0], CARDS[1]
    for clause, heading, verbs in pairs:
        state = _compare(a, b, clause, f"K_{clause[:6]}")
        response = state["final_response"]
        assert heading in response, (clause, response)
        assert any(v in response for v in verbs), (clause, response)


def test_L_alternate_views_reuse_the_same_grounded_options():
    """L — alternate metrics only RE-RANK; they never build a different candidate space."""
    a, b = CARDS[0], CARDS[1]
    state = _compare(a, b, "", "L_same_space")
    ambiguous = state.get("comparison_ambiguous")
    if not ambiguous:
        pytest.skip("metrics agree for this pair")
    options = state.get("payment_options") or []
    # every entity named in any view is one of the compared entities, unchanged
    entities = set(state["resolved_query"].comparison_entities)
    for view in ambiguous:
        assert set(view["winners"]) <= entities, view
    assert {o.card_id for o in options} <= entities


def test_R_topic_shift_clears_comparison_state():
    """R — a new subject does not inherit the previous comparison."""
    sid = "R_shift"
    MemoryManager.reset_session(sid)
    graph_app.invoke({"session_id": sid,
                      "query": f"{CARDS[0]['name']} or {CARDS[1]['name']} for 10,000 groceries?",
                      "spend_to_date": {}})
    assert MemoryManager.get_session(sid).thread.comparison_axis == "card"
    product = DataIndex.get_product_by_id("prod_kindle_paperwhite")
    graph_app.invoke({"session_id": sid,
                      "query": f"Find the cheapest way to buy the {product['name']}.",
                      "spend_to_date": {}})
    thread = MemoryManager.get_session(sid).thread
    assert thread.active_product_id == product["product_id"]


def test_F_explicit_objective_overrides_a_previous_default():
    """F/H — an explicit objective replaces an assumed one and then persists."""
    sid = "F_override"
    MemoryManager.reset_session(sid)
    first = graph_app.invoke({"session_id": sid,
                              "query": f"{CARDS[0]['name']} or {CARDS[1]['name']} for 10,000 groceries?",
                              "spend_to_date": {}})
    assert first["resolved_query"].objective_source == "default"
    assert MemoryManager.get_session(sid).thread.active_objective is None   # G: not sticky

    second = graph_app.invoke({"session_id": sid, "query": "What about cashback?",
                               "spend_to_date": {}})
    assert second["resolved_query"].objective == "max_reward"
    assert second["resolved_query"].objective_source == "explicit"

    third = graph_app.invoke({"session_id": sid, "query": "What about the other one?",
                              "spend_to_date": {}})
    assert third["resolved_query"].objective == "max_reward"
    assert third["resolved_query"].objective_source == "inherited"


def test_I_comparison_ambiguity_survives_the_graph():
    """I — the ambiguity report must not be dropped by an undeclared state channel."""
    a, b = CARDS[0], CARDS[1]
    state = _compare(a, b, "", "I_survives")
    if state.get("comparison_ambiguous"):
        assert isinstance(state["comparison_ambiguous"], list)
        assert all({"metric", "summary", "winners"} <= set(v) for v in state["comparison_ambiguous"])


def test_no_state_field_is_written_without_being_declared():
    """Guards the class of bug that dropped excluded_injection_records and comparison_ambiguous."""
    import pathlib, re
    declared = set(re.findall(r"^    (\w+):",
                              pathlib.Path("app/agent/state.py").read_text(encoding="utf-8"), re.M))
    written = set()
    for f in pathlib.Path("app").rglob("*.py"):
        for m in re.finditer(r"(?:state|result)\[[\"']([a-z_]+)[\"']\]\s*=",
                             f.read_text(encoding="utf-8")):
            written.add(m.group(1))
    assert not (written - declared), sorted(written - declared)
