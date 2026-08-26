from __future__ import annotations

"""
Active Conversation Thread — the structured representation of "what we are currently
talking about".

WHY THIS EXISTS
---------------
Earlier versions inferred conversational continuity from linguistic markers ("what if",
"how much do I save", "this product"). That approach cannot generalize: a real user says
"Amazon?", "And Croma?", "SBI?", "15k?" — none of which contain any marker at all, yet all
of which obviously continue the current purchase.

The generalized mechanism here is DIMENSION-WISE STATE TRANSITION:

  * A turn names entities on zero or more DIMENSIONS (product, merchant, card, deal, amount).
  * Dimensions the turn names explicitly are UPDATED.
  * Dimensions the turn does not name are INHERITED from the active thread.
  * Unless a TOPIC SHIFT is detected, in which case the stale dimensions are dropped.

That single rule covers "Amazon?" (updates merchant, inherits product+card+amount),
"SBI?" (updates card, inherits the rest), "15k?" (updates amount, inherits the rest) and
"How much?" (updates nothing, inherits everything) without one phrase-specific branch.

Topic shift is decided from ENTITY SEMANTICS (does the newly named product/category
conflict with the active one?), never from wording — so it generalizes to any dataset
entity.

FINANCIAL SAFETY INVARIANT
--------------------------
This module decides only WHAT the user is referring to. It never holds or derives prices,
discounts, rates, caps or eligibility. `last_recommendation` caches numbers that were
already produced by the deterministic engine together with the provenance trace that
justified them; it is a memo of a past computation, never a source of new financial fact.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from app.rag.index import DataIndex


# Canonical category nouns. Derived from the dataset's own `category` values plus a
# taxonomy-level synonym map. Deliberately EXCLUDES merchant and brand names: a merchant
# is cross-category (Amazon sells both electronics and groceries), so naming a merchant
# must never look like a category change. This distinction is what lets "And Croma?"
# continue an electronics thread while "cashback on groceries?" starts a new one.
_CATEGORY_SYNONYMS: Dict[str, str] = {
    "grocery": "groceries",
    "groceries": "groceries",
    "supermarket": "groceries",
    "electronics": "electronics",
    "electronic": "electronics",
    "gadget": "electronics",
    "gadgets": "electronics",
    "travel": "travel",
    "flight": "travel",
    "flights": "travel",
    "hotel": "travel",
    "hotels": "travel",
    "trip": "travel",
    "shopping": "shopping",
    "apparel": "shopping",
    "clothes": "shopping",
    "clothing": "shopping",
    "fashion": "shopping",
    "footwear": "shopping",
    "bills": "bills",
    "bill": "bills",
    "utility": "bills",
    "electricity": "bills",
    "recharge": "bills",
    "food": "food",
    "dining": "food",
    "restaurant": "food",
    "meal": "food",
    "takeout": "food",
}


def canonical_category(word: str) -> Optional[str]:
    return _CATEGORY_SYNONYMS.get(word.lower())


def known_categories() -> Set[str]:
    """Canonical category values actually present in the dataset."""
    cats: Set[str] = set()
    for p in DataIndex.get_products():
        if p.get("category"):
            cats.add(p["category"].lower())
    for d in DataIndex.get_deals():
        if d.get("category"):
            cats.add(d["category"].lower())
    return cats


@dataclass
class TurnEntities:
    """What THIS turn explicitly named. `None`/empty means 'not mentioned this turn'."""
    product: Optional[Dict[str, Any]] = None
    merchant: Optional[str] = None
    card: Optional[Dict[str, Any]] = None
    deal: Optional[Dict[str, Any]] = None
    amount: Optional[float] = None
    amount_basis: Optional[str] = None          # "purchase" | "reward" | "budget"
    category_word: Optional[str] = None         # a TRUE category noun said out loud
    comparison_merchants: List[str] = field(default_factory=list)
    comparison_cards: List[str] = field(default_factory=list)
    comparison_deals: List[str] = field(default_factory=list)
    comparison_products: List[str] = field(default_factory=list)

    def names_anything(self) -> bool:
        return any([
            self.product, self.merchant, self.card, self.deal,
            self.amount is not None, self.category_word,
        ])


@dataclass
class ContinuityDecision:
    """Outcome of topic-continuity assessment for one turn."""
    continue_thread: bool
    reason: str
    reset_dimensions: Set[str] = field(default_factory=set)

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.continue_thread


@dataclass
class ConversationThread:
    """
    The active purchase/comparison context.

    Structured state only — never raw chat text. Anything a follow-up might refer back to
    lives here as a typed field so it can be resolved deterministically.
    """
    active_product_id: Optional[str] = None
    active_product_name: Optional[str] = None
    active_category: Optional[str] = None
    active_subcategory: Optional[str] = None
    active_merchant: Optional[str] = None
    active_card: Optional[str] = None
    active_deal_id: Optional[str] = None
    active_amount: Optional[float] = None
    active_amount_basis: Optional[str] = None   # "purchase" | "reward"
    active_budget: Optional[float] = None
    # What the conversation is currently optimising for (see agent/resolver.Objective).
    # A follow-up that states no objective of its own keeps optimising for this one, which
    # is what makes "what if I use SBI?" stay a reward question after a reward question and
    # a price question after a price question.
    active_objective: Optional[str] = None

    # PERSISTENT user preference ("I prefer SBI"), as opposed to `active_card`, which is
    # the card currently in play for this purchase and which a one-turn hypothetical
    # ("what if I use HDFC?") is allowed to move. A hypothetical must never write here;
    # a topic change restores `active_card` from here. See spec section 14.
    preferred_card: Optional[str] = None

    # Memo of the last deterministic computation, so "why?", "what was the discount?",
    # "what's the runner-up?" can be answered without any fresh global retrieval.
    # `last_trace` carries the ORIGINAL provenance Values that justified these numbers, so
    # an explanation re-states already-grounded figures instead of asserting new ones.
    last_recommendation: Optional[Dict[str, Any]] = None
    last_effective_price: Optional[float] = None
    last_discount: Optional[float] = None
    last_reward: Optional[float] = None
    last_trace: List[Any] = field(default_factory=list)
    last_citations: List[str] = field(default_factory=list)

    # Active comparison ("Amazon or Croma?") so a later "which is cheapest?" ranks the
    # alternatives actually under discussion rather than starting an unrelated comparison.
    comparison_entities: List[str] = field(default_factory=list)
    comparison_axis: Optional[str] = None       # "merchant" | "card" | "deal" | "product"

    # Structured memo of the last COMPARISON (every alternative that was ranked, with the
    # deterministic figures each one produced, plus the winner). Distinct from
    # `last_recommendation`, which memoizes only the single option that won: "which one was
    # cheaper?" / "why wasn't <loser> cheaper?" need the losing rows too. See spec
    # section 13.
    # The visible deals the last recommendation could NOT use, with the reason and the
    # figures that reason quotes. Kept so "why didn't you use the 1,500-off deal?" is
    # answered from what was already computed instead of being mistaken for a new purchase.
    last_rejected_deals: List[Dict[str, Any]] = field(default_factory=list)

    last_comparison: Optional[Dict[str, Any]] = None
    last_comparison_turn: int = -1
    last_recommendation_turn: int = -1

    turn_index: int = 0

    # ---------------------------------------------------------------- introspection
    def has_context(self) -> bool:
        return bool(self.active_product_id or self.active_category or self.active_amount is not None)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "active_product_id": self.active_product_id,
            "active_product_name": self.active_product_name,
            "active_category": self.active_category,
            "active_subcategory": self.active_subcategory,
            "active_merchant": self.active_merchant,
            "active_card": self.active_card,
            "active_deal_id": self.active_deal_id,
            "active_amount": self.active_amount,
            "active_amount_basis": self.active_amount_basis,
            "active_budget": self.active_budget,
            "preferred_card": self.preferred_card,
            "comparison_entities": list(self.comparison_entities),
            "comparison_axis": self.comparison_axis,
            "has_comparison_memo": self.last_comparison is not None,
            "has_recommendation_memo": self.last_recommendation is not None,
            "turn_index": self.turn_index,
        }

    # ---------------------------------------------------------------- continuity
    def assess_continuity(self, ents: TurnEntities) -> ContinuityDecision:
        """
        Decide whether `ents` continues this thread or starts a new topic.

        Semantic continuity, not phrasing. A turn breaks continuity when the entities it
        names are INCOMPATIBLE with the active context — a different product, or a
        category noun that contradicts the active category. Merely naming a merchant, a
        card or an amount is always compatible: those are dimensions OF the current
        purchase, not new subjects.
        """
        if not self.has_context():
            return ContinuityDecision(False, "no_active_context")

        # A different product is named outright -> genuinely a new subject. Dimensions
        # that belonged to the old product (merchant/deal/amount) must not carry over;
        # the card is a user-level choice and survives (handled by caller policy).
        if ents.product and ents.product.get("product_id") != self.active_product_id:
            return ContinuityDecision(
                False, "different_product",
                reset_dimensions={"merchant", "deal", "amount"},
            )

        # An explicit category noun contradicting the active category -> new topic.
        # (Only TRUE category nouns reach here; merchant/brand names never do.)
        if ents.category_word:
            said = canonical_category(ents.category_word)
            if said and self.active_category and said != self.active_category:
                return ContinuityDecision(
                    False, "different_category",
                    reset_dimensions={"product", "merchant", "deal", "amount"},
                )

        # A named deal that belongs to another category is likewise a new subject.
        if ents.deal:
            d_cat = (ents.deal.get("category") or "").lower()
            if d_cat and self.active_category and d_cat != self.active_category:
                return ContinuityDecision(
                    False, "deal_category_mismatch",
                    reset_dimensions={"product", "merchant", "amount"},
                )

        return ContinuityDecision(True, "compatible")

    # ---------------------------------------------------------------- transition
    def merge_turn(self, ents: TurnEntities, decision: ContinuityDecision,
                   persistent_card: Optional[str] = None) -> Dict[str, bool]:
        """
        Apply one turn to the thread, dimension by dimension.

        Explicitly named dimensions are updated; unnamed ones are inherited (or dropped
        when the continuity decision says they are stale). Returns a map of which
        dimensions ended up INHERITED rather than stated, so the caller can tell how much
        of the resolved query came from context.
        """
        inherited = {"product": False, "merchant": False, "card": False, "amount": False}
        # The thread is the canonical view of conversational state, so the user's standing
        # preference lives here too — `active_card` may diverge from it for the duration of
        # a hypothetical, and falls back to it when the subject changes.
        if persistent_card is not None:
            self.preferred_card = persistent_card

        if not decision.continue_thread:
            for dim in decision.reset_dimensions:
                if dim == "product":
                    self.active_product_id = None
                    self.active_product_name = None
                    self.active_subcategory = None
                elif dim == "merchant":
                    self.active_merchant = None
                elif dim == "deal":
                    self.active_deal_id = None
                elif dim == "amount":
                    self.active_amount = None
                    self.active_amount_basis = None
            # A brand-new subject drops any one-off hypothetical card back to the user's
            # standing preference (persistent state outlives a scenario, never vice versa).
            if decision.reason in ("different_product", "different_category"):
                self.active_card = persistent_card
                self.comparison_entities = []
                self.comparison_axis = None

        # --- product
        if ents.product:
            self.active_product_id = ents.product.get("product_id")
            self.active_product_name = ents.product.get("name")
            self.active_category = (ents.product.get("category") or self.active_category)
            self.active_subcategory = ents.product.get("subcategory")
        elif self.active_product_id and decision.continue_thread:
            inherited["product"] = True

        # --- category (an explicit category noun updates it even with no product)
        if ents.category_word:
            said = canonical_category(ents.category_word)
            if said:
                self.active_category = said

        # --- merchant. Naming one side of an OPEN merchant comparison refers to it WITHIN
        # that comparison ("why wasn't Croma cheaper?"); it does not pin the merchant
        # dimension, which would quietly collapse the comparison to a single side.
        if ents.merchant:
            if not self._is_open_comparison_member("merchant", ents.merchant):
                self.active_merchant = ents.merchant
        elif self.active_merchant and decision.continue_thread:
            inherited["merchant"] = True

        # --- card (same rule on the card axis)
        if ents.card:
            if not self._is_open_comparison_member("card", ents.card.get("card_id")):
                self.active_card = ents.card.get("card_id")
        elif self.active_card and decision.continue_thread:
            inherited["card"] = True

        # --- deal
        if ents.deal:
            self.active_deal_id = ents.deal.get("deal_id")

        # --- amount vs budget. These are DIFFERENT concepts and are stored separately
        # (spec section 15): "what if I spend 15k?" moves the figure this calculation runs
        # on; "my budget is 15k" declares a persistent ceiling and must not become the
        # amount being spent. A budget statement therefore never touches `active_amount`.
        if ents.amount is not None and ents.amount_basis == "budget":
            self.active_budget = ents.amount
            if self.active_amount is not None and decision.continue_thread:
                inherited["amount"] = True
        elif ents.amount is not None:
            self.active_amount = ents.amount
            self.active_amount_basis = ents.amount_basis or "purchase"
        elif self.active_amount is not None and decision.continue_thread:
            inherited["amount"] = True

        # --- comparison bookkeeping.
        # Naming two or more peers on one dimension opens a comparison along that AXIS.
        # The axis then persists until the user opens a comparison on a different axis, or
        # settles the current axis onto a single entity. Naming a single card while
        # comparing merchants applies that card TO the merchant comparison; it neither
        # turns it into a card comparison nor closes it.
        if len(ents.comparison_products) >= 2:
            self.comparison_entities = list(ents.comparison_products)
            self.comparison_axis = "product"
        elif len(ents.comparison_merchants) >= 2:
            self.comparison_entities = list(ents.comparison_merchants)
            self.comparison_axis = "merchant"
            # The comparison spans these merchants, so no single one is "the" merchant.
            self.active_merchant = None
        elif len(ents.comparison_cards) >= 2:
            self.comparison_entities = list(ents.comparison_cards)
            self.comparison_axis = "card"
        elif len(ents.comparison_deals) >= 2:
            self.comparison_entities = list(ents.comparison_deals)
            self.comparison_axis = "deal"
        elif self.comparison_axis and self._turn_leaves_axis(ents):
            # The user named a single entity on the compared axis that is NOT one of the
            # alternatives under discussion — they have moved on, so the comparison ends.
            # Naming one of the compared entities does the opposite: it refers INTO the
            # comparison ("why wasn't Croma cheaper?") and keeps it alive.
            self.comparison_entities = []
            self.comparison_axis = None

        self.turn_index += 1
        return inherited

    def _open_comparison_set(self) -> Set[str]:
        return {str(e).lower() for e in self.comparison_entities}

    def _is_open_comparison_member(self, axis: str, entity: Optional[str]) -> bool:
        """True when `entity` is one of the alternatives in the open comparison on `axis`."""
        if not entity or self.comparison_axis != axis or len(self.comparison_entities) < 2:
            return False
        return str(entity).lower() in self._open_comparison_set()

    def _turn_leaves_axis(self, ents: TurnEntities) -> bool:
        """
        True when this turn names exactly one entity on the compared axis and that entity
        is NOT part of the comparison — i.e. the user has changed the subject on that
        dimension rather than referring to one of the alternatives.
        """
        named: List[str] = []
        if self.comparison_axis == "merchant":
            named = list(ents.comparison_merchants)
        elif self.comparison_axis == "card":
            named = list(ents.comparison_cards)
        elif self.comparison_axis == "deal":
            named = list(ents.comparison_deals)
        elif self.comparison_axis == "product":
            named = list(ents.comparison_products)

        if len(named) != 1:
            return False
        return named[0].lower() not in self._open_comparison_set()

    # ---------------------------------------------------------------- recommendation memo
    def record_recommendation(self, *, card_id: Optional[str], merchant: Optional[str],
                              deal_id: Optional[str], base_price: Optional[float],
                              discount: Optional[float], reward: Optional[float],
                              effective_price: Optional[float],
                              runner_up_card: Optional[str] = None,
                              runner_up_effective: Optional[float] = None,
                              base_price_after_discount: Optional[float] = None,
                              cap_hit: bool = False,
                              cap_explanation: Optional[str] = None,
                              product_name: Optional[str] = None,
                              trace: Optional[List[Any]] = None,
                              citations: Optional[List[str]] = None) -> None:
        """
        Memoize a computation the deterministic engine already performed, so follow-ups
        like "why?" / "what was the discount?" are answered from the same numbers rather
        than by recomputing or, worse, re-retrieving something unrelated.
        """
        self.last_recommendation = {
            "card_id": card_id,
            "merchant": merchant,
            "deal_id": deal_id,
            "base_price": base_price,
            "price_after_discount": base_price_after_discount,
            "discount": discount,
            "reward": reward,
            "effective_price": effective_price,
            "runner_up_card": runner_up_card,
            "runner_up_effective": runner_up_effective,
            "cap_hit": cap_hit,
            "cap_explanation": cap_explanation,
            "product_name": product_name,
        }
        self.last_effective_price = effective_price
        self.last_discount = discount
        self.last_reward = reward
        kept_trace, kept_citations = list(self.last_trace), list(self.last_citations)
        self.last_trace = list(trace or [])
        self.last_citations = list(citations or [])
        self.last_recommendation_turn = self.turn_index

        # Deliberately NOT written here: active_merchant / active_card / active_deal_id.
        #
        # Those dimensions describe what the USER has chosen to talk about, and the
        # candidate space of every later turn is built from them. Writing the WINNER back
        # into them narrows the next turn to whatever we happened to recommend: after
        # "Compare Amazon and Croma" (won by Amazon on ICICI), the next "which is cheaper?"
        # would price BOTH merchants on ICICI only, quietly changing the comparison. That
        # feedback loop was the root cause of the merchant-comparison bug.
        #
        # The winning card/merchant/deal are fully preserved in `last_recommendation` above,
        # which is what explanations read; nothing needs them as active state.

        # A comparison recorded on this same turn is the richer memo — it holds the losing
        # alternatives too — so its merged trace and citations must not be replaced by the
        # winner's alone, or an explanation could no longer ground the loser's figures.
        if self.last_comparison_turn == self.turn_index:
            self.last_trace = kept_trace
            self.last_citations = kept_citations

    # ---------------------------------------------------------------- comparison memo
    def record_comparison(self, *, axis: str, rows: List[Dict[str, Any]],
                          winner_key: Optional[str],
                          product_name: Optional[str] = None,
                          metric: str = "effective_price",
                          trace: Optional[List[Any]] = None,
                          citations: Optional[List[str]] = None) -> None:
        """
        Memoize a completed comparison: every alternative that was ranked, each with the
        figures the deterministic engine produced for it, plus which one won.

        `rows` are already-computed structured records (see agent/candidates.py), never
        free text and never anything an LLM produced. Keeping the LOSING rows is what makes
        "which one was cheaper?", "why wasn't <loser> cheaper?" and "what was the runner-up?"
        answerable without recomputing or re-retrieving anything.
        """
        self.last_comparison = {
            "axis": axis,
            "metric": metric,
            "rows": [dict(r) for r in rows],
            "winner_key": winner_key,
            "product_name": product_name,
        }
        self.last_comparison_turn = self.turn_index
        if trace is not None:
            self.last_trace = list(trace)
        if citations is not None:
            self.last_citations = list(citations)

    def comparison_is_current(self) -> bool:
        """True when the newest memo we hold is a comparison rather than a lone winner."""
        return (
            self.last_comparison is not None
            and self.last_comparison_turn >= self.last_recommendation_turn
        )

    def clear_card_dimension(self) -> None:
        """
        Drop BOTH card concepts. Every other dimension (product, merchant, comparison,
        budget, recommendation memo) survives untouched — a clear directive names exactly
        one dimension and must not silently reset the conversation (spec section 14).
        """
        self.active_card = None
        self.preferred_card = None

    def clear_budget_dimension(self) -> None:
        self.active_budget = None
