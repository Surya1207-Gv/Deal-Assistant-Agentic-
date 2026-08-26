from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from app.core.thread import ConversationThread

@dataclass
class SessionMemory:
    session_id: str
    budget: Optional[float] = None
    category: Optional[str] = None
    product: Optional[str] = None
    last_deal_id: Optional[str] = None
    last_merchant: Optional[str] = None
    preferred_card: Optional[str] = None
    # `preferred_card` above is the user's PERSISTED preference (only set by genuine
    # persistent-intent language, e.g. "I prefer SBI"). `active_card` is a SEPARATE,
    # weaker concept: the card currently "in play" for the ongoing purchase thread,
    # which a one-turn hypothetical ("what if I use SBI?") is also allowed to set so a
    # follow-up ("how much do I actually save?") stays consistent with it — WITHOUT that
    # hypothetical becoming a persistent cross-topic preference. Defaults to whatever
    # preferred_card is; diverges only within the current purchase thread; resets to
    # preferred_card whenever the product context genuinely changes.
    active_card: Optional[str] = None
    active_amount: Optional[float] = None
    active_amount_basis: Optional[str] = None  # "purchase" | "reward"
    category_preferences: Dict[str, str] = field(default_factory=dict)
    spend_to_date: Dict[str, float] = field(default_factory=dict)
    price_watches: Dict[str, float] = field(default_factory=dict)
    history: List[Dict[str, str]] = field(default_factory=list)
    # The structured "what are we talking about right now" context. The flat fields above
    # (product / last_merchant / active_card / budget ...) remain as the PERSISTENT and
    # back-compatible view; `thread` is the richer per-topic working set that short and
    # elliptical follow-ups resolve against. See app/core/thread.py.
    thread: ConversationThread = field(default_factory=ConversationThread)
    other_relevant_constraints: Dict[str, Any] = field(default_factory=dict)

    @property
    def conversation_state(self) -> Dict[str, Any]:
        """
        A DERIVED VIEW of the persistent session state, not a second copy of it.

        It used to be an independently stored dict holding its own `budget` and
        `preferred_card`. Two stores meant two values: any code path that updated one and
        not the other left the session disagreeing with itself, and the resolver read the
        stale copy — a budget could be raised and the next answer still be rejected against
        the old ceiling. Deriving it makes divergence impossible (spec section 1).
        """
        return {
            "budget": self.budget,
            "preferred_card": self.preferred_card,
            "category_preferences": dict(self.category_preferences),
            "other_relevant_constraints": dict(self.other_relevant_constraints),
        }

    @conversation_state.setter
    def conversation_state(self, value: Dict[str, Any]) -> None:
        """Writes through to the authoritative fields, so assignment cannot fork the state."""
        value = value or {}
        self.budget = value.get("budget")
        self.preferred_card = value.get("preferred_card")
        self.category_preferences = dict(value.get("category_preferences") or {})
        self.other_relevant_constraints = dict(value.get("other_relevant_constraints") or {})

class MemoryManager:
    """
    Manages session memory across multiple turns.
    Tracks category context, budget overrides, card preferences, spend-to-date headroom, and price watches.
    """
    _sessions: Dict[str, SessionMemory] = {}

    @classmethod
    def get_session(cls, session_id: str) -> SessionMemory:
        if session_id not in cls._sessions:
            cls._sessions[session_id] = SessionMemory(session_id=session_id)
        return cls._sessions[session_id]

    @classmethod
    def update_budget(cls, session_id: str, budget: float) -> SessionMemory:
        session = cls.get_session(session_id)
        session.budget = budget
        return session

    @classmethod
    def update_category(cls, session_id: str, category: str) -> SessionMemory:
        session = cls.get_session(session_id)
        session.category = category
        return session

    @classmethod
    def update_card(cls, session_id: str, card_id: str) -> SessionMemory:
        session = cls.get_session(session_id)
        session.preferred_card = card_id
        session.thread.preferred_card = card_id
        return session

    @classmethod
    def update_category_preference(cls, session_id: str, category: str, card_id: str) -> SessionMemory:
        session = cls.get_session(session_id)
        session.category_preferences[category] = card_id
        return session

    @classmethod
    def set_conversation_state(cls, session_id: str, state: Dict[str, Any]) -> SessionMemory:
        session = cls.get_session(session_id)
        session.conversation_state = {
            "budget": state.get("budget"),
            "preferred_card": state.get("preferred_card"),
            "category_preferences": dict(state.get("category_preferences", {})),
            "other_relevant_constraints": dict(state.get("other_relevant_constraints", {})),
        }
        return session

    @classmethod
    def update_last_entities(
        cls,
        session_id: str,
        product_id: Optional[str] = None,
        deal_id: Optional[str] = None,
        merchant: Optional[str] = None,
    ) -> SessionMemory:
        """
        Records the entities actually referenced in this turn so a later follow-up
        ("that deal", "the same product", "that merchant") can be resolved without
        the user repeating themselves. Only overwrites a field when a new value for
        it was actually resolved this turn — an unrelated follow-up should not erase it.
        """
        session = cls.get_session(session_id)
        if product_id:
            session.product = product_id
        if deal_id:
            session.last_deal_id = deal_id
        if merchant:
            session.last_merchant = merchant
        return session

    @classmethod
    def clear_card(cls, session_id: str) -> SessionMemory:
        """
        Clears ONLY the card dimension — persistent preference and in-thread active card.
        Product/category/amount context in the thread is deliberately left intact so a
        follow-up ("what would you recommend now?") still knows what we were shopping for.
        """
        session = cls.get_session(session_id)
        session.preferred_card = None
        session.active_card = None
        session.thread.clear_card_dimension()
        return session

    @classmethod
    def set_active_card(cls, session_id: str, card_id: Optional[str]) -> SessionMemory:
        """
        Sets the in-thread active card (hypothetical or persisted) WITHOUT touching the
        persisted preferred_card. Pass None to clear it (e.g. on an explicit clear
        directive, or when the purchase topic changes and the hypothetical should not
        carry over).
        """
        session = cls.get_session(session_id)
        session.active_card = card_id
        return session

    @classmethod
    def set_active_amount(cls, session_id: str, amount: Optional[float], basis: Optional[str]) -> SessionMemory:
        session = cls.get_session(session_id)
        session.active_amount = amount
        session.active_amount_basis = basis
        return session

    @classmethod
    def clear_budget(cls, session_id: str) -> SessionMemory:
        """Clears ONLY the budget dimension; the rest of the thread survives."""
        session = cls.get_session(session_id)
        session.budget = None
        session.thread.clear_budget_dimension()
        return session

    @classmethod
    def record_spend(cls, session_id: str, card_id: str, category: str, amount: float):
        session = cls.get_session(session_id)
        monthly_key = f"{card_id}_monthly"
        cat_key = f"{card_id}_{category}"
        session.spend_to_date[monthly_key] = session.spend_to_date.get(monthly_key, 0.0) + amount
        session.spend_to_date[cat_key] = session.spend_to_date.get(cat_key, 0.0) + amount

    @classmethod
    def reset_session(cls, session_id: str):
        if session_id in cls._sessions:
            del cls._sessions[session_id]
