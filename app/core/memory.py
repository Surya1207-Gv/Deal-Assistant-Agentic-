from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

@dataclass
class SessionMemory:
    session_id: str
    budget: Optional[float] = None
    category: Optional[str] = None
    product: Optional[str] = None
    preferred_card: Optional[str] = None
    spend_to_date: Dict[str, float] = field(default_factory=dict)
    price_watches: Dict[str, float] = field(default_factory=dict)
    history: List[Dict[str, str]] = field(default_factory=list)

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
