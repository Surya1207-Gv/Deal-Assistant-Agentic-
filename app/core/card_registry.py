from __future__ import annotations
import time
from typing import Dict, Any, List, Optional
from app.rag.index import DataIndex

class CardPolicyRegistry:
    """
    Dynamic Card Policy & Live Issuer Program Registry.
    Provides runtime policy updates, live issuer API synchronization,
    dynamic multiplier/cap overrides, and policy versioning.
    """
    _live_overrides: Dict[str, Dict[str, Any]] = {}
    _last_sync_timestamp: float = time.time()
    _policy_versions: Dict[str, int] = {}

    @classmethod
    def register_policy_update(
        cls,
        card_id: str,
        base_rate: Optional[float] = None,
        category_multipliers: Optional[Dict[str, float]] = None,
        caps: Optional[Dict[str, Any]] = None,
        min_spend: Optional[float] = None,
        exclusions: Optional[List[str]] = None,
        effective_until: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Dynamically update or override reward policy for a card at runtime.
        """
        cid = card_id.lower()
        curr = DataIndex.get_card_by_id(cid) or {}
        policy = dict(curr)

        if base_rate is not None:
            policy["base_rate"] = float(base_rate)
        if category_multipliers is not None:
            policy["category_multipliers"] = dict(category_multipliers)
        if caps is not None:
            policy["caps"] = dict(caps)
        if min_spend is not None:
            policy["min_spend"] = float(min_spend)
        if exclusions is not None:
            policy["exclusions"] = list(exclusions)
        if effective_until is not None:
            policy["effective_until"] = float(effective_until)

        policy["version"] = cls._policy_versions.get(cid, 1) + 1
        policy["updated_at"] = time.time()

        cls._policy_versions[cid] = policy["version"]
        cls._live_overrides[cid] = policy

        # Update in-memory index
        cards = DataIndex.get_cards()
        for idx, c in enumerate(cards):
            if c["card_id"].lower() == cid:
                cards[idx] = policy
                break

        return policy

    @classmethod
    def get_card_policy(cls, card_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch active policy for a card, incorporating any live issuer overrides.
        """
        cid = card_id.lower()
        if cid in cls._live_overrides:
            override = cls._live_overrides[cid]
            if "effective_until" in override and time.time() > override["effective_until"]:
                del cls._live_overrides[cid]
            else:
                return cls._live_overrides[cid]

        return DataIndex.get_card_by_id(cid)

    @classmethod
    def sync_live_issuer_feed(cls, feed_data: List[Dict[str, Any]]) -> int:
        """
        Bulk sync with external issuer API feed.
        """
        updated_count = 0
        for item in feed_data:
            cid = item.get("card_id")
            if cid:
                cls.register_policy_update(
                    card_id=cid,
                    base_rate=item.get("base_rate"),
                    category_multipliers=item.get("category_multipliers"),
                    caps=item.get("caps"),
                    min_spend=item.get("min_spend"),
                    exclusions=item.get("exclusions")
                )
                updated_count += 1
        cls._last_sync_timestamp = time.time()
        return updated_count

    @classmethod
    def reset_overrides(cls):
        """
        Reset all runtime overrides back to authoritative snapshot.
        """
        cls._live_overrides.clear()
        cls._policy_versions.clear()
        DataIndex.load_all(force=True)
