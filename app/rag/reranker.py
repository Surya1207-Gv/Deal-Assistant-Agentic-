from __future__ import annotations
import os
import re
from typing import List, Dict, Any

class Reranker:
    """
    Cross-Feature Reranker.
    When ENABLE_RERANKER=true, re-scores retrieved candidates using fine-grained
    entity alignment (merchant exact match, category fit, card specific offer match).
    When ENABLE_RERANKER=false, returns raw un-reranked candidates.
    """
    @classmethod
    def rerank(cls, query: str, records: List[Dict[str, Any]], top_n: int = 3) -> List[Dict[str, Any]]:
        enable_reranker = os.environ.get("ENABLE_RERANKER", "true").lower() == "true"
        if not records or not enable_reranker:
            return records[:top_n]

        query_tokens = set(w.lower() for w in re.findall(r"\w+", query))

        def score_record(r: Dict[str, Any]) -> float:
            base_score = r.get("retrieval_score", 0.5)
            merchant = r.get("merchant", "").lower()
            category = r.get("category", "").lower()
            card_sp = (r.get("card_specific") or "").lower().replace("_", " ")

            merchant_boost = 0.35 if merchant and merchant in query_tokens else 0.0
            category_boost = 0.20 if category and category in query.lower() else 0.0
            card_boost = 0.30 if card_sp and any(t in query.lower() for t in card_sp.split()) else 0.0

            return base_score + merchant_boost + category_boost + card_boost

        ranked = sorted(records, key=score_record, reverse=True)
        return ranked[:top_n]
