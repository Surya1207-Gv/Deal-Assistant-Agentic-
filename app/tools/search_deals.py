from __future__ import annotations
from typing import List, Dict, Any, Optional
from app.rag.retriever import HybridRetriever
from app.rag.reranker import Reranker

_retriever: Optional[HybridRetriever] = None

def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever

def search_deals(query: str, category: Optional[str] = None, merchant: Optional[str] = None) -> Dict[str, Any]:
    """
    Tool 1: Queries hybrid retriever for active promotional deals matching query.
    """
    retriever = get_retriever()
    search_query = query
    if category:
        search_query += f" {category}"
    if merchant:
        search_query += f" {merchant}"

    raw_results, abstained, max_score = retriever.search(search_query, top_k=6)
    reranked = Reranker.rerank(search_query, raw_results, top_n=3)

    return {
        "query": query,
        "deals": reranked,
        "abstained": abstained,
        "max_score": max_score,
        "deal_ids": [d["deal_id"] for d in reranked]
    }
