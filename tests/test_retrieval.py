import pytest
from app.rag.retriever import HybridRetriever, RETRIEVAL_THRESHOLD

def test_retrieval_exact_keyword_matching():
    retriever = HybridRetriever()
    results, abstained, max_score = retriever.search("HDFC Millennia card discount on groceries", top_k=3)
    assert abstained is False
    assert len(results) > 0
    # Millennia deal or HDFC grocery deal should be top retrieved
    deal_ids = [r["deal_id"] for r in results]
    assert "deal_002" in deal_ids or "deal_017" in deal_ids or "deal_001" in deal_ids

def test_retrieval_abstention_on_unknown_query():
    retriever = HybridRetriever()
    # Query for completely non-existent brand/product/category in dataset
    results, abstained, max_score = retriever.search("Tesla CyberTruck SpaceX Falcon Rocket purchase discount", top_k=3)
    assert abstained is True
    assert max_score < RETRIEVAL_THRESHOLD

def test_retrieval_prompt_injection_sanitized():
    retriever = HybridRetriever()
    results, abstained, max_score = retriever.search("Prompt Injection Test Deal 1 SuperStore", top_k=1)
    if results:
        clean_desc = results[0]["clean_description"]
        assert "IGNORE PREVIOUS INSTRUCTIONS" not in clean_desc
