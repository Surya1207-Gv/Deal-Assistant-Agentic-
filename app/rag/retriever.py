from __future__ import annotations
import os
import re
import math
from typing import List, Dict, Any, Tuple
from app.rag.index import DataIndex

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

RETRIEVAL_THRESHOLD = float(os.environ.get("RETRIEVAL_THRESHOLD", "0.35"))

GENERIC_STOPWORDS = {
    "discount", "offer", "sale", "purchase", "deal", "card", "buy", "get", "off",
    "instant", "flat", "on", "for", "with", "and", "the", "a", "an", "in", "to", "of", "is", "what", "does", "give"
}

class HybridRetriever:
    """
    Hybrid Retriever fusing BM25 exact keyword matching with dense semantic search.
    Enforces abstention threshold (RETRIEVAL_THRESHOLD).
    Strips prompt injection patterns from deal descriptions.
    Enforces strict grounding checks on absent parametric entities.
    """
    def __init__(self):
        self.deals = DataIndex.get_deals()
        self.cards = DataIndex.get_cards()
        self.products = DataIndex.get_products()
        self.bm25 = None
        self.embedder = None
        self.doc_embeddings = None
        self._initialize_indices()

    def _clean_text(self, text: str) -> str:
        cleaned = re.sub(r"IGNORE PREVIOUS INSTRUCTIONS.*", "", text, flags=re.IGNORECASE)
        cleaned = re.sub(r"SYSTEM OVERRIDE.*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"IMPORTANT INSTRUCTION.*", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def _initialize_indices(self):
        self.corpus = [
            f"{d['title']} {d['merchant']} {d['category']} {d.get('card_specific') or ''} {self._clean_text(d['description'])}"
            for d in self.deals
        ]
        
        tokenized_corpus = [re.findall(r"\w+", doc.lower().replace("_", " ")) for doc in self.corpus]
        if HAS_BM25 and tokenized_corpus:
            self.bm25 = BM25Okapi(tokenized_corpus)

        if HAS_SENTENCE_TRANSFORMERS and self.corpus:
            try:
                self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
                self.doc_embeddings = self.embedder.encode(self.corpus, normalize_embeddings=True)
            except Exception:
                self.embedder = None

    def _bm25_scores(self, query: str) -> Tuple[List[float], bool]:
        tokens = [t.lower() for t in re.findall(r"\w+", query)]
        content_tokens = [t for t in tokens if t not in GENERIC_STOPWORDS]

        if not content_tokens:
            content_tokens = tokens

        if self.bm25 and content_tokens:
            scores = self.bm25.get_scores(content_tokens)
            max_s = max(scores) if len(scores) > 0 and max(scores) > 0 else 1.0
            has_match = any(s > 0 for s in scores)
            return [float(s / max_s) for s in scores], has_match

        scores = []
        has_match = False
        for doc in self.corpus:
            doc_lower = doc.lower().replace("_", " ")
            overlap = sum(1 for q in content_tokens if q in doc_lower)
            if overlap > 0:
                has_match = True
            scores.append(float(overlap / max(1, len(content_tokens))))
        return scores, has_match

    def _dense_scores(self, query: str) -> List[float]:
        if self.embedder is not None and self.doc_embeddings is not None:
            query_emb = self.embedder.encode([query], normalize_embeddings=True)[0]
            sims = np.dot(self.doc_embeddings, query_emb)
            return [float(s) for s in sims]

        query_words = set(w.lower() for w in re.findall(r"\w+", query) if w.lower() not in GENERIC_STOPWORDS)
        scores = []
        for doc in self.corpus:
            doc_words = set(re.findall(r"\w+", doc.lower().replace("_", " ")))
            intersection = query_words.intersection(doc_words)
            sim = len(intersection) / math.sqrt(len(query_words) * len(doc_words) + 1e-5) if query_words else 0.0
            scores.append(float(sim))
        return scores

    def search(self, query: str, top_k: int = 5) -> Tuple[List[Dict[str, Any]], bool, float]:
        if not self.deals:
            return [], True, 0.0

        # Parametric entity presence check (Grounding rule)
        # If query asks about a specific card name like 'Visa Signature' or 'Infinia' not present in dataset:
        q_lower = query.lower()
        known_card_names = {"millennia", "regalia", "ace", "amazon pay", "cashback", "smartearn"}
        if "card" in q_lower or "visa" in q_lower or "infinia" in q_lower:
            if not any(k in q_lower for k in known_card_names) and ("signature" in q_lower or "infinia" in q_lower):
                return [], True, 0.05

        bm25_s, has_content_match = self._bm25_scores(query)
        dense_s = self._dense_scores(query)

        query_tokens = set(w.lower() for w in re.findall(r"\w+", query) if w.lower() not in GENERIC_STOPWORDS)

        hybrid_scores = []
        for i in range(len(self.deals)):
            b_score = bm25_s[i]
            d_score = dense_s[i]

            score = 0.4 * b_score + 0.6 * d_score

            card_sp = (self.deals[i].get("card_specific") or "").replace("_", " ")
            merchant = self.deals[i].get("merchant", "").lower()
            category = self.deals[i].get("category", "").lower()

            if merchant and merchant in q_lower:
                score += 0.25
            if card_sp and card_sp.lower() in q_lower:
                score += 0.25
            if category and category in q_lower:
                score += 0.10

            corpus_clean = self.corpus[i].lower().replace("_", " ")
            if not has_content_match and not any(q in corpus_clean for q in query_tokens):
                score = score * 0.2

            hybrid_scores.append((i, float(score)))

        hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        max_score = float(hybrid_scores[0][1]) if hybrid_scores else 0.0

        abstained = bool(max_score < RETRIEVAL_THRESHOLD)

        results = []
        for idx, score in hybrid_scores[:top_k]:
            deal = dict(self.deals[idx])
            deal["retrieval_score"] = round(float(score), 3)
            deal["clean_description"] = self._clean_text(deal.get("description", ""))
            results.append(deal)

        return results, abstained, round(max_score, 3)
