from __future__ import annotations
import os
import re
import math
from typing import List, Dict, Any, Tuple
from app.rag.index import DataIndex, GENERIC_WORDS

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

    def _is_ungrounded_entity(self, query: str) -> bool:
        """
        Does this query name a product, service or card the catalogue does not contain?

        Entirely case-folded and entirely data-driven. The two vocabularies come from
        DataIndex (see `entity_tokens` / `catalogue_tokens`) — there is no list of things to
        reject, and nothing is refused for being absent from a hand-written set.

        The decision runs in two steps:

          0. ANCHOR. If any word in the query identifies a catalogue record — a product,
             merchant, card, category or deal id — the query is about something we hold, and
             we stop. This is what keeps a legitimate request grounded no matter how it is
             phrased or capitalised.

          Only for a query with no anchor at all:

          a. An explicit purchase phrasing whose object is entirely unknown to the dataset
             ("cheapest way to buy a Dyson vacuum cleaner"). Catches single-word names.

          b. A RUN of consecutive words the dataset has never used anywhere, and which are
             not ordinary connective or shopping words. Two in a row is a name for something
             out of scope ("tesla cybertruck", "visa signature", "peloton bike"). A
             recognised word BREAKS the run rather than being skipped, so ordinary phrasing
             never accumulates into a false name.

        This replaced a rule that looked for runs of CAPITALISED words. Capitalisation is
        orthography, not evidence: it vanished in a lower-cased request and became
        meaningless in a Title-Cased one, so the same question was in scope or out of scope
        depending only on which shift key the user pressed.
        """
        query_tokens = re.findall(r"[a-z0-9]+", query.lower())
        if not query_tokens:
            return False

        # 0. Anchor: does anything here identify a record we hold?
        if any(t in DataIndex.entity_tokens() for t in query_tokens):
            return False

        vocabulary = DataIndex.catalogue_tokens()

        def unseen(token: str) -> bool:
            return (
                len(token) >= 3
                and not token.isdigit()
                and token not in vocabulary
                and token not in GENERIC_WORDS
                and token not in GENERIC_STOPWORDS
            )

        # (a) An explicit purchase phrasing with an object the dataset has never used.
        action_match = re.search(
            r"(?:cheapest|best price for|buy|purchase|way to buy|price for|drops below|"
            r"promo code for|deal for|quota for|compare prices for)\s+([a-zA-Z0-9\s'-]+)",
            query, re.IGNORECASE,
        )
        if action_match:
            candidate_tokens = [
                t for t in re.findall(r"[a-z0-9]+", action_match.group(1).strip().lower())
                if t not in GENERIC_STOPWORDS and len(t) > 2
            ]
            # "drops below RS 85000" captures a PRICE, not a name; a candidate with no
            # alphabetic token names nothing.
            if candidate_tokens and any(not t.isdigit() for t in candidate_tokens):
                if all(unseen(t) for t in candidate_tokens):
                    return True

        # (b) A run of consecutive unseen words.
        run = 0
        for token in query_tokens:
            if unseen(token):
                run += 1
                if run >= 2:
                    return True
            else:
                run = 0

        return False

    def search(self, query: str, category: str | None = None, top_k: int = 5) -> Tuple[List[Dict[str, Any]], bool, float]:
        if not self.deals:
            return [], True, 0.0

        # Dynamic Parametric Entity Grounding Check (No hardcoded entity lists)
        if self._is_ungrounded_entity(query):
            return [], True, 0.0

        q_lower = query.lower()

        # Dynamic Category Resolution from matching product in products.json
        if not category:
            for p in self.products:
                p_name = p["name"].lower()
                p_tokens = set(re.findall(r"[a-z0-9]+", p_name))
                q_tokens = set(re.findall(r"[a-z0-9]+", q_lower))
                if p_name in q_lower or (len(p_tokens.intersection(q_tokens)) >= 2 and not q_tokens.isdisjoint(p_tokens)):
                    category = p.get("category")
                    break

        if not category:
            for d in self.deals:
                d_cat = d.get("category", "").lower()
                if d_cat and d_cat in q_lower:
                    category = d_cat
                    break

        effective_query = f"{category} {query}" if category and (category not in q_lower) else query

        bm25_s, has_content_match = self._bm25_scores(effective_query)
        dense_s = self._dense_scores(effective_query)

        query_tokens = set(w.lower() for w in re.findall(r"\w+", effective_query) if w.lower() not in GENERIC_STOPWORDS)

        hybrid_scores = []
        for i in range(len(self.deals)):
            deal_cat = self.deals[i].get("category", "").lower()
            # Category Isolation Filter: If query category is known, exclude non-matching categories (unless universal)
            if category and deal_cat and deal_cat not in ["all", "any"] and deal_cat != category.lower():
                continue

            b_score = bm25_s[i]
            d_score = dense_s[i]

            score = 0.4 * b_score + 0.6 * d_score

            card_sp = (self.deals[i].get("card_specific") or "").replace("_", " ")
            merchant = self.deals[i].get("merchant", "").lower()

            if merchant and merchant in q_lower:
                score += 0.25
            if card_sp and card_sp.lower() in q_lower:
                score += 0.25
            if deal_cat and (deal_cat in q_lower or (category and deal_cat == category.lower())):
                score += 0.20

            corpus_clean = self.corpus[i].lower().replace("_", " ")
            if not has_content_match and not any(q in corpus_clean for q in query_tokens):
                score = score * 0.2

            hybrid_scores.append((i, float(score)))

        hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        max_score = float(hybrid_scores[0][1]) if hybrid_scores else 0.0

        abstained = bool(max_score < RETRIEVAL_THRESHOLD or len(hybrid_scores) == 0)

        results = []
        for idx, score in hybrid_scores[:top_k]:
            deal = dict(self.deals[idx])
            deal["retrieval_score"] = round(float(score), 3)
            deal["clean_description"] = self._clean_text(deal.get("description", ""))
            results.append(deal)

        return results, abstained, round(max_score, 3)
