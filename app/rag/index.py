import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set

# Words that carry no entity identity: articles, prepositions, and the promotional filler
# that appears in almost every deal title. Used to keep ordinary phrasing from being read as
# a name. This is a closed linguistic set, not a list of things to reject — nothing is ever
# refused for BEING in it.
GENERIC_WORDS: Set[str] = {
    "the", "a", "an", "and", "or", "of", "for", "with", "from", "in", "on", "at", "to",
    "is", "are", "was", "were", "be", "been", "it", "its", "this", "that", "these", "those",
    "my", "me", "i", "you", "your", "we", "our", "us", "they", "them",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "do", "does", "did", "can", "could", "will", "would", "should", "shall", "may", "might",
    "have", "has", "had", "get", "got", "give", "gives", "make", "makes", "let", "tell",
    "if", "then", "than", "so", "but", "not", "no", "yes", "any", "all", "some", "each",
    "there", "here", "about", "into", "over", "under", "up", "out", "off", "per", "via",
    "deal", "deals", "offer", "offers", "discount", "discounts", "sale", "instant", "flat",
    "buy", "buying", "purchase", "order", "price", "prices", "pay", "paying", "cost",
    "cheap", "cheaper", "cheapest", "best", "better", "good", "great", "way", "ways",
    "card", "cards", "cashback", "reward", "rewards", "save", "saving", "savings",
    "want", "need", "find", "show", "looking", "recommend", "please", "thanks",
}

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

class DataIndex:
    _deals: List[Dict[str, Any]] = []
    _cards: List[Dict[str, Any]] = []
    _products: List[Dict[str, Any]] = []
    _loaded: bool = False
    _catalogue_tokens: Set[str] = set()
    _entity_tokens: Set[str] = set()

    @classmethod
    def load_all(cls, force: bool = False):
        if cls._loaded and not force:
            return

        deals_file = DATA_DIR / "deals.json"
        cards_file = DATA_DIR / "cards.json"
        products_file = DATA_DIR / "products.json"

        if deals_file.exists():
            with open(deals_file, "r", encoding="utf-8") as f:
                cls._deals = json.load(f)

        if cards_file.exists():
            with open(cards_file, "r", encoding="utf-8") as f:
                cls._cards = json.load(f)

        if products_file.exists():
            with open(products_file, "r", encoding="utf-8") as f:
                cls._products = json.load(f)

        cls._loaded = True
        cls._catalogue_tokens = set()
        cls._entity_tokens = set()

    # ------------------------------------------------------------------ vocabularies
    #
    # Two case-folded views of what the catalogue knows, built once and shared by every
    # component that has to decide whether a query refers to something we hold.
    #
    #   catalogue_tokens() — EVERY word in every record: names, titles, descriptions,
    #     terms, merchants, categories. Deliberately generous. A word in here is something
    #     the dataset talks about, so it can never be evidence that a query is out of scope.
    #
    #   entity_tokens()   — only words that IDENTIFY a record: product names and ids,
    #     merchants, card names and ids, categories, deal ids. Promotional filler from deal
    #     titles ("sale", "instant", "flat") is excluded, because a query containing it has
    #     not thereby named anything.
    #
    # Both are lower-cased, so every consumer is casing-independent by construction.

    @staticmethod
    def _tokens(text: Any) -> Set[str]:
        return set(re.findall(r"[a-z0-9]+", str(text or "").lower()))

    @classmethod
    def _walk_strings(cls, node: Any, acc: Set[str]) -> None:
        if isinstance(node, str):
            acc |= cls._tokens(node)
        elif isinstance(node, dict):
            for k, v in node.items():
                acc |= cls._tokens(k)
                cls._walk_strings(v, acc)
        elif isinstance(node, list):
            for v in node:
                cls._walk_strings(v, acc)

    @classmethod
    def catalogue_tokens(cls) -> Set[str]:
        """Every word appearing anywhere in the dataset, lower-cased."""
        cls.load_all()
        if not cls._catalogue_tokens:
            acc: Set[str] = set()
            for records in (cls._products, cls._deals, cls._cards):
                cls._walk_strings(records, acc)
            cls._catalogue_tokens = acc
        return cls._catalogue_tokens

    @classmethod
    def entity_tokens(cls) -> Set[str]:
        """
        Words that IDENTIFY one specific catalogue record, lower-cased.

        A token qualifies only if, within at least one kind of record, it belongs to exactly
        one of them. "kindle" names one product, "croma" one merchant, "groceries" one
        category — each identifies something we hold.

        A token shared by several records of its kind names a FAMILY, not a record: "hdfc"
        covers two cards, "iphone" two phones, "monthly" three products. Treating a family
        word as identification is what let "Does HDFC Infinia give 5x points?" look grounded
        — the issuer is ours, the card is not — so those are deliberately excluded here and
        left to the entity resolver, which asks or refuses depending on the evidence.
        """
        cls.load_all()
        if not cls._entity_tokens:
            kinds: List[Dict[str, Set[str]]] = []

            products: Dict[str, Set[str]] = {}
            for p in cls._products:
                key = p.get("product_id", "")
                bucket = products.setdefault(key, set())
                bucket |= cls._tokens(p.get("name"))
                bucket |= cls._tokens(p.get("product_id"))
            kinds.append(products)

            cards: Dict[str, Set[str]] = {}
            for c in cls._cards:
                key = c.get("card_id", "")
                bucket = cards.setdefault(key, set())
                bucket |= cls._tokens(c.get("name"))
                bucket |= cls._tokens(c.get("card_id"))
            kinds.append(cards)

            merchants: Dict[str, Set[str]] = {}
            for p in cls._products:
                for m in (p.get("prices") or {}):
                    merchants.setdefault(str(m).lower(), set()).update(cls._tokens(m))
            for d in cls._deals:
                m = d.get("merchant")
                if m:
                    merchants.setdefault(str(m).lower(), set()).update(cls._tokens(m))
            kinds.append(merchants)

            taxonomy: Dict[str, Set[str]] = {}
            for rec in list(cls._products) + list(cls._deals):
                for field in ("category", "subcategory"):
                    v = rec.get(field)
                    if v:
                        taxonomy.setdefault(str(v).lower(), set()).update(cls._tokens(v))
            kinds.append(taxonomy)

            ids: Dict[str, Set[str]] = {
                str(d.get("deal_id", "")).lower(): {str(d.get("deal_id", "")).lower()}
                for d in cls._deals
            }
            kinds.append(ids)

            acc: Set[str] = set()
            for kind in kinds:
                owners: Dict[str, int] = {}
                for record_tokens in kind.values():
                    for t in record_tokens:
                        owners[t] = owners.get(t, 0) + 1
                acc |= {t for t, n in owners.items() if n == 1}

            cls._entity_tokens = {t for t in acc if len(t) >= 3 and t not in GENERIC_WORDS}
        return cls._entity_tokens

    @classmethod
    def get_deals(cls) -> List[Dict[str, Any]]:
        cls.load_all()
        return cls._deals

    @classmethod
    def get_cards(cls) -> List[Dict[str, Any]]:
        cls.load_all()
        return cls._cards

    @classmethod
    def get_products(cls) -> List[Dict[str, Any]]:
        cls.load_all()
        return cls._products

    @classmethod
    def get_card_by_id(cls, card_id: str) -> Dict[str, Any] | None:
        cls.load_all()
        cid = card_id.lower()
        for c in cls._cards:
            if c["card_id"].lower() == cid:
                return c
        return None

    @classmethod
    def get_deal_by_id(cls, deal_id: str) -> Dict[str, Any] | None:
        cls.load_all()
        for d in cls._deals:
            if d["deal_id"].lower() == deal_id.lower():
                return d
        return None

    @classmethod
    def get_product_by_id(cls, product_id: str) -> Dict[str, Any] | None:
        cls.load_all()
        pid = product_id.lower()
        for p in cls._products:
            if p["product_id"].lower() == pid:
                return p
        return None
