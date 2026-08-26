from __future__ import annotations
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Set, Tuple

from app.rag.index import DataIndex
from app.core.llm_gate import LLMGate
from app.core.memory import MemoryManager
from app.core.thread import ConversationThread, TurnEntities, canonical_category

class Operation(str, Enum):
    LOOKUP = "LOOKUP"
    LIST = "LIST"
    AGGREGATE = "AGGREGATE"
    COMPUTE = "COMPUTE"
    OPTIMIZE = "OPTIMIZE"
    COMPARE = "COMPARE"
    ELIGIBILITY = "ELIGIBILITY"
    WATCH = "WATCH"
    STATE = "STATE"
    CLARIFY = "CLARIFY"
    EXPLAIN = "EXPLAIN"
    ABSTAIN = "ABSTAIN"

class Objective(str, Enum):
    """
    WHAT the user is optimising for. A separate semantic dimension from WHICH operation they
    are asking for: "best card for X" and "cheapest way to buy X" are both optimisations over
    the same candidate space, but they rank it by different quantities and admit different
    evidence.

    Previously this was a by-product of money classification — a query counted as a reward
    question only if it happened to contain one of five reward words. The same intent phrased
    two ways ("which card gives the most cashback on X" vs "best card for X") therefore
    produced two different answers: one compared cards, the other silently attached a merchant
    coupon. Objective is now resolved in its own right, and it drives both the ranking metric
    and whether merchant promotions may enter the calculation at all.
    """
    MIN_EFFECTIVE_PRICE = "min_effective_price"
    MAX_REWARD = "max_reward"
    MAX_DISCOUNT = "max_discount"


# Objective vocabularies. Pattern CLASSES over shopping language, not sentences: each covers a
# family of paraphrases, and none of them names a product, merchant, card, deal or amount.

# Minimising what you actually pay.
PRICE_OBJECTIVE_RE = re.compile(
    r"\bcheap(?:er|est|ly)?\b|\bbest\s+price\b|"
    r"\blowest\s+(?:\w+\s+)?(?:price|cost|total|effective|amount)\b|"
    r"\bpay\s+(?:the\s+)?least\b|\bleast\s+(?:expensive|amount)\b|\bminimi[sz]e\b|"
    r"\bpay\s+less\b|\bspend\s+(?:the\s+)?least\b|\bcosts?\s+(?:me\s+)?(?:the\s+)?least\b|"
    r"\b(?:save|saves|saving)\s+(?:me\s+)?(?:the\s+)?most\s+money\b"
)

# Maximising what the CARD pays back.
REWARD_OBJECTIVE_RE = re.compile(
    r"\bcashback\b|\bcash\s+back\b|\brewards?\b|\brewarding\b|\bpoints\b|\bearns?\b|\bearning\b|"
    r"\breward\s+rate\b|\bmoney\s+back\b|\b(?:best|biggest|highest|most)\s+return\b|"
    r"\b(?:best|which|what|right)\s+card\b|\bcard\s+(?:should|would|to\s+use|is\s+best|gives)\b|"
    r"\bcard\s+for\b|\bpayment\s+option\s+gives\b"
)

# Maximising the merchant discount itself.
DISCOUNT_OBJECTIVE_RE = re.compile(
    r"\bdiscounts?\b|\bsaves?\s+(?:me|you)\b|\bsavings\b|\bmost\s+off\b|"
    r"\b(?:biggest|largest|highest|most|best)\s+(?:discount|saving|offer)\b"
)

# Indifference: the user explicitly does NOT care about a dimension. An objective word
# inside such a clause is being waived, not requested — "I don't care which card I use" is a
# price question that happens to contain the words "which card". The clause is removed before
# the objective is read, the same way card names are removed before reward words are read.
INDIFFERENCE_RE = re.compile(
    r"\b(?:do(?:es)?n'?t|do not|does not|dont)\s+(?:care|mind|matter)\b[^.?!]*|"
    r"\b(?:no|any)\s+preference\b[^.?!]*|"
    r"\b(?:any|whichever|whatever|either)\s+(?:card|cards|merchant|store|deal|one|option)\b"
)


# Talk ABOUT offers, as opposed to a request to buy something.
DEAL_VOCABULARY_RE = re.compile(
    r"\b(?:deal|deals|offer|offers|discount|discounts|promotion|promotions|coupon|coupons)\b"
)


# A request for the single best of something, as opposed to a browse.
SUPERLATIVE_RE = re.compile(
    r"\b(?:best|biggest|largest|highest|most|top|greatest|cheapest|lowest)\b"
)


@dataclass
class ResolvedQuery:
    operation: Operation
    products: List[Dict[str, Any]] = field(default_factory=list)
    merchants: List[str] = field(default_factory=list)
    cards: List[Dict[str, Any]] = field(default_factory=list)
    deals: List[Dict[str, Any]] = field(default_factory=list)
    category: Optional[str] = None
    purchase_amount: Optional[float] = None
    reward_spend: Optional[float] = None
    budget: Optional[float] = None
    target_price: Optional[float] = None
    amount: Optional[float] = None
    target: Optional[float] = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    unresolved: List[str] = field(default_factory=list)
    # The CANONICAL conversation state this query was resolved against. Handlers receive
    # the thread itself rather than each re-deriving context from the session, so there is
    # exactly one authoritative representation of "what we are talking about"
    # (spec section 1, invariant A).
    thread: Optional[ConversationThread] = None
    # What the user is optimising for. Drives the ranking metric and whether merchant
    # promotions may enter the calculation.
    objective: str = Objective.MIN_EFFECTIVE_PRICE.value
    # Where the objective came from: "explicit" (this turn's words), "inherited" (explicit
    # earlier in the thread) or "default" (nobody ever said). Only "default" is ambiguous,
    # and a comparison must not silently pick a ranking metric when it is.
    objective_source: str = "default"
    # True only for a genuine reward question ("how much cashback on X?"), which is answered
    # from card rules alone rather than by stacking an unrequested merchant coupon.
    reward_question: bool = False
    comparison_axis: Optional[str] = None
    comparison_entities: List[str] = field(default_factory=list)
    classification_source: str = "deterministic"   # "structural" | "llm" | "deterministic"
    aggregate: Optional[str] = None
    field: Optional[str] = None
    clear_preference: bool = False
    clear_budget: bool = False
    is_followup: bool = False          # entities were inherited from session context, not stated this turn
    is_hypothetical: bool = False       # "what if I use X" — a one-turn override, not to be persisted as state
    inherited_product: bool = False
    inherited_card: bool = False
    clarification: Optional[str] = None   # set when a reference is genuinely ambiguous



# --- Linguistic intent markers -------------------------------------------------------
# These are PATTERN CLASSES (categories of phrasing), not literal sentences — they exist
# so the resolver can deterministically recognize the *kind* of follow-up a query is
# without depending on an LLM call (which has no visibility into conversation state
# anyway; see QueryResolver.resolve). Each one covers a family of paraphrases, not a
# fixed set of exact questions.

# "what if I use SBI", "suppose I spend 15000", "hypothetically, would HDFC be better" —
# a ONE-TURN override/scenario, never a persistent state change.
HYPOTHETICAL_RE = re.compile(
    r"\b(?:what if|what would happen if|what happens if|suppose|hypothetically|"
    r"would\s+\w+\s+be\s+better|would\s+it\s+be\s+better|"
    r"can\s+i\s+use|what\s+about\s+(?:using\s+)?|try\s+\w+\s+instead|"
    r"use\s+.+\s+instead|what\s+happens\s+with)\b"
)

# A follow-up asking for the OUTCOME of an already-established purchase context, with no
# new subject of its own: "how much do I save", "what's my final price", "how much
# cashback", "how much cheaper is that", "what do I actually pay".
OUTCOME_QUERY_RE = re.compile(
    r"\bhow\s+much\s+(?:do\s+i|would\s+i|will\s+i|can\s+i)?\s*"
    r"(?:save|pay|owe|earn|get\s+back)\b|"
    r"\bwhat'?s?\s+my\s+(?:final|effective|actual)\s+price\b|"
    r"\bwhat\s+do\s+i\s+(?:actually\s+)?pay\b|"
    r"\bhow\s+much\s+(?:cheaper|more\s+expensive|cashback)\b|"
    r"\bhow\s+much\s+(?:do\s+i\s+)?actually\s+save\b"
)

# Genuine PERSISTENT intent — as opposed to a hypothetical one-turn "what if". Only these
# should ever write to the persisted preferred_card/budget.
PERSISTENT_INTENT_RE = re.compile(
    r"\bi\s+prefer\b|\bfrom\s+now\s+on\b|\balways\s+use\b|\bby\s+default\b|"
    r"\bset\s+my\s+(?:preferred\s+)?card\b|\bmake\s+(?:that|this)\s+my\s+(?:card|preference)\b|"
    r"\buse\s+.+\s+(?:from\s+now\s+on|going\s+forward)\b"
)

# Negation / clearing verbs, generalized (not tied to "preference" appearing right next
# to the verb — "forget the SBI preference", "stop preferring HDFC", "don't use SBI
# anymore" must all match even with a card name in between).
NEGATION_VERB_RE = re.compile(
    r"\b(?:forget|clear|remove|stop|cancel|drop|un-?set)\b|"
    r"\bdon'?t\s+want\b|\bdo\s+not\s+want\b|\bno\s+longer\b|\bnot\s+use\b|"
    r"\bdon'?t\s+use\b|\bdo\s+not\s+use\b"
)
PREFERENCE_REFERENT_RE = re.compile(r"\bpreference\b|\bpreferred\b|\bprefer(?:ring)?\b|\bcard\b|\buse\b")
BUDGET_REFERENT_RE = re.compile(r"\bbudget\b")

# Vocabulary that marks a turn as being about SHOPPING/PAYMENT at all — acquiring
# something, what it costs, how to pay for it, or what a payment earns. Used as a
# confidence gate on the weakest entity matcher (see the distinctive-word pass in
# QueryResolver.resolve): a single word that happens to appear in one catalog title is only
# taken as a product reference when the turn is otherwise recognisably in-domain. Without
# that gate, "what's the weather in Mumbai?" resolved to the Delhi-Mumbai flight, because
# "mumbai" occurs in exactly one product title.
DOMAIN_INTENT_RE = re.compile(
    r"\b(?:buy|buying|bought|purchase|purchasing|order|ordering|book|booking|"
    r"shop|shopping|pay|paying|payment|price|prices|priced|cost|costs|"
    r"cheap|cheaper|cheapest|expensive|afford|"
    r"deal|deals|offer|offers|discount|discounts|coupon|coupons|promo|promotion|"
    r"cashback|reward|rewards|points|save|saving|savings|"
    r"spend|spending|worth|budget|card|cards|emi|"
    r"want|need|looking\s+for|recommend|best|compare|cart|checkout)\b"
)

# A question ABOUT a result already produced, rather than a request for a new one:
# a causal probe ("why", "how come"), a reference to the alternative that lost
# ("runner-up", "the other option", "why not X"), or a past-tense attribute lookup
# ("what WAS the discount", "how much cashback DID I get"). This is only ever consulted
# when a recommendation memo exists AND the turn names no new entity, so it selects
# between "explain the last answer" and "compute a new one" — it never invents either.
RETROSPECTIVE_RE = re.compile(
    r"\bwhy\b|\bhow\s+come\b|\breason\b|"
    r"\brunner[-\s]?up\b|\bnext\s+best\b|\bother\s+option\b|\bsecond\s+best\b|"
    r"\bwhat\s+was\b|\bwhat\s+were\b|\bdid\s+i\s+(?:get|earn|save|pay)\b|"
    r"\bhow\s+much\s+\w+\s+did\s+i\b"
)


class VocabularyIndex:
    _initialized = False
    _product_names: List[Tuple[str, Dict[str, Any]]] = []
    _merchants: Set[str] = set()
    _card_alias_map: Dict[str, str] = {}
    _deal_map: Dict[str, Dict[str, Any]] = {}
    _product_word_owners: Dict[str, Set[str]] = {}
    _ambiguous_product_words: Dict[str, Set[str]] = {}
    _ambiguous_card_words: Dict[str, Set[str]] = {}
    _category_keywords: Dict[str, str] = {
        "groceries": "groceries", "grocery": "groceries", "milk": "groceries", "supermarket": "groceries", "bigbasket": "groceries", "blinkit": "groceries", "zepto": "groceries", "instamart": "groceries",
        "electronics": "electronics", "electronic": "electronics", "laptop": "electronics", "phone": "electronics", "headphones": "electronics", "tv": "electronics", "croma": "electronics", "apple": "electronics", "sony": "electronics",
        "travel": "travel", "flight": "travel", "hotel": "travel", "resort": "travel", "cleartrip": "travel", "agoda": "travel", "makemytrip": "travel",
        "shopping": "shopping", "clothes": "shopping", "shoes": "shopping", "clothing": "shopping", "myntra": "shopping", "ajio": "shopping",
        "bills": "bills", "bill": "bills", "electricity": "bills", "utility": "bills", "recharge": "bills",
        "food": "food", "dining": "food", "restaurant": "food", "swiggy": "food", "zomato": "food", "meal": "food", "eatsure": "food"
    }

    @classmethod
    def initialize(cls):
        DataIndex.load_all(force=True)
        products = DataIndex.get_products()
        deals = DataIndex.get_deals()
        cards = DataIndex.get_cards()

        cls._product_names = [(p["name"].lower(), p) for p in products]

        # Word -> owning product_ids, used to resolve a single distinctive brand/model
        # word ("Kindle", "Paperwhite", "my Kindle") to its product even when fewer than
        # two tokens overlap — but only when that word is unique to one product (an
        # ambiguous shared word like "iPhone" across two models must NOT guess) AND is not
        # a generic category/descriptor word ("grocery", "monthly", "meal", ...) that just
        # happens to appear in only one product's title — that describes a category of
        # goods, not a specific product, and must not be treated as a product reference.
        product_generic_words = {"apple", "samsung", "nike", "levi's", "sony"}.union(
            cls._category_keywords.keys()
        )
        word_owners: Dict[str, Set[str]] = {}
        for p in products:
            for w in re.findall(r"[a-z0-9]+", p["name"].lower()):
                if len(w) >= 4 and w not in product_generic_words:
                    word_owners.setdefault(w, set()).add(p["product_id"])
        cls._product_word_owners = word_owners
        # A word that names several catalog products ("iphone" across two models) can never
        # identify one on its own. Kept so the resolver can ASK instead of guessing — the
        # rival readings have different prices, so a guess would change the arithmetic.
        # Symmetric to the card-family ambiguity handled below; both are derived from the
        # catalog, never from a hand-written confusables list.
        cls._ambiguous_product_words = {w: o for w, o in word_owners.items() if len(o) > 1}

        merchants = set()
        for d in deals:
            if d.get("merchant"):
                merchants.add(d["merchant"].lower())
        for p in products:
            for m in p.get("prices", {}).keys():
                merchants.add(m.lower())
        cls._merchants = merchants

        card_map = {}
        stop_card_parts = {"bank", "card", "credit", "pay", "payment", "cashback", "reward", "rewards", "smart", "earn", "prime", "the", "and"}.union(merchants)

        # A name fragment shared by multiple cards (e.g. "hdfc" on both HDFC Millennia and
        # HDFC Regalia) can never disambiguate on its own — registering it as an alias for
        # whichever card happens to be processed last would make ANY mention of one HDFC
        # card also silently match the other. Only register single-word aliases that
        # uniquely identify exactly one card.
        part_owners: Dict[str, Set[str]] = {}
        for c in cards:
            cid = c["card_id"].lower()
            for part in c["name"].lower().split():
                if len(part) > 2 and part not in stop_card_parts:
                    part_owners.setdefault(part, set()).add(cid)

        for c in cards:
            cid = c["card_id"].lower()
            cname = c["name"].lower()
            card_map[cid] = cid
            card_map[cname] = cid
            for part in cname.split():
                if len(part) > 2 and part not in stop_card_parts and len(part_owners.get(part, set())) == 1:
                    card_map[part] = cid
        cls._card_alias_map = card_map
        # Words that name a card family but not a single card ("hdfc" -> Millennia AND
        # Regalia). Kept so the resolver can ASK which one is meant instead of guessing —
        # picking either would silently change the money math.
        cls._ambiguous_card_words = {w: owners for w, owners in part_owners.items() if len(owners) > 1}

        deal_map = {}
        for d in deals:
            deal_map[d["deal_id"].lower()] = d
            deal_map[d["title"].lower()] = d
        cls._deal_map = deal_map

        cls._initialized = True

    @classmethod
    def get_merchants(cls) -> Set[str]:
        if not cls._initialized:
            cls.initialize()
        return cls._merchants

    @classmethod
    def get_card_alias_map(cls) -> Dict[str, str]:
        if not cls._initialized:
            cls.initialize()
        return cls._card_alias_map

    @classmethod
    def get_deal_map(cls) -> Dict[str, Dict[str, Any]]:
        if not cls._initialized:
            cls.initialize()
        return cls._deal_map

    @classmethod
    def get_product_names(cls) -> List[Tuple[str, Dict[str, Any]]]:
        if not cls._initialized:
            cls.initialize()
        return cls._product_names

    @classmethod
    def get_product_word_owners(cls) -> Dict[str, Set[str]]:
        if not cls._initialized:
            cls.initialize()
        return cls._product_word_owners

    @classmethod
    def get_ambiguous_card_words(cls) -> Dict[str, Set[str]]:
        if not cls._initialized:
            cls.initialize()
        return cls._ambiguous_card_words

    @classmethod
    def get_ambiguous_product_words(cls) -> Dict[str, Set[str]]:
        if not cls._initialized:
            cls.initialize()
        return cls._ambiguous_product_words


class QueryResolver:
    """
    Unified, data-driven resolver that maps queries to one of the 9 operations.
    """

    @classmethod
    def resolve(cls, query: str, session_id: Optional[str] = None, session_state: Optional[Dict[str, Any]] = None) -> ResolvedQuery:
        VocabularyIndex.initialize()
        q_raw = (query or "").strip()
        q_low = q_raw.lower()

        # 1. Match Products.
        # ALL products named this turn are collected, in the order they appear, not just the
        # first: "Kindle or MacBook?" is a product comparison and both sides have to reach
        # the candidate builder. Downstream code that wants a single subject still reads
        # `products[0]`, which stays the first-mentioned one.
        matched_products: List[Dict[str, Any]] = []
        _seen_pids: Set[str] = set()

        def _add_product(p_data: Dict[str, Any], position: int) -> None:
            pid = p_data.get("product_id")
            if pid and pid not in _seen_pids:
                _seen_pids.add(pid)
                matched_products.append((position, p_data))

        for p_name, p_data in VocabularyIndex.get_product_names():
            idx = q_low.find(p_name)
            if idx >= 0:
                _add_product(p_data, idx)
                continue
            # Check base name without specs
            clean_name = re.sub(r"\b(?:\d+gb|\d+-inch|noise cancelling headphones|smartphone|laptop)\b", "", p_name).strip()
            if len(clean_name) > 3:
                idx = q_low.find(clean_name)
                if idx >= 0:
                    _add_product(p_data, idx)

        matched_products = [p for _, p in sorted(matched_products, key=lambda t: t[0])]

        if not matched_products:
            # Token match with model numbers < 1000
            q_digits = {d for d in re.findall(r"\b\d+\b", q_low) if int(d) < 1000}
            stopwords = {"can", "get", "the", "for", "and", "deal", "deals", "best", "price", "prices", "card", "cards", "with", "from", "find", "order", "want", "need", "offer", "discount", "cheaper", "cheapest", "way", "buy", "purchase", "where", "how", "what", "much", "tell"}
            tokens = [t for t in re.findall(r"[a-z0-9]+", q_low) if t not in stopwords]
            
            # Every product clearing the threshold is kept, not only the top scorer:
            # "iPhone 15 vs iPhone 16" scores both identically and both sides are needed
            # for a product comparison. The model-number guard above still keeps each match
            # honest, so this widens the candidate set without loosening the match test.
            scored: List[Tuple[int, Dict[str, Any]]] = []
            for p_name, p_data in VocabularyIndex.get_product_names():
                p_digits = {d for d in re.findall(r"\b\d+\b", p_name) if int(d) < 1000}
                p_tokens = set(re.findall(r"[a-z0-9]+", p_name))
                if q_digits and p_digits and not q_digits.intersection(p_digits):
                    continue
                common_tokens = [t for t in tokens if t in p_tokens or (len(t) > 3 and t in p_name)]
                if len(common_tokens) >= 2:
                    score = len(common_tokens) * 2 + (5 if (q_digits and q_digits.intersection(p_digits)) else 0)
                    if score >= 4:
                        scored.append((score, p_data))
            if scored:
                top = max(sc for sc, _ in scored)
                for sc, p_data in scored:
                    if sc == top and p_data["product_id"] not in _seen_pids:
                        _seen_pids.add(p_data["product_id"])
                        matched_products.append(p_data)

        # Distinctive-word candidates: "my Kindle", "the Paperwhite" — a query may name a
        # product by one strong brand/model word instead of its full title. Only words that
        # uniquely identify ONE product catalog-wide qualify (an ambiguous shared word like
        # "iPhone" across two models must never guess). Several distinct products can be
        # named this way ("Kindle or MacBook?"), so all of them are collected.
        #
        # This is the WEAKEST matcher in the resolver — one word against one title — so the
        # candidates are held back here and only admitted below, once we can see whether the
        # turn is about shopping at all.
        word_owners = VocabularyIndex.get_product_word_owners()
        word_candidate_pids: List[str] = []
        for wm in re.finditer(r"[a-z0-9]+", q_low):
            owners = word_owners.get(wm.group(0))
            if owners and len(owners) == 1:
                pid = next(iter(owners))
                if pid not in word_candidate_pids and pid not in _seen_pids:
                    word_candidate_pids.append(pid)

        # 2. Match Cards.
        # Deliberately BEFORE merchants: card names can contain merchant names (a
        # co-branded card such as "ICICI Amazon Pay" literally contains "Amazon" and
        # "Amazon Pay"). Resolving cards first lets the merchant pass below discount any
        # merchant token that is really just part of a card name the user already named.
        matched_cards: List[Dict[str, Any]] = []
        cards_data = DataIndex.get_cards()
        card_alias_map = VocabularyIndex.get_card_alias_map()
        found_cids = set()
        for alias, cid in card_alias_map.items():
            if re.search(r"\b" + re.escape(alias) + r"\b", q_low):
                found_cids.add(cid)
        for c in cards_data:
            if c["card_id"].lower() in found_cids:
                matched_cards.append(c)

        # 3. Match Merchants.
        # A merchant token that lies inside an already-matched product or card name is
        # only taken as a real merchant when an explicit locative preposition marks it as
        # one ("at Amazon", "on Croma") — otherwise it belongs to that other entity.
        # Longest match wins and CLAIMS its span, so an overlapping shorter merchant name
        # cannot also match: "Amazon Pay" must not additionally register "Amazon" as a
        # second merchant — a spurious second entity on a dimension is what makes a plain
        # question look like a comparison.
        matched_merchants: List[str] = []
        _claimed: List[Tuple[int, int]] = []
        matched_entity_names = " ".join(
            [p["name"].lower() for p in matched_products] + [c["name"].lower() for c in matched_cards]
        )
        for m in sorted(VocabularyIndex.get_merchants(), key=len, reverse=True):
            for mt in re.finditer(r"\b" + re.escape(m) + r"\b", q_low):
                if m in matched_entity_names and not re.search(r"\b(?:at|on|from|in|store)\s+" + re.escape(m) + r"\b", q_low):
                    break
                a, b = mt.span()
                if any(a < ce and b > cs for cs, ce in _claimed):
                    continue
                _claimed.append((a, b))
                if m not in matched_merchants:
                    matched_merchants.append(m)
                break

        # 4. Match Deals
        matched_deals: List[Dict[str, Any]] = []
        deal_map = VocabularyIndex.get_deal_map()
        for k, d in deal_map.items():
            if k in q_low:
                if d not in matched_deals:
                    matched_deals.append(d)

        # 4a. Admit the held-back distinctive-word candidates, but only when the turn is
        # recognisably about shopping: it named another catalog entity (product, merchant,
        # card or deal), it carries shopping/payment vocabulary, or it continues an active
        # purchase thread. A lone title word in an otherwise out-of-domain sentence is a
        # coincidence, not a product reference — "what's the weather in Mumbai?" must not
        # become a Delhi-Mumbai flight recommendation just because one title contains that
        # city. Stronger matchers (full title, multi-token) are never gated this way.
        _sess_ctx = MemoryManager.get_session(session_id) if session_id else None
        _thread_has_context = bool(_sess_ctx and _sess_ctx.thread.has_context())
        _in_domain = bool(
            matched_products or matched_merchants or matched_cards or matched_deals
            or DOMAIN_INTENT_RE.search(q_low)
            or _thread_has_context
        )
        if _in_domain:
            for pid in word_candidate_pids:
                match = DataIndex.get_product_by_id(pid)
                if match and pid not in _seen_pids:
                    _seen_pids.add(pid)
                    matched_products.append(match)

        # 4b. Linguistic intent signals — computed BEFORE follow-up inheritance so the
        # inheritance rules below can respect negation/hypothetical framing. These are
        # pattern CLASSES (see module-level *_RE constants), not literal sentences.
        is_negation = bool(NEGATION_VERB_RE.search(q_low))
        clear_preference = is_negation and bool(PREFERENCE_REFERENT_RE.search(q_low) or matched_cards)
        clear_budget = is_negation and bool(BUDGET_REFERENT_RE.search(q_low))
        is_hypothetical = bool(HYPOTHETICAL_RE.search(q_low)) and not is_negation
        is_outcome_query = bool(OUTCOME_QUERY_RE.search(q_low))
        is_persistent_intent = bool(PERSISTENT_INTENT_RE.search(q_low))

        # 4c. Follow-up context inheritance. An ongoing purchase conversation ("what if I
        # use SBI?", "how much do I actually save?", "what would you recommend now?")
        # rarely re-states the product/card already established — it should inherit them
        # from the session's active purchase context rather than being treated as a
        # fresh, entity-less request. Anything explicitly named THIS turn always wins;
        # inheritance only fills in what the query itself left unstated, and never runs
        # on a clear/reset directive (that must not resurrect the thing being cleared).
        sess_for_anaphora = MemoryManager.get_session(session_id) if session_id else None
        inherited_product = False
        inherited_card = False

        # Literal demonstrative anaphora ("this product", "that deal", "that merchant").
        if not matched_products and sess_for_anaphora and sess_for_anaphora.product:
            if re.search(r"\b(?:this|that|the\s+same)\s+(?:product|item)\b", q_low):
                anaphora_prod = DataIndex.get_product_by_id(sess_for_anaphora.product)
                if anaphora_prod:
                    matched_products.append(anaphora_prod)
                    inherited_product = True

        if not matched_deals and sess_for_anaphora and sess_for_anaphora.last_deal_id:
            if re.search(r"\b(?:this|that|the\s+same)\s+deal\b", q_low):
                anaphora_deal = DataIndex.get_deal_by_id(sess_for_anaphora.last_deal_id)
                if anaphora_deal:
                    matched_deals.append(anaphora_deal)

        if not matched_merchants and sess_for_anaphora and sess_for_anaphora.last_merchant:
            if re.search(r"\b(?:this|that|the\s+same)\s+(?:merchant|store)\b", q_low):
                matched_merchants.append(sess_for_anaphora.last_merchant)

        # Whether the user named a card THIS turn (vs. it being inherited from context)
        # matters for operation routing later, so capture it before any inheritance runs.
        explicit_card_this_turn = bool(matched_cards)
        explicit_product_this_turn = bool(matched_products)
        explicit_merchant_this_turn = bool(matched_merchants)

        # NOTE: general dimension-wise thread inheritance happens in step 6b, after the
        # amount has been parsed — see ConversationThread.merge_turn(). It is deliberately
        # NOT keyed on any phrasing.
        is_followup = inherited_product or inherited_card

        # 5. Category
        category: Optional[str] = None
        if matched_products:
            category = matched_products[0].get("category")
        if not category:
            for kw, cat in VocabularyIndex._category_keywords.items():
                if re.search(r"\b" + re.escape(kw) + r"\b", q_low):
                    category = cat
                    break
        # A merchant whose catalogue deals all sit in ONE category tells us the category
        # without the user naming it: "What is TravelFly deal 3 discount?" is a travel
        # question. Derived from the dataset, never a hand-written merchant->category map,
        # and only used when nothing more specific was resolved — otherwise a stale session
        # category leaks in and filters the merchant's own deals out of retrieval.
        if not category and matched_merchants and not matched_products:
            _m_cats = {
                (d.get("category") or "").lower()
                for d in DataIndex.get_deals()
                if (d.get("merchant") or "").lower() == matched_merchants[0].lower()
            }
            _m_cats.discard("")
            if len(_m_cats) == 1:
                category = next(iter(_m_cats))

        if not category and session_state:
            category = session_state.get("category")
        if not category and session_id:
            sess = MemoryManager.get_session(session_id)
            if sess.category:
                category = sess.category

        # 6. Amounts & Financial Semantics
        amount, match_k = cls._parse_amount(q_raw, q_low)

        # Financial semantics classification:
        reward_spend: Optional[float] = None
        purchase_amount: Optional[float] = None
        budget: Optional[float] = None

        # `reward_question` marks a query that asks what a CARD pays on a spend, as opposed
        # to one that asks how to buy something. Only the former may suppress merchant
        # deals (spec section 19). It is deliberately NOT inferred from "no product
        # matched": "buy groceries worth RS 4,000" names no catalog product yet is plainly
        # a purchase, and treating it as a reward question hid every grocery deal from it.
        # OBJECTIVE. Resolved before money is classified, because what the user is
        # optimising for decides whether a figure is a spend to be rewarded or a purchase to
        # be discounted. An objective the turn does not state is inherited from the thread,
        # so a follow-up keeps optimising for the same thing.
        #
        # The active thread is needed first (it supplies an unstated objective), and the
        # objective is read from text with the matched card MENTIONS masked out — see
        # _mask_card_mentions for why that masking has to be span-based.
        thread = sess_for_anaphora.thread if sess_for_anaphora else ConversationThread()
        q_low_wo_cards = cls._mask_card_mentions(q_low, matched_cards)
        waived_dimensions = cls._waived_dimensions(q_low)
        objective, objective_source = cls._resolve_objective(q_low_wo_cards, thread)
        reward_question = objective == Objective.MAX_REWARD.value

        if any(w in q_low for w in ["budget", "want to spend", "planning to spend"]):
            budget = amount
        elif reward_question and not matched_products:
            reward_spend = amount
        elif matched_products:
            purchase_amount = amount
        else:
            reward_spend = amount

        # 6b. ACTIVE CONVERSATION THREAD — topic continuity + dimension-wise transition.
        #
        # This is the generalized replacement for phrase-driven follow-up handling. The
        # turn declares which dimensions it named; the thread supplies the rest. That one
        # rule makes "Amazon?", "SBI?", "15k?", "How much?" and "And Croma?" all work
        # without a single query-specific branch, because none of them is treated as a
        # phrase — each is just a turn that happens to name some dimensions and not others.

        category_word = None
        for _w in re.findall(r"[a-z]+", q_low):
            if canonical_category(_w):
                category_word = _w
                break

        if budget is not None:
            _basis = "budget"
        elif reward_spend is not None:
            _basis = "reward"
        else:
            _basis = "purchase"

        turn_ents = TurnEntities(
            product=matched_products[0] if matched_products else None,
            merchant=matched_merchants[0] if matched_merchants else None,
            card=matched_cards[0] if matched_cards else None,
            deal=matched_deals[0] if matched_deals else None,
            amount=amount,
            amount_basis=_basis,
            category_word=category_word,
            comparison_merchants=list(matched_merchants),
            comparison_cards=[c["card_id"] for c in matched_cards],
            comparison_deals=[d["deal_id"] for d in matched_deals],
            comparison_products=[p["product_id"] for p in matched_products],
        )

        continuity = thread.assess_continuity(turn_ents)
        persistent_card = sess_for_anaphora.preferred_card if sess_for_anaphora else None
        # A clear/forget directive must not resurrect the dimension being cleared, so the
        # card dimension is dropped from the thread before the merge inherits anything.
        if clear_preference:
            thread.clear_card_dimension()
        if clear_budget:
            thread.clear_budget_dimension()
        inherited_map = thread.merge_turn(turn_ents, continuity, persistent_card=persistent_card)
        # Only an EXPLICIT objective becomes sticky. Persisting an assumed one would make the
        # assumption look like intent on every later turn.
        if objective_source == "explicit":
            thread.active_objective = objective

        # Project the merged thread back onto this turn's resolved entities. Only
        # dimensions the turn did NOT state are filled in from context.
        if inherited_map["product"] and thread.active_product_id and not matched_products:
            _p = DataIndex.get_product_by_id(thread.active_product_id)
            if _p:
                matched_products.append(_p)
                inherited_product = True

        if (inherited_map["merchant"] and thread.active_merchant and not matched_merchants
                and "merchant" not in waived_dimensions):
            # Only inherit a merchant that can actually price the active product;
            # otherwise leave it open so the optimizer considers every valid merchant.
            _prod = matched_products[0] if matched_products else None
            _prices = {m.lower() for m in (_prod or {}).get("prices", {}).keys()} if _prod else set()
            if not _prod or thread.active_merchant.lower() in _prices:
                matched_merchants.append(thread.active_merchant)

        if (inherited_map["card"] and thread.active_card and not matched_cards
                and not clear_preference and "card" not in waived_dimensions):
            _c = DataIndex.get_card_by_id(thread.active_card)
            if _c:
                matched_cards.append(_c)
                inherited_card = True

        # An amount established earlier in the thread stays in force until the user
        # changes it or changes subject (a hypothetical is sticky WITHIN its thread).
        if inherited_map["amount"] and amount is None and thread.active_amount is not None:
            amount = thread.active_amount
            if thread.active_amount_basis == "reward":
                reward_spend = amount
            else:
                purchase_amount = amount

        # Re-classify money semantics now that context is known. The step-6 pass ran
        # before inheritance, so a bare "what if I spend 15k?" looked product-less and was
        # filed as reward_spend; against an active PRODUCT thread it is really a purchase
        # amount for that product. A budget statement is never reinterpreted this way.
        if matched_products and amount is not None and budget is None:
            if not any(w in q_low for w in ["reward", "cashback", "points", "rate", "earn"]):
                purchase_amount = amount
                reward_spend = None

        # Category follows the (possibly inherited) product.
        if matched_products and not category_word:
            category = matched_products[0].get("category") or category
        elif thread.active_category and not category:
            category = thread.active_category

        is_followup = inherited_product or inherited_card or bool(inherited_map["amount"] and amount is not None)

        # 6c. AMBIGUOUS REFERENCE DETECTION.
        # A reference is only worth asking about when context cannot settle it AND the
        # rival readings would produce different money. Data-driven: the candidates come
        # from the catalog, never from a hand-written list of confusable names.
        clarification: Optional[str] = None
        ungrounded_card_ref: Optional[str] = None
        _q_words = set(re.findall(r"[a-z]+", q_low))

        # Tested against what the turn EXPLICITLY named, not the post-inheritance set:
        # an inherited card must not mask an ambiguous new reference the user just made.
        # Vocabulary of every word that appears in a card the catalogue actually holds.
        # Used to tell "which HDFC card do you mean?" (a real ambiguity) apart from
        # "HDFC Infinia" (a card we simply do not have, where offering a choice between two
        # other HDFC cards answers a question nobody asked).
        _card_vocab: Set[str] = set()
        for _c in cards_data:
            _card_vocab.update(re.findall(r"[a-z0-9]+", _c["name"].lower()))
            _card_vocab.update(re.findall(r"[a-z0-9]+", _c["card_id"].lower()))

        if not explicit_card_this_turn and not clear_preference:
            for _w in _q_words:
                _owners = VocabularyIndex.get_ambiguous_card_words().get(_w)
                if _owners and len(_owners) > 1:
                    # Does the turn name a specific card of that family that we don't stock?
                    _after = re.search(
                        r"\b" + re.escape(_w) + r"\s+([A-Za-z][A-Za-z0-9]{2,})", q_low
                    )
                    if _after and _after.group(1).lower() not in _card_vocab:
                        ungrounded_card_ref = f"{_w} {_after.group(1)}"
                        break
                    # Names a card FAMILY but not a specific card (e.g. two cards from the
                    # same issuer). Their rates/caps differ, so guessing would change the
                    # answer's arithmetic — ask instead.
                    _names = sorted(
                        (DataIndex.get_card_by_id(cid) or {}).get("name", cid) for cid in _owners
                    )
                    clarification = (
                        f"Which card do you mean — {' or '.join(_names)}? "
                        f"They have different reward rates, so the numbers would differ."
                    )
                    break

        # A word that names a product FAMILY but not one specific product (two catalog
        # models sharing a name fragment). Their prices differ, so guessing would change the
        # arithmetic — ask instead. Skipped when a product we already matched contains that
        # word, since the fuller phrase has settled it.
        if clarification is None and not matched_products:
            for _w in _q_words:
                _p_owners = VocabularyIndex.get_ambiguous_product_words().get(_w)
                if not _p_owners or len(_p_owners) <= 1:
                    continue
                _all_rivals = [DataIndex.get_product_by_id(pid) for pid in _p_owners]
                _all_rivals = [r for r in _all_rivals if r]
                if len(_all_rivals) < 2:
                    continue
                # Only a word that names one PRODUCT FAMILY is a genuine ambiguity. A word
                # whose owners span DIFFERENT categories is an ordinary descriptor that
                # unrelated titles happen to share ("monthly" across a grocery basket, a milk
                # pass and an electricity bill) — a modifier, not a product reference, and
                # asking about it derails a perfectly answerable category question. Tested on
                # the full owner set, before any narrowing, so scoping cannot manufacture a
                # single-category family out of a descriptor.
                if len({(r.get("category") or "").lower() for r in _all_rivals}) != 1:
                    continue
                # A known category then narrows which rivals we name back to the user.
                _rivals = _all_rivals
                if category:
                    _scoped = [r for r in _rivals if (r.get("category") or "").lower() == category.lower()]
                    if _scoped:
                        _rivals = _scoped
                if len(_rivals) < 2:
                    continue
                _p_names = sorted(r.get("name", "") for r in _rivals)
                clarification = (
                    f"Which one do you mean — {' or '.join(_p_names)}? "
                    f"They are priced differently, so the numbers would differ."
                )
                break

        # Cross-type (merchant vs card) ambiguity only genuinely arises from ELLIPSIS:
        # a bare "Amazon?" has no syntax to fix the role, whereas "What deals does Amazon
        # have?" plainly uses it as a merchant. So this check applies only to elliptical
        # turns. (Within-type ambiguity above — two cards from one issuer — is different:
        # no amount of syntax can pick between them, so it always asks.)
        # Ellipsis is measured by raw length: a real sentence carries the syntax that
        # fixes the entity's role, an elliptical fragment does not. Counting *content*
        # words would be wrong here — words like "deals"/"have" are precisely the syntax
        # that disambiguates, so discounting them would make a full question look bare.
        _is_elliptical = len(re.findall(r"[a-z0-9]+", q_low)) <= 3

        if clarification is None and _is_elliptical and matched_merchants and not matched_cards:
            # A token that names both a merchant and a card (e.g. a co-branded card).
            # Context resolves it safely when the merchant actually prices the active
            # product; otherwise ask rather than silently picking a reading.
            _cards_all = DataIndex.get_cards()
            for _m in matched_merchants:
                _rival = [c for c in _cards_all if _m in c["name"].lower()]
                if not _rival:
                    continue
                _prod = matched_products[0] if matched_products else None
                _prices = {k.lower() for k in (_prod or {}).get("prices", {}).keys()} if _prod else set()
                _merchant_reading_supported = bool(_prod and _m in _prices)
                _explicit_role = re.search(
                    r"\b(?:at|on|from|in|store)\s+" + re.escape(_m) + r"\b|"
                    r"\b(?:with|using|pay\s+(?:by|with))\s+" + re.escape(_m) + r"\b", q_low
                )
                if not _merchant_reading_supported and not _explicit_role:
                    clarification = (
                        f"Do you mean {_m.title()} as the merchant, "
                        f"or the {_rival[0]['name']} card?"
                    )
                    break

        # 7. Target Price (for price alerts)
        target: Optional[float] = None
        # Word-bounded: without \b, "at" matches inside "th-AT-", so "make that ₹3,300"
        # was silently parsed as a price-watch target and routed to WATCH.
        match_target = re.search(
        r"\b(?:below|at|target|drops?\s*to|under|hits|cheaper\s+than|less\s+than|"
        r"lower\s+than|drops?\s+below)\s*(?:RS\s*|₹\s*)?(\d{1,3}(?:,\d{3})+|\d{3,6})(?:\.\d+)?",
        q_raw, re.IGNORECASE)
        if match_target:
            target = float(match_target.group(1).replace(",", ""))
        elif match_k and any(w in q_low for w in ["hits", "drops", "alert", "ping", "watch", "track"]):
            target = float(match_k.group(1)) * 1000.0

        # 8. Constraints (Session memory & in-query constraints)
        constraints: Dict[str, Any] = {}
        if session_state:
            constraints.update(session_state)
        if session_id:
            sess = MemoryManager.get_session(session_id)
            if sess.budget is not None and "budget" not in constraints:
                constraints["budget"] = sess.budget
            if sess.preferred_card and "preferred_card" not in constraints:
                constraints["preferred_card"] = sess.preferred_card
            if sess.category_preferences:
                constraints.setdefault("category_preferences", {}).update(sess.category_preferences)

        # A category-scoped preference ("I prefer SBI for electronics") is more specific
        # than the global one, so it wins for its own category. It was being stored and
        # never consulted; a card named THIS turn still overrides it below.
        _cat_prefs = constraints.get("category_preferences") or {}
        if category and _cat_prefs.get(category):
            constraints["preferred_card"] = _cat_prefs[category]

        # A constraint pins the candidate space to ONE entity, so it may only be written
        # when the turn really named one. Pinning to `[0]` while the user named two was the
        # card-axis twin of the merchant-comparison bug: the second alternative was resolved
        # and then silently excluded before any money was computed.
        if len(matched_merchants) == 1:
            constraints["named_merchant"] = matched_merchants[0]
        elif len(matched_merchants) >= 2:
            constraints.pop("named_merchant", None)

        if len(matched_cards) == 1:
            constraints["preferred_card"] = matched_cards[0]["card_id"]
        elif len(matched_cards) >= 2:
            constraints.pop("preferred_card", None)

        if matched_deals:
            constraints["named_deal_ids"] = [d["deal_id"] for d in matched_deals]

        # Parse Stated Cap Usage & Headroom
        spend_to_date = dict(constraints.get("spend_to_date") or {})
        c_id = matched_cards[0]["card_id"] if matched_cards else (constraints.get("preferred_card") or "hdfc_millennia")
        c_obj = next((c for c in cards_data if c["card_id"] == c_id), None)

        cap_used_match = re.search(r"(?:used\s*(?:RS\s*|₹\s*)?(\d{2,5})|(\d{2,5})\s*cap\s*used)", q_raw, re.IGNORECASE)
        if cap_used_match:
            c_amt = float(cap_used_match.group(1) or cap_used_match.group(2))
            spend_to_date[f"{c_id}_{category or 'general'}"] = c_amt

        headroom_match = re.search(r"(?:(\d{2,5})\s*headroom|headroom\s*(?:of\s*)?(?:RS\s*|₹\s*)?(\d{2,5}))", q_raw, re.IGNORECASE)
        if headroom_match:
            h_amt = float(headroom_match.group(1) or headroom_match.group(2))
            real_cap = 500.0
            is_monthly = "monthly" in q_low
            if c_obj:
                caps_dict = c_obj.get("caps", {})
                if is_monthly:
                    real_cap = float(caps_dict.get("monthly_cashback_cap", 500.0))
                else:
                    cat_caps = caps_dict.get("category_caps", {})
                    real_cap = float(cat_caps.get(category or "general", caps_dict.get("monthly_cashback_cap", 500.0)))
            
            prior_spend = max(0.0, real_cap - h_amt)
            if is_monthly:
                spend_to_date[f"{c_id}_monthly"] = prior_spend
            else:
                spend_to_date[f"{c_id}_{category or 'general'}"] = prior_spend
        constraints["spend_to_date"] = spend_to_date

        # An open comparison acts as a CONSTRAINT on the candidate space rather than as a
        # separate operation: whatever the user asks next ("which is cheapest?", "now use
        # SBI") is answered over the alternatives already under discussion. This keeps the
        # comparison alive across follow-ups without needing to detect ranking phrasing.
        if thread.comparison_entities and thread.comparison_axis:
            constraints["comparison_axis"] = thread.comparison_axis
            constraints["comparison_entities"] = list(thread.comparison_entities)

        # A dimension the user has just waived carries no constraint this turn. The standing
        # preference itself is untouched — it applies again on the next turn that does not
        # waive it, and only an explicit clear directive removes it for good.
        if "card" in waived_dimensions:
            constraints.pop("preferred_card", None)
        if "merchant" in waived_dimensions:
            constraints.pop("named_merchant", None)

        # clear_preference / clear_budget were already computed in step 4b (negation
        # detection needs to happen before follow-up inheritance, not after it).
        if clear_preference:
            constraints["clear_preference"] = True
        if clear_budget:
            constraints["clear_budget"] = True

        # 9. Unresolved Entities Detection
        unresolved: List[str] = []
        # A specific card of a known issuer that the catalogue does not hold ("HDFC Infinia")
        # is an ungrounded entity, not an ambiguity: we cannot answer it, and offering a
        # choice between two other HDFC cards would answer a question nobody asked.
        if ungrounded_card_ref:
            unresolved.append(ungrounded_card_ref)
        if not matched_products and not matched_deals and not matched_cards and not (amount and category):
            act_match = re.search(r"\b(?:buy|purchase|deals?\s+on|deal\s+for|price\s+of|how\s+much\s+for|compare)\s+([A-Za-z0-9\s\-]+)", q_low)
            if act_match:
                candidate_entity = act_match.group(1).strip(" .?!,")
                ignore_tokens = {"groceries", "electronics", "travel", "shopping", "bills", "food", "cheapest", "way", "the", "a", "an", "best", "deal", "deals"}
                cand_words = [w.strip(" .?!,") for w in candidate_entity.split() if w.strip(" .?!,") not in ignore_tokens and len(w.strip(" .?!,")) > 2]
                if cand_words and not any(w in VocabularyIndex.get_merchants() for w in cand_words):
                    unresolved.append(candidate_entity)

        # 10. Operation Classification. The LLM classifier only ever sees the raw query
        # text and has no visibility into conversation state, so it cannot by itself
        # recognize "how much do I actually save?" as a continuation of an existing
        # purchase computation — pass it a short, factual context hint (never inventing
        # content, just stating what's already resolved) to help it generalize.
        context_hint = None
        if is_followup:
            hint_parts = []
            if matched_products:
                hint_parts.append(f"an active product in context: {matched_products[0]['name']}")
            if matched_cards:
                hint_parts.append(f"an active card in context: {matched_cards[0]['card_id']}")
            if hint_parts:
                context_hint = "Conversation context (already resolved, from a previous turn): " + "; ".join(hint_parts) + "."

        # STRUCTURAL CLASSIFICATION FIRST (spec section 23). Some cases are decidable from
        # the structured state alone — an ambiguous reference, a clear directive, two peers
        # named on one dimension, a retrospective question against a stored memo, an amount
        # with nothing to spend it on. Those must not depend on an LLM guessing from bare
        # query text, and when one applies we skip the LLM call entirely: it cannot improve
        # a decision the state already settles, and skipping it keeps the system working
        # identically when the provider is unavailable.
        structural_op, structural_clarification = cls._structural_operation(
            q_low=q_low,
            thread=thread,
            products=matched_products,
            merchants=matched_merchants,
            cards=matched_cards,
            deals=matched_deals,
            category=category,
            amount=amount,
            purchase_amount=purchase_amount,
            reward_spend=reward_spend,
            objective=objective,
            reward_question=reward_question,
            budget=budget,
            unresolved=unresolved,
            clarification=clarification,
            clear_preference=clear_preference,
            clear_budget=clear_budget,
            explicit_product=explicit_product_this_turn,
            explicit_merchant=explicit_merchant_this_turn,
            explicit_card=explicit_card_this_turn,
            stated_amount=turn_ents.amount,
        )
        if structural_clarification and not clarification:
            clarification = structural_clarification

        classification_source = "structural"
        if structural_op is not None:
            operation = structural_op
        else:
            operation = cls._classify_operation(
                query=q_raw,
                products=matched_products,
                merchants=matched_merchants,
                cards=matched_cards,
                deals=matched_deals,
                category=category,
                purchase_amount=purchase_amount,
                reward_spend=reward_spend,
                budget=budget,
                target_price=target,
                amount=amount,
                target=target,
                unresolved=unresolved,
                context_hint=context_hint
            )
            classification_source = "llm" if LLMGate.is_available() else "deterministic"

        # Remaining refinements. These only ADJUST a classifier guess using linguistic
        # framing plus what we actually resolved; the structural decisions above already
        # won outright and are never revisited here.
        if structural_op is not None:
            pass
        elif is_hypothetical and not is_persistent_intent and not matched_deals:
            # Phrasing like "use X instead" is genuinely ambiguous on its own — whether
            # it's a one-turn scenario or a persistent preference change depends on
            # whether there's actually anything to compute against: an active purchase
            # context, a product named this turn, or a spend amount named this turn.
            # (Excluded when a specific deal is named: "can I use X with deal_001?" is an
            # ELIGIBILITY question, not a hypothetical purchase override — HYPOTHETICAL_RE's
            # "can i use" trigger is about card substitution in a purchase, not deal rules.)
            has_computable_signal = is_followup or bool(matched_products) or purchase_amount is not None or reward_spend is not None
            if has_computable_signal:
                # Something to recompute — "what if"/"use X instead" applies to THIS turn
                # only, never a budget/preference update.
                if operation in (Operation.STATE, Operation.ABSTAIN):
                    operation = Operation.COMPUTE
            elif explicit_card_this_turn:
                # Nothing established or stated to apply a hypothetical to — "use X
                # instead" is the user setting their card preference going forward.
                operation = Operation.STATE
        elif is_outcome_query and is_followup:
            # "how much do I actually save?" against an already-established purchase
            # context must resolve as a calculation, not a fresh lookup/listing/abstain.
            if operation in (Operation.ABSTAIN, Operation.LOOKUP, Operation.LIST, Operation.STATE):
                operation = Operation.COMPUTE
        elif is_followup and operation == Operation.ABSTAIN:
            # We successfully inherited context (product and/or card) for this turn —
            # abstaining would throw away a follow-up we can actually answer.
            operation = Operation.COMPUTE if explicit_card_this_turn else Operation.OPTIMIZE

        aggregate: Optional[str] = None
        if operation == Operation.AGGREGATE:
            # Word-boundary matched, not substring matched. "discount" CONTAINS "count", so
            # "which deal gives the biggest discount?" was being aggregated as a COUNT and
            # answered with a nonsense figure. Counting is tested first and anchored, so a
            # word that merely contains a keyword can never trigger it.
            if re.search(r"\b(?:how many|count|number of)\b", q_low):
                aggregate = "count"
            elif re.search(r"\b(?:largest|biggest|highest|most|max|maximum|best|top|greatest)\b", q_low):
                aggregate = "max"
            elif re.search(r"\b(?:lowest|smallest|cheapest|least|min|minimum|fewest)\b", q_low):
                aggregate = "min"
            else:
                aggregate = "max"

        # Same anchoring rule: "cap" sits inside "capped"/"capacity", "rate" inside
        # "accurate", and an unanchored match picks the wrong field.
        field_name: Optional[str] = None
        if re.search(r"\b(?:rate|rates|cashback)\b", q_low):
            field_name = "reward_rate"
        elif re.search(r"\bcaps?\b", q_low):
            field_name = "cap"
        elif re.search(r"\bdiscounts?\b", q_low):
            field_name = "discount"
        elif re.search(r"\b(?:min|minimum)\s+spend\b", q_low):
            field_name = "min_spend"

        if session_id:
            MemoryManager.update_last_entities(
                session_id,
                product_id=matched_products[0]["product_id"] if matched_products else None,
                deal_id=matched_deals[0]["deal_id"] if matched_deals else None,
                merchant=matched_merchants[0] if matched_merchants else None,
            )

            # The ConversationThread (merged in step 6b) is now the single source of truth
            # for in-thread working state. Mirror it onto the flat SessionMemory fields,
            # which remain the back-compatible view used elsewhere. A clear directive is
            # NOT overwritten here — handle_state nulls those fields authoritatively.
            if not clear_preference:
                MemoryManager.set_active_card(session_id, thread.active_card)
            if not clear_budget:
                MemoryManager.set_active_amount(
                    session_id, thread.active_amount, thread.active_amount_basis
                )

        return ResolvedQuery(
            operation=operation,
            products=matched_products,
            merchants=matched_merchants,
            cards=matched_cards,
            deals=matched_deals,
            category=category,
            purchase_amount=purchase_amount,
            reward_spend=reward_spend,
            budget=budget,
            target_price=target,
            amount=amount,
            target=target,
            constraints=constraints,
            unresolved=unresolved,
            aggregate=aggregate,
            field=field_name,
            clear_preference=clear_preference,
            clear_budget=clear_budget,
            is_followup=is_followup,
            is_hypothetical=is_hypothetical,
            inherited_product=inherited_product,
            inherited_card=inherited_card,
            clarification=clarification,
            thread=thread,
            objective=objective,
            objective_source=objective_source,
            reward_question=reward_question,
            comparison_axis=thread.comparison_axis,
            comparison_entities=list(thread.comparison_entities),
            classification_source=classification_source,
        )

    @classmethod
    def _mask_card_mentions(cls, q_low: str, matched_cards: List[Dict[str, Any]]) -> str:
        """
        Blank out where a matched card is NAMED, so brand words are not mistaken for the
        user's own vocabulary.

        Several cards are literally called "... Cashback ...", so a card's brand would
        otherwise make any sentence mentioning it look like a cashback question. The previous
        masking removed every occurrence of every word in the card's name — which deleted the
        user's OWN word too. In

            "What cashback would I get on a RS 10,000 grocery purchase with SBI Cashback?"

        it deleted both occurrences of "cashback", leaving no reward vocabulary at all, so the
        turn resolved as a price question. The candidate builder then correctly honoured that
        frame and admitted a merchant coupon the user never asked for: the frame was already
        wrong before any downstream stage saw it.

        Masking is therefore SPAN-based. For each matched card, remove the single longest
        contiguous mention of that card and nothing else. Multi-word spans are tried longest
        first; a lone word is removed only when it uniquely identifies that card by itself, so
        a generic brand word such as "cashback" or "card" is never removed on its own.
        """
        alias_map = VocabularyIndex.get_card_alias_map()
        text = q_low
        for card in matched_cards:
            card_id = (card.get("card_id") or "").lower()
            tokens = re.findall(r"[a-z0-9]+", (card.get("name") or "").lower())

            phrases: List[str] = [card_id] if card_id else []
            for size in range(len(tokens), 1, -1):
                for start in range(0, len(tokens) - size + 1):
                    phrases.append(" ".join(tokens[start:start + size]))
            phrases += [t for t in tokens if alias_map.get(t) == card_id]

            for phrase in phrases:
                parts = phrase.split()
                if not parts:
                    continue
                pattern = r"\b" + r"\s+".join(re.escape(t) for t in parts) + r"\b"
                masked, hits = re.subn(pattern, " ", text, count=1)
                if hits:
                    text = masked
                    break
        return text

    # ------------------------------------------------------------------ money parsing
    @classmethod
    def _parse_amount(cls, q_raw: str, q_low: str):
        """
        Find the transaction figure in a turn, whatever shape the sentence gives it.

        The previous parser only recognised a number when a currency symbol or one of a
        handful of verbs sat immediately before it, so the SAME amount was read in some
        phrasings and silently dropped in others: "spend 8000" parsed, while "an 8,000
        grocery purchase", "8,000 worth of groceries", "if I pay 8,000" and "on an 8,000
        grocery bill" all resolved to no amount at all — and a missing amount quietly
        degraded the answer into a rate card instead of a calculation.

        A number is now taken as money when its FORM says so, independent of phrasing:

          * a currency marker precedes it, or
          * it is written with thousands separators, or
          * it is bare and at least four digits, or
          * it is written with a "k" suffix.

        Two data-driven guards keep specifications out. A "k" figure is a specification, not
        money, when the catalogue itself uses that token — "4k" appears in a television's
        name, so "4K display" is not RS 4,000, while "8k" appears nowhere and means RS 8,000.
        And a bare number under four digits is never money on its own, which is what keeps
        model numbers and pack sizes ("iPhone 15", "Box for 4", "2 Nights") out.

        Returns (amount, k_match) — the second value is kept because the price-watch parser
        downstream still uses it.
        """
        catalogue = DataIndex.catalogue_tokens()
        candidates = []   # (position, value)

        # (a) currency-marked or verb-marked.
        pattern_num = r"\d{1,3}(?:,\d{3})+|\d{4,}"
        for m in re.finditer(
            r"(?:₹|\$|RS\s*|Rs\.?\s*)(" + pattern_num + r"|\d{3,})", q_raw, re.IGNORECASE
        ):
            candidates.append((m.start(), float(m.group(1).replace(",", ""))))

        for m in re.finditer(
            r"(?:worth|spending|spend|budget|pay|paying|paid|costs?|of|for|on|about|around|"
            r"upto|up\s+to|under|below)\s*(?:is|of|a|an|the)?\s*(" + pattern_num + r"|\d{3,})",
            q_raw, re.IGNORECASE,
        ):
            candidates.append((m.start(1), float(m.group(1).replace(",", ""))))

        # (b) any number whose own FORM is monetary: grouped, or four digits or more.
        for m in re.finditer(r"\b(" + pattern_num + r")\b", q_raw):
            candidates.append((m.start(1), float(m.group(1).replace(",", ""))))

        # (c) "k" shorthand, unless the catalogue uses that token as a specification.
        match_k = None
        for m in re.finditer(r"\b(\d+(?:\.\d+)?)\s*k\b", q_low):
            token = m.group(0).replace(" ", "")
            if token in catalogue:
                continue
            match_k = match_k or m
            candidates.append((m.start(1), float(m.group(1)) * 1000.0))

        if not candidates:
            return None, None
        candidates.sort(key=lambda c: c[0])
        return candidates[0][1], match_k

    # ------------------------------------------------------------------ objective
    @classmethod
    def _waived_dimensions(cls, q_low: str) -> Set[str]:
        """
        Which dimensions the user has explicitly said they do not care about.

        Indifference was already recognised well enough to stop "I don't care which card I
        use" reading as a reward question. But recognising it there and doing nothing with it
        left the constraint in place: a card the user had preferred EARLIER still pinned the
        answer on a turn where they had just said any card would do. Saying a dimension does
        not matter releases the constraint on it for this turn, while leaving the standing
        preference intact — clearing that permanently needs an explicit "forget".
        """
        waived: Set[str] = set()
        for match in INDIFFERENCE_RE.finditer(q_low):
            span = match.group(0)
            if re.search(r"\bcards?\b", span):
                waived.add("card")
            if re.search(r"\b(?:merchants?|stores?|sellers?)\b", span):
                waived.add("merchant")
            if re.search(r"\bdeals?\b", span):
                waived.add("deal")
        return waived

    @classmethod
    def _resolve_objective(cls, q_low: str, thread: ConversationThread) -> str:
        """
        What is the user optimising for?

        Precedence is deliberate. An explicit request to pay less settles it, even alongside
        card talk — "cheapest way to buy X on the best card" is a price question in which the
        card is one of the variables. Otherwise a request about the discount itself, then a
        request about what the card pays back. A turn that states no objective inherits the
        one already in play, so a follow-up keeps optimising for the same thing.

        Returns (objective, source). The SOURCE matters: an objective nobody ever stated is an
        assumption, and an assumption must not quietly decide a comparison whose answer
        changes with the metric.
        """
        q_low = INDIFFERENCE_RE.sub(" ", q_low)
        if PRICE_OBJECTIVE_RE.search(q_low):
            return Objective.MIN_EFFECTIVE_PRICE.value, "explicit"
        if DISCOUNT_OBJECTIVE_RE.search(q_low):
            return Objective.MAX_DISCOUNT.value, "explicit"
        if REWARD_OBJECTIVE_RE.search(q_low):
            return Objective.MAX_REWARD.value, "explicit"
        inherited = getattr(thread, "active_objective", None)
        if inherited:
            return inherited, "inherited"
        return Objective.MIN_EFFECTIVE_PRICE.value, "default"

    # ------------------------------------------------------------------ structural rules
    @classmethod
    def _structural_operation(
        cls,
        *,
        q_low: str,
        thread: ConversationThread,
        products: List[Dict[str, Any]],
        merchants: List[str],
        cards: List[Dict[str, Any]],
        deals: List[Dict[str, Any]],
        category: Optional[str],
        amount: Optional[float],
        purchase_amount: Optional[float],
        reward_spend: Optional[float],
        reward_question: bool,
        budget: Optional[float],
        unresolved: List[str],
        objective: str,
        clarification: Optional[str],
        clear_preference: bool,
        clear_budget: bool,
        explicit_product: bool,
        explicit_merchant: bool,
        explicit_card: bool,
        stated_amount: Optional[float],
    ) -> Tuple[Optional[Operation], Optional[str]]:
        """
        Decide the operation from STRUCTURED STATE where the state is conclusive.

        Returns (operation, clarification) or (None, None) to defer to the LLM/keyword
        classifier. Nothing here inspects the query for a particular sentence — each rule
        is a statement about the shape of the resolved state (how many entities on which
        dimension, whether a memo exists, whether a purchase can be identified at all), so
        it generalizes to any wording over any dataset.
        """
        # Never guess when the rival readings would change the money. Checked BEFORE the
        # abstention rule: an ambiguous reference means the entity IS in the catalog and we
        # cannot tell which record is meant — that is a question to ask, not an unknown
        # entity to refuse ("deals on iPhone?" when the catalog holds two iPhones).
        if clarification:
            return Operation.CLARIFY, None

        # An entity the catalog does not contain cannot be priced or reasoned about.
        if unresolved and not (products or deals or cards):
            return Operation.ABSTAIN, None

        # An explicit clear/forget directive is always a session mutation.
        if clear_preference or clear_budget:
            return Operation.STATE, None

        # A question ABOUT an answer already produced, introducing no new subject of its
        # own, is answered from the stored memo — never by a fresh retrieval that could
        # surface different candidates and therefore different numbers (spec section 38).
        has_memo = bool(thread.last_recommendation or thread.last_comparison)
        retrospective = bool(RETROSPECTIVE_RE.search(q_low))
        # An outcome question about the answer just given ("how much do I save?") is the same
        # kind of turn as "why?" — it asks about a result already produced, not for a new one.
        outcome_about_memo = bool(OUTCOME_QUERY_RE.search(q_low))
        if has_memo and (retrospective or outcome_about_memo):
            # A figure that MATCHES one already quoted in the memo is a reference back to it,
            # not a new spend. "Why didn't you use the 1,500-off deal?" names a rejected
            # deal's discount; treating it as a purchase amount priced a RS 1,500 basket and
            # answered a question nobody asked.
            amount_is_reference = (
                stated_amount is not None
                and cls._amount_refers_to_memo(thread, stated_amount)
            )
            names_new_subject = (
                explicit_product or bool(deals)
                or (stated_amount is not None and not amount_is_reference)
                or ((explicit_merchant or explicit_card)
                    and not cls._entities_are_in_memo(thread, merchants, cards))
            )
            if not names_new_subject:
                return Operation.EXPLAIN, None

        # A comparison needs something to compare prices OF: a product in scope or a stated
        # amount. Without one, naming two merchants is deal browsing, not a purchase.
        has_financial_basis = bool(products) or amount is not None

        # Two or more peers named on ONE dimension is structurally a comparison, whatever
        # the sentence looks like ("A vs B", "A or B", "compare A and B", "A, B?").
        peer_counts = (len(merchants), len(cards), len(deals), len(products))
        if max(peer_counts) >= 2 and has_financial_basis:
            return Operation.COMPARE, None

        # An OPEN comparison stays in force across follow-ups: "which is cheaper?", "what if
        # I use SBI?" are answered over the alternatives already under discussion rather
        # than by starting an unrelated global search (spec sections 7 and 8).
        if (thread.comparison_axis and len(thread.comparison_entities) >= 2
                and has_financial_basis and not explicit_product and not deals):
            return Operation.COMPARE, None

        # Asking which single deal discounts the most is an aggregate over deal records,
        # not a purchase to price: without a product there is nothing to buy, only offers to
        # rank. The objective says which quantity to rank by; the superlative says the user
        # wants one answer rather than a browse.
        if (objective == Objective.MAX_DISCOUNT.value and SUPERLATIVE_RE.search(q_low)
                and not products and (category or merchants or deals)):
            return Operation.AGGREGATE, None

        # A stated spend figure turns a question ABOUT offers into a calculation over that
        # spend: "what deal applies to my RS 2,000 order?" asks what it will cost, not what
        # exists. Browsing has no figure in it. This is settled by the structured state, so
        # the classifier cannot turn the calculation back into a listing — which it did
        # intermittently, since its output is not deterministic.
        if (amount is not None and budget is None and not products
                and (merchants or category or deals)
                and DEAL_VOCABULARY_RE.search(q_low)):
            return Operation.COMPUTE, None

        # A pure reward question names a card and a spend but no purchase to optimize —
        # answer what the card pays, do not attach an unrequested merchant coupon
        # (spec section 19).
        # Requires an actual spend figure: without one there is nothing to compute, and the
        # turn is a rate/cap LOOKUP or a preference change, not a reward calculation.
        if (reward_question and reward_spend is not None and explicit_card
                and not products and not merchants and not deals):
            return Operation.COMPUTE, None

        # An amount with nothing to spend it on. A budget is not a product and an amount is
        # not an intent: rather than picking an arbitrary catalog row to make an answer
        # possible, ask what the purchase is (spec sections 18 and 36).
        ungrounded_amount = (
            amount is not None and budget is None
            and not products and not category and not merchants and not deals and not cards
            and not thread.has_context()
        )
        if ungrounded_amount:
            cats = sorted({
                (p.get("category") or "").lower() for p in DataIndex.get_products() if p.get("category")
            })
            listed = ", ".join(cats)
            return Operation.CLARIFY, (
                f"I can work out the cheapest way to pay RS {amount:,.0f}, but I need to know what "
                f"the payment is for. Which product or category is it — {listed}?"
            )

        return None, None

    @classmethod
    def _amount_refers_to_memo(cls, thread: ConversationThread, amount: float) -> bool:
        """
        True when this figure is one the previous answer already quoted.

        Rejected-deal headlines and minimum spends, and the recommendation's own base price,
        discount, reward and effective price. Matching one of them means the user is pointing
        at part of the last answer, not stating a new amount to spend.
        """
        known: List[float] = []
        for row in (thread.last_rejected_deals or []):
            known.extend(float(f) for f in row.get("figures", []))
        rec = thread.last_recommendation or {}
        for key in ("base_price", "discount", "price_after_discount", "reward", "effective_price"):
            if rec.get(key) is not None:
                known.append(float(rec[key]))
        for row in (thread.last_comparison or {}).get("rows", []):
            for key in ("base_price", "discount", "reward", "effective_price"):
                if row.get(key) is not None:
                    known.append(float(row[key]))
        return any(abs(amount - k) < 0.01 for k in known)

    @classmethod
    def _entities_are_in_memo(
        cls,
        thread: ConversationThread,
        merchants: List[str],
        cards: List[Dict[str, Any]],
    ) -> bool:
        """
        True when every merchant/card named this turn already appears in the stored memo.

        This is what lets "why wasn't <loser> cheaper?" be answered from the comparison we
        already computed: the entity is not a NEW subject, it is one of the rows we ranked.
        """
        known: Set[str] = set()
        rec = thread.last_recommendation or {}
        for key in ("merchant", "card_id", "deal_id"):
            if rec.get(key):
                known.add(str(rec[key]).lower())
        cmp_memo = thread.last_comparison or {}
        for row in cmp_memo.get("rows", []):
            for key in ("key", "merchant", "card_id", "deal_id", "product_id"):
                if row.get(key):
                    known.add(str(row[key]).lower())
        for e in thread.comparison_entities:
            known.add(str(e).lower())

        if not known:
            return False
        named = [m.lower() for m in merchants] + [c["card_id"].lower() for c in cards]
        return bool(named) and all(n in known for n in named)

    @classmethod
    def _classify_operation(
        cls,
        query: str,
        products: List[Dict[str, Any]],
        merchants: List[str],
        cards: List[Dict[str, Any]],
        deals: List[Dict[str, Any]],
        category: Optional[str] = None,
        amount: Optional[float] = None,
        target: Optional[float] = None,
        unresolved: Optional[List[str]] = None,
        purchase_amount: Optional[float] = None,
        reward_spend: Optional[float] = None,
        budget: Optional[float] = None,
        target_price: Optional[float] = None,
        context_hint: Optional[str] = None,
        **kwargs: Any
    ) -> Operation:
        q_low = query.lower()

        # If entity is unresolved -> ABSTAIN
        if unresolved and not (products or deals or cards):
            return Operation.ABSTAIN

        # 1. LLM classification if available. The LLM is an INTERPRETATION aid only: it
        # picks which question is being asked, never any figure in the answer. When it is
        # unavailable the deterministic fallback below takes over unchanged.
        if LLMGate.is_available():
            try:
                context_block = f"\n{context_hint}\nA question that only makes sense against that context (e.g. asking for a different card, or the resulting price/savings) is COMPUTE, not ABSTAIN or LOOKUP.\n" if context_hint else ""
                prompt = (
                    f"You are a strict financial operation classifier. Classify the user query into EXACTLY ONE operation:\n\n"
                    f"1. LOOKUP: Asking for a specific attribute or field of a named record (e.g. 'what is Millennia base rate', 'what does deal_018 offer', 'how much is the Kindle on Croma', 'what is the min spend on deal_002').\n"
                    f"2. LIST: Asking to browse or list available deals/offers matching a category or merchant (e.g. 'grocery deals at BigBasket', 'all Amex-eligible offers', 'what does Blinkit have').\n"
                    f"3. AGGREGATE: Asking for max/min/count/sum across records (e.g. 'largest discount at Amazon', 'how many travel deals', 'which card has highest grocery rate', 'cheapest merchant for MacBook').\n"
                    f"4. COMPUTE: Calculating the exact price for ONE specified combination of merchant, card, deal or stated spend without optimizing (e.g. '₹4,000 groceries on Millennia at BigBasket', 'what do I pay for the Kindle with ICICI', '₹800 electronics on SBI', '₹20,000 groceries on Millennia'). Also COMPUTE: a hypothetical follow-up ('what if I use SBI instead?', 'how much do I actually save?') asked against an already-established purchase context.\n"
                    f"5. OPTIMIZE: Finding the globally cheapest purchase route or best card for a purchase (e.g. 'cheapest way to buy iPhone 15', 'best card for ₹6,000 travel', 'how do I pay least for groceries', 'what would you recommend now?' when a product context is already active).\n"
                    f"6. COMPARE: Comparing two or more named cards, merchants, or deals side by side (e.g. 'Millennia vs SBI for groceries', 'Amazon or Flipkart for MacBook', 'is deal_002 better than deal_017').\n"
                    f"7. ELIGIBILITY: Asking a yes/no or eligibility question (e.g. 'can I use SBI for deal_018', 'does deal_004 apply to a ₹800 order', 'do I qualify for Amex bonus').\n"
                    f"8. WATCH: Setting a price alert threshold (e.g. 'tell me if X drops below Y', 'ping me when Sony hits 25k').\n"
                    f"9. STATE: Setting or updating a PERSISTENT session preference/budget (e.g. 'my budget is ₹3,300', 'I prefer Amex from now on', 'forget my card preference'). A hypothetical ('what if I use X', 'suppose I spend X') is NEVER STATE even if it names a card or amount — it is COMPUTE/OPTIMIZE.\n"
                    f"10. ABSTAIN: Out of scope or ungrounded entity not in catalog.\n"
                    f"{context_block}\n"
                    f"Query: {query}\n"
                    f"Return JSON ONLY: {{\"operation\": \"<OPERATION_NAME>\"}}"
                )
                txt = LLMGate.generate(prompt)
                if txt:
                    m = re.search(r"\{\s*\"operation\"\s*:\s*\"([A-Z_]+)\"\s*\}", txt)
                    if m:
                        cand = m.group(1).strip()
                        if hasattr(Operation, cand):
                            return Operation[cand]
            except Exception:
                pass

        # 2. Deterministic Fallback
        if any(w in q_low for w in ["ping me", "alert", "watch", "drops below", "hits", "track"]) or target is not None:
            return Operation.WATCH

        if any(w in q_low for w in ["prefer", "rather use", "actually use", "my budget is", "make that", "scratch that", "instead"]):
            if not any(w in q_low for w in ["cheapest", "best price", "buy", "purchase", "order", "what is"]):
                return Operation.STATE

        # Declaring a budget ("I have a ₹3,500 grocery budget", "my budget: 5000") is a
        # session mutation. Requires an actual figure so that merely REFERRING to the
        # budget ("cheapest X within my budget") stays a purchase request, and defers to
        # the same purchase-verb guard as above.
        if "budget" in q_low and amount is not None:
            if not any(w in q_low for w in ["cheapest", "best price", "buy", "purchase", "order", "within", "what is"]):
                return Operation.STATE

        if any(w in q_low for w in ["can i use", "is it eligible", "does deal_", "qualify", "does sbi cover", "does hdfc cover"]):
            return Operation.ELIGIBILITY

        if " vs " in q_low or " or " in q_low and (len(cards) >= 2 or len(merchants) >= 2 or len(deals) >= 2):
            return Operation.COMPARE

        if any(w in q_low for w in ["largest discount", "highest discount", "highest rate", "cheapest merchant", "how many deals", "which deal has the largest"]):
            return Operation.AGGREGATE

        if any(w in q_low for w in ["what does deal_", "details for deal_", "what is millennia base rate", "reward cap on", "min spend on", "how much is the"]):
            if not any(w in q_low for w in ["cheapest way", "buy", "how do i pay least"]):
                return Operation.LOOKUP

        if any(w in q_low for w in ["what deals", "what grocery deals", "what discounts", "all deals", "available at", "offers at", "does blinkit have"]):
            # Same principle as the generalized rule below: a stated spend figure turns a
            # browse into a calculation ("what deal applies to my ₹2,000 order?").
            if amount is None and not any(w in q_low for w in ["cheapest way", "how do i pay least"]):
                return Operation.LIST

        # Generalized deal discovery: the query is ABOUT deals/offers scoped to a merchant
        # or category, states no spend figure, and asks for no purchase. Browsing offers is
        # a LIST; the moment a figure is supplied it becomes a calculation instead, which is
        # what separates "is there a deal for X?" from "what deal applies to my ₹2,000 order?".
        if any(w in q_low for w in ["deal", "deals", "offer", "offers", "discount", "discounts", "promotion", "promotions"]):
            # A deal listing is scoped to a merchant or a category. Once a specific PRODUCT
            # is named the turn is about acquiring that product ("best price and deal for
            # the Sony WH-1000XM5 with my SBI card"), which is an optimization — the word
            # "deal" in it describes what to apply, not what to browse.
            if amount is None and not products and (merchants or category) and not any(
                w in q_low for w in ["cheapest", "pay least", "buy", "purchase", "order", "best card", "which card"]
            ):
                return Operation.LIST

        # If a single specific card AND merchant OR product are specified with amount -> COMPUTE
        if amount is not None and (cards or merchants) and not any(w in q_low for w in ["cheapest", "best way", "best card"]):
            return Operation.COMPUTE

        return Operation.OPTIMIZE
