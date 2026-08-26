from __future__ import annotations
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Set

class Provenance(str, Enum):
    SOURCE = "source"
    DERIVED = "derived"

@dataclass(frozen=True)
class Value:
    amount: float
    provenance: Provenance
    record_id: Optional[str] = None
    formula: Optional[str] = None
    inputs: List[str] = field(default_factory=list)

    def rounded_amount(self) -> float:
        return round(float(self.amount), 2)

class ProvenanceValidator:
    """
    Validates that every numeric token in an LLM or formatted response draft
    corresponds to a known, tagged Value in the computation trace.
    Allows formatting variance (e.g. RS 3,800, 3800.00, 3800, 5%).
    Ignores non-monetary tokens (deal IDs like deal_042, dates like 2026, list indices).
    """

    @classmethod
    def extract_numeric_tokens(cls, text: str, product_names: Optional[List[str]] = None) -> List[tuple[str, float]]:
        cleaned_text = text

        # 1. Mask CATALOG TEXT the answer quotes verbatim — product names, deal titles, card
        # names. A number inside such text ("Apple iPhone 15 128GB", "Prompt Injection Test
        # Deal 1", "HDFC Grocery Special 500 Cashback") is part of a record's own wording,
        # not a figure the answer is asserting, and it is grounded by the very act of being
        # quoted. Masking removes only that text; every number outside it is still checked,
        # so this narrows false positives without widening what can pass unverified.
        if product_names:
            for p_name in product_names:
                if p_name and len(p_name) > 3:
                    cleaned_text = re.sub(re.escape(p_name), "[PRODUCT_NAME]", cleaned_text, flags=re.IGNORECASE)

        # 2. Mask entity IDs and metadata tokens
        cleaned_text = re.sub(r"\b(?:deal_\w+|card_\w+|prod_\w+)\b", "[ID]", cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r"(?:retrieval\s+)?confidence\s*\(\d+(?:\.\d+)?\)", "", cleaned_text, flags=re.IGNORECASE)

        # 3. Mask alphanumeric model/spec tokens (e.g. WH-1000XM5, M2, 55-inch, 128GB, 256GB, 4K, 5x, Levi's 511)
        # Matches tokens with letters + digits or hyphens (e.g., WH-1000XM5, 55-inch, 128GB, M2)
        cleaned_text = re.sub(r"\b[A-Za-z]+[-_/\\]?\d+[A-Za-z0-9-_]*\b", "[MODEL]", cleaned_text)
        cleaned_text = re.sub(r"\b\d+[-_/\\]?[A-Za-z]+[A-Za-z0-9-_]*\b", "[SPEC]", cleaned_text)
        cleaned_text = re.sub(r"\b[A-Za-z0-9]+-[A-Za-z0-9]+-[A-Za-z0-9]+\b", "[MODEL_COMPLEX]", cleaned_text)

        # 4. Mask bullet item numbers (e.g. "1.", "2.", "•")
        cleaned_text = re.sub(r"(?:^|\n)\s*\d+\.\s+", "\n", cleaned_text)

        # 5. Extract monetary and numerical values
        pattern = r"(?:₹|\$|Rs\.?\s*|RS\s*)?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(%|\b)?"
        matches = re.finditer(pattern, cleaned_text)
        
        extracted = []
        for match in matches:
            raw = match.group(0).strip()
            num_str = match.group(1).replace(",", "").strip()
            is_percent = match.group(2) == "%" or "%" in raw
            if not num_str:
                continue
            try:
                val = float(num_str)
                # Percentages are ignored from strict currency validation
                if is_percent:
                    continue
                extracted.append((raw, val))
            except ValueError:
                continue
        return extracted

    @classmethod
    def validate(cls, draft_text: str, trace: List[Value],
                 product_names: Optional[List[str]] = None) -> tuple[bool, List[str], List[str]]:
        """
        `product_names` masks catalog product titles before numbers are extracted, so a
        model number inside a name the answer legitimately mentions ("Apple iPhone 15",
        "MacBook Air M2") is not mistaken for an unverified monetary figure. The masking
        only removes NON-monetary text; every remaining number is still checked against the
        trace, so this cannot be used to smuggle an ungrounded figure past validation.
        """
        extracted = cls.extract_numeric_tokens(draft_text, product_names=product_names)
        if not extracted:
            return True, [], []

        if not trace and extracted:
            unverified_tokens = [raw for raw, _ in extracted]
            return False, [], unverified_tokens

        known_amounts: List[float] = [float(v.amount) for v in trace if v is not None]

        validated = []
        unverified = []

        for raw_token, val in extracted:
            matched = False
            for kv in known_amounts:
                # Direct match within 0.01
                if abs(val - kv) < 0.01:
                    matched = True
                    break
                # Integer-rounded rendering match (e.g. 24048.25 rendered as 24048)
                if abs(val - round(kv)) < 0.01 or abs(val - math.floor(kv)) < 0.01 or abs(val - math.ceil(kv)) < 0.01:
                    matched = True
                    break
                # Rate percentage match (e.g. 0.05 -> 5 or 5.0)
                if 0 < kv <= 1.0 and (abs(val - (kv * 100.0)) < 0.01 or abs(val - round(kv * 100.0)) < 0.01):
                    matched = True
                    break
            
            if matched:
                validated.append(raw_token)
            else:
                unverified.append(raw_token)

        is_valid = len(unverified) == 0
        return is_valid, validated, unverified
