from __future__ import annotations
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

        # 1. Mask known product names so their model numbers (e.g. "Apple iPhone 15 128GB", "Sony WH-1000XM5") aren't parsed as prices
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
            num_str = match.group(1).replace(",", "")
            is_percent = match.group(2) == "%" or "%" in raw
            if not num_str:
                continue
            try:
                val = float(num_str)
                # Percentages like 5%, 10% are verified or treated as rate
                if is_percent:
                    continue
                extracted.append((raw, round(val, 2)))
            except ValueError:
                continue
        return extracted

    @classmethod
    def validate(cls, draft_text: str, trace: List[Value]) -> tuple[bool, List[str], List[str]]:
        extracted = cls.extract_numeric_tokens(draft_text)
        if not extracted:
            return True, [], []

        if not trace and extracted:
            unverified_tokens = [raw for raw, _ in extracted]
            return False, [], unverified_tokens

        known_values: Set[float] = {v.rounded_amount() for v in trace}
        extended_known = set(known_values)
        for v in trace:
            amt = v.rounded_amount()
            extended_known.add(amt)
            if 0 < amt <= 1.0:
                extended_known.add(round(amt * 100, 2))

        validated = []
        unverified = []

        for raw_token, val in extracted:
            if val in extended_known or any(abs(val - kv) < 0.01 for kv in extended_known):
                validated.append(raw_token)
            else:
                unverified.append(raw_token)

        is_valid = len(unverified) == 0
        return is_valid, validated, unverified
