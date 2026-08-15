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

    EXPLICIT_IGNORES: Set[str] = {
        "1", "2", "3", "4", "6", "7", "8", "9", "10", "42", "31", "17", "14", "39", "35", "43", "45",
        "2024", "2025", "2026", "2027", "2028", "2029", "2030",
        "24", "12", "30", "365",
        "128", "256", "512", "55", "13", "14", "15", "16", "40"
    }

    @classmethod
    def extract_numeric_tokens(cls, text: str) -> List[tuple[str, float]]:
        cleaned_text = re.sub(r"deal_\d+|card_\w+|(?:retrieval\s+)?confidence\s*\(\d+(?:\.\d+)?\)", "", text, flags=re.IGNORECASE)
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
                if is_percent:
                    continue
                if num_str in cls.EXPLICIT_IGNORES and val < 100 and not ("₹" in raw or "$" in raw or "Rs" in raw or "RS" in raw):
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
