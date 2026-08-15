from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class EvalMetrics:
    total_cases: int
    recall_at_3: float
    mrr: float
    answer_accuracy: float
    hallucination_rate: float
    abstention_precision: float
    parametric_leak_rate: float
    injection_resistance: float
    p50_latency_s: float
    p95_latency_s: float
    mean_cost_usd: float
    planner_mode: str
    reranker_enabled: bool
    passed_verified: int = 0
    total_verified: int = 0
    case_details: List[Dict[str, Any]] = field(default_factory=list)

class MetricsCalculator:
    @staticmethod
    def calculate_retrieval_mrr(gold_id: Optional[str], retrieved_ids: List[str]) -> float:
        if not gold_id:
            return 0.0
        for rank, rid in enumerate(retrieved_ids, 1):
            if rid.lower() == gold_id.lower():
                return 1.0 / rank
        return 0.0

    @staticmethod
    def calculate_recall_at_3(gold_id: Optional[str], retrieved_ids: List[str]) -> bool:
        if not gold_id:
            return False
        top_3 = [rid.lower() for rid in retrieved_ids[:3]]
        return gold_id.lower() in top_3
