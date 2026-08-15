from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class TurnTelemetry:
    turn_id: str
    latency_seconds: float
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0

class TelemetryTracker:
    """
    Tracks latency, token usage, and execution cost per turn.
    Computes p50 and p95 latency statistics.
    """
    _turns: List[TurnTelemetry] = []

    # Model rate pricing defaults (e.g. Gemini / GPT-3.5/4o-mini rates per 1k tokens)
    INPUT_COST_PER_1K = 0.00015
    OUTPUT_COST_PER_1K = 0.00060

    @classmethod
    def record_turn(cls, turn_id: str, latency: float, input_tokens: int = 0, output_tokens: int = 0) -> TurnTelemetry:
        cost = (input_tokens / 1000.0) * cls.INPUT_COST_PER_1K + (output_tokens / 1000.0) * cls.OUTPUT_COST_PER_1K
        telemetry = TurnTelemetry(
            turn_id=turn_id,
            latency_seconds=round(latency, 3),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=round(cost, 6)
        )
        cls._turns.append(telemetry)
        return telemetry

    @classmethod
    def get_summary(cls) -> Dict[str, float]:
        if not cls._turns:
            return {
                "p50_latency_s": 0.0,
                "p95_latency_s": 0.0,
                "mean_cost_usd": 0.0,
                "total_turns": 0
            }

        latencies = sorted([t.latency_seconds for t in cls._turns])
        n = len(latencies)
        p50 = latencies[int(n * 0.50)]
        p95 = latencies[min(n - 1, int(n * 0.95))]
        mean_cost = sum(t.estimated_cost_usd for t in cls._turns) / n

        return {
            "p50_latency_s": round(p50, 3),
            "p95_latency_s": round(p95, 3),
            "mean_cost_usd": round(mean_cost, 6),
            "total_turns": n
        }
