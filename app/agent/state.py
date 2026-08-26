from __future__ import annotations
from typing import List, Dict, Any, Optional, TypedDict
from app.core.provenance import Value
from app.core.reward_engine import PaymentOption

class AgentState(TypedDict, total=False):
    session_id: str
    messages: List[Dict[str, str]]
    query: str
    category: Optional[str]
    merchant: Optional[str]
    budget: Optional[float]
    preferred_card: Optional[str]
    conversation_state: Dict[str, Any]
    spend_to_date: Dict[str, float]
    
    # Query Resolution state
    resolved_query: Any
    operation: str
    
    # RAG Retrieval state
    retrieved_records: List[Dict[str, Any]]
    # Deal records whose text tried to override instructions. They are sanitized rather than
    # dropped, and listed here so callers can see the defence fired.
    excluded_injection_records: List[str]
    abstained: bool
    max_retrieval_score: float
    
    # Tool Execution state
    planned_tools: List[str]
    tool_results: Dict[str, Any]
    tool_mapping: Dict[str, str]
    # What the planner's tool calls independently confirmed about the derived answer, and
    # any point on which they contradicted it. See graph.corroborate_with_tools.
    tool_corroboration: Dict[str, Any]
    
    # Reward Math state
    payment_options: List[PaymentOption]
    best_option: Optional[PaymentOption]
    runner_up_option: Optional[PaymentOption]
    trace: List[Value]
    skip_planning: bool
    is_info_query: bool
    is_tie: bool
    tied_merchants: List[str]
    # The joint-best options, the metric they tie on, and the dimension that distinguishes
    # them. Presentation only — the ranking value itself comes from the engine.
    tie: Optional[Dict[str, Any]]

    # Comparison state: the axis being compared and one fully-derived candidate per entity
    # on it. Present whenever a COMPARE ran, so the CLI/API can show every alternative that
    # was costed rather than only the winner.
    comparison_axis: Optional[str]
    comparison_candidates: List[Any]
    # Visible deals the candidate space declined, with the reason. A report over the
    # candidate space — never an input to it.
    rejected_visible_deals: List[Any]
    # Which retrieved records the candidate space admitted vs declined after eligibility.
    evidence_report: Dict[str, List[str]]
    
    # Provenance Validation state
    draft_response: str
    provenance_valid: bool
    # Set by a handler that already validated its own draft against its own trace, so the
    # validation node keeps that verdict instead of passing the answer through unchecked.
    provenance_checked: bool
    # Catalog wording the answer quotes verbatim; masked before numeric extraction so a
    # record's own title is not mistaken for an asserted figure.
    provenance_masked_names: List[str]
    unverified_tokens: List[str]
    final_response: str
    citations: List[str]
    
    # Telemetry
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    planner_mode: str
