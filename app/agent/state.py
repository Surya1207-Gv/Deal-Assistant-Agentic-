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
    
    # RAG Retrieval state
    retrieved_records: List[Dict[str, Any]]
    abstained: bool
    max_retrieval_score: float
    
    # Tool Execution state
    planned_tools: List[str]
    tool_results: Dict[str, Any]
    tool_mapping: Dict[str, str]
    
    # Reward Math state
    payment_options: List[PaymentOption]
    best_option: Optional[PaymentOption]
    runner_up_option: Optional[PaymentOption]
    trace: List[Value]
    skip_planning: bool
    is_info_query: bool
    is_tie: bool
    tied_merchants: List[str]
    
    # Provenance Validation state
    draft_response: str
    provenance_valid: bool
    unverified_tokens: List[str]
    final_response: str
    citations: List[str]
    
    # Telemetry
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    planner_mode: str
