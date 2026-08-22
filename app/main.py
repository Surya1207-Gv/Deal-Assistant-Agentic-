from __future__ import annotations
import sys
import time
import json
import asyncio
import re
import traceback
from pathlib import Path
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent.graph import graph_app
from app.core.memory import MemoryManager
from app.core.telemetry import TelemetryTracker

app = FastAPI(title="Deal Assistant API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str
    session_id: str = "default_session"
    budget: float = None
    preferred_card: str = None
    category: str = None

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "Deal Assistant Agentic RAG + Planner"}

async def event_generator(request: ChatRequest) -> AsyncGenerator[str, None]:
    start_time = time.time()
    session = MemoryManager.get_session(request.session_id)

    q_lower = (request.query or "").lower()

    if request.category:
        session.category = request.category
    elif "grocery" in q_lower or "groceries" in q_lower:
        session.category = "groceries"
    elif "electronic" in q_lower or "phone" in q_lower:
        session.category = "electronics"

    match_budget = re.search(r"(?:budget\s*is\s*|worth\s*|spending\s*|my\s*budget\s*(?:is\s*)?)(?:RS\s*|₹\s*)?(\d{3,6})", request.query or "", re.IGNORECASE)
    if match_budget:
        session.budget = float(match_budget.group(1))
        session.conversation_state["budget"] = float(match_budget.group(1))
    elif request.budget is not None:
        session.budget = request.budget
        session.conversation_state["budget"] = request.budget

    card_aliases = {
        "hdfc millennia": "hdfc_millennia",
        "millennia": "hdfc_millennia",
        "sbi cashback": "sbi_cashback",
        "sbi": "sbi_cashback",
        "axis ace": "axis_ace",
        "axis": "axis_ace",
        "amex smart earn": "amex_smartearn",
        "amex": "amex_smartearn",
        "icici amazon pay": "icici_amazon_pay",
        "icici": "icici_amazon_pay",
        "hdfc regalia": "hdfc_regalia",
        "regalia": "hdfc_regalia",
    }

    preferred_card = None
    for phrase, card_id in card_aliases.items():
        if phrase in q_lower:
            preferred_card = card_id
            break
    if preferred_card is not None:
        if "actually" in q_lower or "instead" in q_lower or "use" in q_lower:
            session.preferred_card = preferred_card
            session.conversation_state["preferred_card"] = preferred_card
        elif "prefer" in q_lower or "i prefer" in q_lower:
            session.preferred_card = preferred_card
            session.conversation_state["preferred_card"] = preferred_card
        elif "for " in q_lower and any(cat in q_lower for cat in ["electronics", "groceries", "travel", "shopping", "food", "bills"]):
            category_name = None
            if "electronics" in q_lower:
                category_name = "electronics"
            elif "groceries" in q_lower or "grocery" in q_lower:
                category_name = "groceries"
            elif "travel" in q_lower:
                category_name = "travel"
            elif "shopping" in q_lower:
                category_name = "shopping"
            elif "food" in q_lower or "dining" in q_lower:
                category_name = "food"
            elif "bills" in q_lower:
                category_name = "bills"
            if category_name:
                session.conversation_state.setdefault("category_preferences", {})
                session.conversation_state["category_preferences"][category_name] = preferred_card
    if request.preferred_card is not None:
        session.preferred_card = request.preferred_card
        session.conversation_state["preferred_card"] = request.preferred_card

    initial_state = {
        "session_id": request.session_id,
        "query": request.query or "",
        "category": session.category or "groceries",
        "budget": session.budget,
        "preferred_card": session.preferred_card,
        "conversation_state": session.conversation_state,
        "spend_to_date": session.spend_to_date,
        "messages": []
    }

    try:
        final_state = graph_app.invoke(initial_state)

        planned_tools = final_state.get("planned_tools", [])
        planner_mode = final_state.get("planner_mode") or "llm"
        plan_payload = {
            "planned_tools": planned_tools,
            "planner_mode": planner_mode
        }
        yield f"event: plan\ndata: {json.dumps(plan_payload)}\n\n"
        await asyncio.sleep(0.05)

        response_text = final_state.get("final_response", "")
        words = response_text.split(" ")
        for word in words:
            token_payload = {"token": word + " "}
            yield f"event: token\ndata: {json.dumps(token_payload)}\n\n"
            await asyncio.sleep(0.02)

        citations = final_state.get("citations", [])
        citations_payload = {"citations": citations}
        yield f"event: citations\ndata: {json.dumps(citations_payload)}\n\n"
        await asyncio.sleep(0.05)

        elapsed = time.time() - start_time
        telemetry_item = TelemetryTracker.record_turn(request.session_id, elapsed, input_tokens=180, output_tokens=100)
        telemetry_payload = {
            "latency_s": telemetry_item.latency_seconds,
            "cost_usd": telemetry_item.estimated_cost_usd,
            "planner_mode": planner_mode,
            "provenance_validated": final_state.get("provenance_valid", True)
        }
        yield f"event: telemetry\ndata: {json.dumps(telemetry_payload)}\n\n"
        await asyncio.sleep(0.05)

        yield f"event: done\ndata: {json.dumps({'status': 'completed'})}\n\n"
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    return StreamingResponse(
        event_generator(request),
        media_type="text/event-stream"
    )

@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    session = MemoryManager.get_session(session_id)
    return {
        "session_id": session.session_id,
        "budget": session.budget,
        "category": session.category,
        "preferred_card": session.preferred_card,
        "spend_to_date": session.spend_to_date,
        "price_watches": session.price_watches
    }

@app.post("/sessions/{session_id}/reset")
def reset_session(session_id: str):
    MemoryManager.reset_session(session_id)
    return {"status": "reset_successful", "session_id": session_id}
