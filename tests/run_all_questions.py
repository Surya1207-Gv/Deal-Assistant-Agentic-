from __future__ import annotations
import os
import sys
import json
import time
import re
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 standard output encoding on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from app.agent.graph import graph_app
from app.core.memory import MemoryManager
from app.core.telemetry import TelemetryTracker
from app.core.provenance import ProvenanceValidator


def run_all():
    eval_file = PROJECT_ROOT / "eval" / "labeled_set.json"
    if not eval_file.exists():
        print(f"Error: {eval_file} not found.", file=sys.stderr)
        sys.exit(1)

    with open(eval_file, "r", encoding="utf-8") as f:
        cases = json.load(f)

    summary_rows = []
    multiturn_session_id = "multiturn_eval_session"
    MemoryManager.reset_session(multiturn_session_id)

    for case in cases:
        case_id = case.get("id", "")
        case_type = case.get("type", "")
        query = case.get("query", "")
        is_multiturn = case_type == "multiturn" or case_id in ["eval_16", "eval_17", "eval_18", "eval_19"]

        if is_multiturn:
            session_id = multiturn_session_id
            session = MemoryManager.get_session(session_id)
            print(f"\n[SESSION MEMORY BEFORE TURN] budget={session.budget}, category={session.category}, preferred_card={session.preferred_card}")
        else:
            session_id = f"session_{case_id}"
            MemoryManager.reset_session(session_id)
            session = MemoryManager.get_session(session_id)
        if "prior_spend" in case and isinstance(case["prior_spend"], dict):
            session.spend_to_date.update(case["prior_spend"])

        initial_state = {
            "session_id": session_id,
            "query": query,
            "category": session.category,
            "budget": session.budget,
            "preferred_card": session.preferred_card,
            "spend_to_date": session.spend_to_date,
            "history": session.history,
            "messages": []
        }

        start_time = time.time()
        result = graph_app.invoke(initial_state)
        elapsed = time.time() - start_time

        telemetry_item = TelemetryTracker.record_turn(session_id, elapsed, input_tokens=180, output_tokens=100)

        # Update session memory state for subsequent multi-turn queries
        if is_multiturn:
            best_opt_res = result.get("best_option")
            if best_opt_res and best_opt_res.card_id:
                session.preferred_card = best_opt_res.card_id
            if result.get("category"):
                session.category = result["category"]

        planner_mode = result.get("planner_mode", "llm")
        planned_tools = result.get("planned_tools", [])
        retrieved = result.get("retrieved_records", [])
        tool_mapping = result.get("tool_mapping", {})
        trace = result.get("trace", [])
        final_response = result.get("final_response", "")
        citations = result.get("citations", [])
        best_opt = result.get("best_option")

        is_valid, validated, unverified = ProvenanceValidator.validate(final_response, trace)
        prov_status = "PASSED" if is_valid else "FAILED"
        total_checked = len(validated) + len(unverified)

        # Print Case Block
        print("=" * 66)
        if planner_mode != "llm":
            print(f"WARNING: LLM unavailable - fallback triggered ({planner_mode})")
        print(f"CASE ID      : {case_id}  ({case_type})")
        print(f"User Query   > {query}")
        print(f"PLANNER MODE : {planner_mode}")
        print(f"PLANNED TOOLS: {' -> '.join(planned_tools) if planned_tools else 'None'}")
        print(f"RETRIEVED    : {len(retrieved)} records")
        for r in retrieved:
            score = r.get("retrieval_score", 0.0)
            print(f"  [{r.get('deal_id')}] {r.get('title')} ({r.get('merchant')})  score={score:.2f}")

        if tool_mapping:
            print("TOOL RESULTS :")
            for tool_name, summary in tool_mapping.items():
                print(f"  • {tool_name} -> {summary}")
        else:
            print("TOOL RESULTS : N/A")

        if trace:
            print("REWARD CALC  :")
            for v in trace:
                print(f"  • {v}")
        else:
            print("REWARD CALC  : N/A (no calculation path)")

        print(f"PROVENANCE   : {total_checked} checked, {len(validated)} validated, {len(unverified)} unverified  <{prov_status}>")
        print("-" * 66)
        print("FINAL RESPONSE:")
        print(final_response)
        print("-" * 66)
        print(f"Citations    : {citations}")
        print(f"LATENCY/COST : {elapsed:.2f}s / ${telemetry_item.estimated_cost_usd:.6f}")
        print("=" * 66)

        eff_price_str = f"₹{best_opt.effective_price.amount:,.2f}" if best_opt else "N/A"
        summary_rows.append({
            "case": case_id,
            "category": case_type,
            "effective_price": eff_price_str,
            "provenance": prov_status,
            "citations_count": len(citations)
        })

    # Summary Table
    print("\n" + "=" * 66)
    print("SUMMARY TABLE:")
    print("=" * 66)
    print(f"| {'Case':<8} | {'Category':<22} | {'Effective price':<24} | {'Provenance':<10} | {'Citations count':<15} |")
    print(f"|{'-'*10}|{'-'*24}|{'-'*26}|{'-'*12}|{'-'*17}|")
    for row in summary_rows:
        print(f"| {row['case']:<8} | {row['category']:<22} | {row['effective_price']:<24} | {row['provenance']:<10} | {row['citations_count']:<15} |")


if __name__ == "__main__":
    run_all()
