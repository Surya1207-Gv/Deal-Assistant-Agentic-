from __future__ import annotations
import sys
import site
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure site packages (global user site & venv Lib) are included in sys.path
user_site = site.getusersitepackages()
if user_site and user_site not in sys.path:
    sys.path.append(user_site)

venv_site = PROJECT_ROOT / "venv" / "Lib" / "site-packages"
if venv_site.exists() and str(venv_site) not in sys.path:
    sys.path.append(str(venv_site))

from app.agent.graph import build_agent_graph
from app.core.memory import MemoryManager

def print_banner():
    print("=" * 65)
    print("      BILLGPT / DEAL ASSISTANT INTERACTIVE TERMINAL")
    print("=" * 65)
    print("Type your shopping query, product search, or reward question.")
    print("Commands:")
    print("  'exit' or 'quit' : Exit the CLI")
    print("  'reset'          : Clear session memory / spend to date")
    print("  'help'           : Show sample query ideas")
    print("=" * 65 + "\n")

def print_help():
    print("\n--- SAMPLE QUERIES YOU CAN TRY ---")
    print("1. Product comparison: 'Find best price and deal for Sony WH-1000XM5 headphones with SBI Cashback card'")
    print("2. Multi-merchant check: 'Cheapest place to buy Apple MacBook Air M2 256GB with ICICI Amazon Pay'")
    print("3. Fashion/Shoes: 'I need Nike Pegasus 40 Running Shoes'")
    print("4. Food/Dining: 'Swiggy gourmet dinner meal box for 4 using Amex SmartEarn'")
    print("5. Cap binding: 'Buying RS 6000 groceries on HDFC Millennia after spending RS 300 of category cap'")
    print("6. Multi-turn follow-up: 'Actually my budget is RS 2500'")
    print("7. Price watch: 'Tell me if the MacBook Air M2 drops below RS 85000'")
    print("8. Out of domain: 'Find deals on Tesla Cybertruck EV charging stations'")
    print("----------------------------------\n")

def run_interactive_cli():
    print_banner()
    graph = build_agent_graph()
    session_id = "terminal_user_session"
    session = MemoryManager.get_session(session_id)

    while True:
        try:
            query = input("\n\033[1;36mUser Query > \033[0m").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if not query:
            continue

        if query.lower() in ["exit", "quit"]:
            print("Exiting Deal Assistant. Goodbye!")
            break

        if query.lower() == "reset":
            MemoryManager.reset_session(session_id)
            session = MemoryManager.get_session(session_id)
            print("Session state reset.")
            continue

        if query.lower() == "help":
            print_help()
            continue

        state_input = {
            "session_id": session_id,
            "query": query,
            "category": session.category,
            "budget": session.budget,
            "preferred_card": session.preferred_card,
            "spend_to_date": session.spend_to_date,
            "history": session.history,
            "messages": []
        }

        print("\nProcessing request...")
        result = graph.invoke(state_input)

        # Print Execution details
        mode = result.get("planner_mode", "llm").upper()
        tools = result.get("planned_tools", [])
        print(f"PLANNER MODE : {mode}")
        print(f"PLANNED TOOLS: {' -> '.join(tools) if tools else 'None'}")

        retrieved = result.get("retrieved_records", [])
        if retrieved:
            print(f"RETRIEVED    : {len(retrieved)} records")
            for r in retrieved[:3]:
                print(f"  [{r.get('deal_id')}] {r.get('title')} ({r.get('merchant')})")

        # A comparison is about several alternatives, so show the candidate space that
        # was actually costed rather than only the winner's derivation.
        cands = result.get("comparison_candidates") or []
        if cands:
            axis = result.get("comparison_axis", "option")
            print(f"\nCANDIDATE SPACE ({len(cands)} {axis}s, each independently derived):")
            for c in cands:
                o = c.option
                print(
                    f"  - {c.label:<34}"
                    f" base RS {o.base_price.amount:>10,.2f}"
                    f" | discount RS {o.discount_applied.amount:>8,.2f} ({o.discount_source_id or 'none'})"
                    f" | reward RS {o.reward_earned.amount:>8,.2f} ({o.card_id})"
                    f" | effective RS {o.effective_price.amount:>10,.2f}"
                )

        best_opt = result.get("best_option")
        if best_opt:
            print("\nFINANCIAL DERIVATION:")
            print(f"  • Base Price          : ₹{best_opt.base_price.amount:,.2f} [{best_opt.base_price.record_id}]")
            print(f"  • Discount Applied    : ₹{best_opt.discount_applied.amount:,.2f} ({best_opt.discount_source_id or 'none'})")
            print(f"  • Post-Discount Price : ₹{best_opt.price_after_discount.amount:,.2f}")
            print(f"  • Best Card           : {best_opt.card_id}")
            print(f"  • Reward Earned       : ₹{best_opt.reward_earned.amount:,.2f}")
            if best_opt.cap_hit:
                print(f"  • Cap Headroom Hit    : Yes (Lost to cap: ₹{best_opt.reward_lost_to_cap.amount:,.2f})")
            print(f"  • Effective Final     : ₹{best_opt.effective_price.amount:,.2f}")

        prov_valid = result.get("provenance_valid", True)
        unverified = result.get("unverified_tokens", [])
        print(f"PROVENANCE   : {'PASSED (Zero Hallucination)' if prov_valid else 'BLOCKED: ' + str(unverified)}")

        print("\n" + "-" * 50)
        print("FINAL ASSISTANT RESPONSE:")
        print("-" * 50)
        print(result.get("final_response", ""))
        print("-" * 50)

        citations = result.get("citations", [])
        if citations:
            print(f"Citations: {citations}")

if __name__ == "__main__":
    run_interactive_cli()
