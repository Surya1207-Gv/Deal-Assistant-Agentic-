from __future__ import annotations
from typing import Dict, Any, Optional
from app.tools.compare_prices import compare_prices
from app.core.provenance import Value, Provenance
from app.core.memory import MemoryManager

def watch_price(product_name: str, target_price: Optional[float] = None, session_id: str = "default") -> Dict[str, Any]:
    """
    Tool 5: Registers price watch for product, calculates gap to target price,
    and returns status with tagged Value objects.
    """
    cmp_res = compare_prices(product_name)
    resolved_name = cmp_res.get("product_name", product_name)
    current_lowest = cmp_res.get("lowest_price")
    current_amount = current_lowest.amount if current_lowest else float("inf")

    clean_target = float(target_price) if target_price is not None else 0.0

    target_val = Value(
        amount=clean_target,
        provenance=Provenance.SOURCE,
        record_id="target_price"
    )

    gap_amount = max(0.0, current_amount - clean_target)
    gap_val = Value(
        amount=gap_amount,
        provenance=Provenance.DERIVED,
        record_id="gap_price",
        formula=f"{current_amount} - {clean_target}",
        inputs=["current_price", "target_price"]
    )

    target_met = current_amount <= clean_target

    # Register watch in session memory
    session = MemoryManager.get_session(session_id)
    session.price_watches[resolved_name] = clean_target

    lowest_merch = cmp_res.get("lowest_price_merchant") or "Catalog"

    return {
        "product_name": resolved_name,
        "product_id": cmp_res.get("product_id"),
        "found": cmp_res.get("found", False),
        "target_price": target_val,
        "current_lowest_price": current_lowest,
        "lowest_price_merchant": lowest_merch,
        "gap": gap_val,
        "target_met": target_met,
        "status": "WATCH_REGISTERED",
        "message": f"Price watch registered for {resolved_name} at target ₹{clean_target:.0f}. Current lowest is ₹{current_amount:.0f} on {lowest_merch}."
    }
