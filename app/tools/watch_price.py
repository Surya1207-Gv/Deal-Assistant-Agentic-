from __future__ import annotations
from typing import Dict, Any
from app.tools.compare_prices import compare_prices
from app.core.provenance import Value, Provenance
from app.core.memory import MemoryManager

def watch_price(product_name: str, target_price: float = 25000.0, session_id: str = "default") -> Dict[str, Any]:
    """
    Tool 5: Registers price watch for product, calculates gap to target price,
    and returns status with tagged Value objects.
    """
    clean_target = float(target_price if target_price is not None else 25000.0)
    cmp_res = compare_prices(product_name)

    target_val = Value(
        amount=clean_target,
        provenance=Provenance.SOURCE,
        record_id="user_target_price"
    )

    current_lowest = cmp_res.get("lowest_price")
    current_amount = current_lowest.amount if current_lowest else float("inf")

    gap_amount = max(0.0, current_amount - float(target_price))
    gap_val = Value(
        amount=gap_amount,
        provenance=Provenance.DERIVED,
        formula=f"{current_amount} - {target_price}",
        inputs=["current_price", "target_price"]
    )

    target_met = current_amount <= float(target_price)

    # Register watch in session memory
    session = MemoryManager.get_session(session_id)
    session.price_watches[product_name] = float(target_price)

    return {
        "product_name": cmp_res.get("product_name", product_name),
        "found": cmp_res.get("found", False),
        "target_price": target_val,
        "current_lowest_price": current_lowest,
        "lowest_price_merchant": cmp_res.get("lowest_price_merchant"),
        "gap": gap_val,
        "target_met": target_met,
        "status": "WATCH_REGISTERED",
        "message": f"Price watch registered for {product_name} at target ₹{target_price}. Current lowest is ₹{current_amount} on {cmp_res.get('lowest_price_merchant')}."
    }
