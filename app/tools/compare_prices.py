from __future__ import annotations
from typing import Dict, Any, Optional
from app.rag.index import DataIndex
from app.core.provenance import Value, Provenance

def compare_prices(product_name: str) -> Dict[str, Any]:
    """
    Tool 2: Compares multi-merchant prices for products in products.json.
    Returns merchant prices as tagged Value objects.
    Safely converts input product_name to string.
    """
    products = DataIndex.get_products()
    product_lower = str(product_name or "").lower()

    matched_product: Optional[Dict[str, Any]] = None

    # 1. Direct name match or substring match
    for p in products:
        p_name = p["name"].lower()
        if p_name in product_lower or product_lower in p_name:
            matched_product = p
            break

    # 2. Token match fallback for specific product keywords
    if not matched_product:
        tokens = [t for t in product_lower.split() if len(t) > 3 and t not in ["deal", "deals", "best", "price", "card", "with", "from", "find", "order", "want", "need", "offer", "discount"]]
        for p in products:
            p_name = p["name"].lower()
            if any(t in p_name for t in tokens):
                matched_product = p
                break

    if not matched_product:
        return {
            "found": False,
            "product_name": product_name,
            "merchant_prices": {},
            "lowest_price_merchant": None,
            "lowest_price": None
        }

    prod_id = matched_product["product_id"]
    raw_prices = matched_product["prices"]
    
    merchant_values: Dict[str, Value] = {}
    lowest_merchant = None
    lowest_amount = float("inf")

    for merchant, price in raw_prices.items():
        v = Value(
            amount=float(price),
            provenance=Provenance.SOURCE,
            record_id=f"{prod_id}_{merchant.lower()}"
        )
        merchant_values[merchant] = v
        if float(price) < lowest_amount:
            lowest_amount = float(price)
            lowest_merchant = merchant

    # Honor explicitly requested merchant if present in catalog prices
    for m in raw_prices.keys():
        if m.lower() in product_lower:
            lowest_merchant = m
            break

    lowest_val = merchant_values[lowest_merchant] if lowest_merchant else None

    return {
        "found": True,
        "product_id": prod_id,
        "product_name": matched_product["name"],
        "category": matched_product["category"],
        "merchant_prices": merchant_values,
        "raw_prices": raw_prices,
        "lowest_price_merchant": lowest_merchant,
        "lowest_price": lowest_val
    }
