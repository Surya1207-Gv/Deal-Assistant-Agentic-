from __future__ import annotations
import re
from typing import Dict, Any, Optional
from app.rag.index import DataIndex
from app.core.provenance import Value, Provenance

def compare_prices(product_name: str) -> Dict[str, Any]:
    """
    Tool 2: Compares multi-merchant prices for products in products.json.
    Returns merchant prices as tagged Value objects.
    Safely converts input product_name to string.
    """
    DataIndex.load_all(force=True)
    products = DataIndex.get_products()
    product_lower = str(product_name or "").lower()

    matched_product: Optional[Dict[str, Any]] = None

    # 1. Direct name match or substring match
    for p in products:
        p_name = p["name"].lower()
        if p_name in product_lower:
            matched_product = p
            break

    # 2. Token match fallback with digit model scoring
    if not matched_product:
        query_digits = {d for d in re.findall(r"\b\d+\b", product_lower) if int(d) < 1000}
        stopwords = {"can", "get", "the", "for", "and", "deal", "deals", "best", "price", "prices", "card", "cards", "with", "from", "find", "order", "want", "need", "offer", "discount", "cheaper", "cheapest", "way", "buy", "purchase", "where", "how", "what", "much", "tell"}
        tokens = [t for t in re.findall(r"[a-z0-9]+", product_lower) if t not in stopwords]
        
        best_match = None
        max_score = 0

        for p in products:
            p_name = p["name"].lower()
            p_digits = {d for d in re.findall(r"\b\d+\b", p_name) if int(d) < 1000}
            p_tokens = set(re.findall(r"[a-z0-9]+", p_name))
            
            # If query specified specific model numbers (e.g. 16 vs 15), product must have matching digits
            if query_digits and p_digits and not query_digits.intersection(p_digits):
                continue
                
            token_matches = sum(2 for t in tokens if t in p_tokens or (len(t) > 3 and t in p_name))
            digit_matches = sum(5 for d in query_digits if d in p_digits)
            total_score = token_matches + digit_matches

            if total_score > max_score:
                max_score = total_score
                best_match = p
                
        if max_score >= 2:
            matched_product = best_match

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
