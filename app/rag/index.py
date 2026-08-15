import json
from pathlib import Path
from typing import List, Dict, Any

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

class DataIndex:
    _deals: List[Dict[str, Any]] = []
    _cards: List[Dict[str, Any]] = []
    _products: List[Dict[str, Any]] = []
    _loaded: bool = False

    @classmethod
    def load_all(cls):
        if cls._loaded:
            return

        deals_file = DATA_DIR / "deals.json"
        cards_file = DATA_DIR / "cards.json"
        products_file = DATA_DIR / "products.json"

        if deals_file.exists():
            with open(deals_file, "r", encoding="utf-8") as f:
                cls._deals = json.load(f)

        if cards_file.exists():
            with open(cards_file, "r", encoding="utf-8") as f:
                cls._cards = json.load(f)

        if products_file.exists():
            with open(products_file, "r", encoding="utf-8") as f:
                cls._products = json.load(f)

        cls._loaded = True

    @classmethod
    def get_deals(cls) -> List[Dict[str, Any]]:
        cls.load_all()
        return cls._deals

    @classmethod
    def get_cards(cls) -> List[Dict[str, Any]]:
        cls.load_all()
        return cls._cards

    @classmethod
    def get_products(cls) -> List[Dict[str, Any]]:
        cls.load_all()
        return cls._products

    @classmethod
    def get_card_by_id(cls, card_id: str) -> Dict[str, Any] | None:
        cls.load_all()
        for c in cls._cards:
            if c["card_id"].lower() == card_id.lower():
                return c
        return None

    @classmethod
    def get_deal_by_id(cls, deal_id: str) -> Dict[str, Any] | None:
        cls.load_all()
        for d in cls._deals:
            if d["deal_id"].lower() == deal_id.lower():
                return d
        return None
