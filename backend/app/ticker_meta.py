"""In-memory ticker metadata — loaded once at startup from ticker_metadata.json."""
import json
from pathlib import Path

_META: dict = {}
_META_PATH = Path(__file__).resolve().parent.parent / "ticker_metadata.json"


def load():
    global _META
    if _META_PATH.exists():
        _META = json.loads(_META_PATH.read_text())


def search(q: str, limit: int = 15) -> list:
    """Search by ticker prefix or company name substring (case-insensitive)."""
    q = q.strip().upper()
    if not q or len(q) < 1:
        return []
    results = []
    # Exact ticker prefix first
    for ticker, info in _META.items():
        if ticker.startswith(q):
            results.append({"ticker": ticker, **info})
    # Then name substring matches not already included
    q_lower = q.lower()
    for ticker, info in _META.items():
        if ticker not in {r["ticker"] for r in results}:
            if q_lower in info.get("name", "").lower():
                results.append({"ticker": ticker, **info})
    return results[:limit]


def list_stocks(sector: str = "", page: int = 1, limit: int = 50) -> dict:
    items = [{"ticker": t, **info} for t, info in _META.items()]
    if sector:
        items = [i for i in items if i.get("sector", "").lower() == sector.lower()]
    items.sort(key=lambda x: x["ticker"])
    total = len(items)
    start = (page - 1) * limit
    return {"items": items[start:start + limit], "total": total, "page": page, "limit": limit}


def get_sectors() -> list:
    sectors = sorted({info.get("sector", "") for info in _META.values() if info.get("sector")})
    return sectors
