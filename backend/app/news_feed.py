"""Aggregated market-wide news for the home page."""
from datetime import datetime, timezone
from typing import List, Dict
import feedparser

FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^IXIC&region=US&lang=en-US",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^DJI&region=US&lang=en-US",
]


def fetch_market_news(limit: int = 30) -> List[Dict]:
    items = []
    seen_titles = set()
    for url in FEEDS:
        try:
            feed = feedparser.parse(url)
        except Exception:
            continue
        for entry in feed.entries:
            title = entry.get("title")
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)
            try:
                ts = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc).timestamp()
            except Exception:
                ts = datetime.now(timezone.utc).timestamp()
            items.append({
                "title": title,
                "summary": entry.get("summary", ""),
                "url": entry.get("link"),
                "publisher": "Yahoo Finance",
                "ts": ts,
                "date": datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat(),
                "thumbnail": None,
            })
    items.sort(key=lambda x: x["ts"], reverse=True)
    return items[:limit]
