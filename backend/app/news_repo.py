"""News persistence layer — read from DB, refresh in background, score-once."""
import hashlib
from datetime import datetime, timezone, timedelta
from typing import List, Dict

from sqlalchemy.orm import Session

from . import market, sentiment
from .db import SessionLocal
from .models import NewsArticle


# After this much time we consider the local DB stale and trigger a refresh.
REFRESH_AFTER_SECONDS = 180  # 3 minutes


def _fingerprint(ticker: str, title: str) -> str:
    raw = f"{ticker.upper()}::{(title or '').strip().lower()}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _to_dict(row: NewsArticle) -> dict:
    pub = row.published_at
    ts = pub.replace(tzinfo=timezone.utc).timestamp() if pub else 0.0
    return {
        "title": row.title,
        "summary": row.summary,
        "publisher": row.publisher,
        "url": row.url,
        "thumbnail": row.thumbnail or None,
        "ts": ts,
        "date": pub.date().isoformat() if pub else "",
        "label": row.sentiment_label or "neutral",
        "confidence": row.sentiment_confidence,
        "signed": row.sentiment_signed,
    }


def get_news_from_db(db: Session, ticker: str, limit: int) -> List[NewsArticle]:
    return (
        db.query(NewsArticle)
        .filter(NewsArticle.ticker == ticker.upper())
        .order_by(NewsArticle.published_at.desc().nullslast(),
                  NewsArticle.fetched_at.desc())
        .limit(limit)
        .all()
    )


def needs_refresh(rows: List[NewsArticle]) -> bool:
    if not rows:
        return True
    last_fetch = max((r.fetched_at for r in rows if r.fetched_at), default=None)
    if last_fetch is None:
        return True
    age = (datetime.utcnow() - last_fetch).total_seconds()
    return age > REFRESH_AFTER_SECONDS


def refresh_news_for_ticker(ticker: str, db: Session = None) -> int:
    """Fetch fresh news from sources, dedup against DB, score only NEW articles
    with FinBERT, persist. Returns number of new articles inserted."""
    own_session = db is None
    if own_session:
        db = SessionLocal()
    inserted = 0
    try:
        ticker = ticker.upper()
        try:
            items = market.fetch_news(ticker, limit=80)
        except Exception as e:
            print(f"[news_repo] fetch_news failed for {ticker}: {e}")
            items = []
        if not items:
            return 0

        existing_fps = {
            r[0] for r in db.query(NewsArticle.fingerprint)
            .filter(NewsArticle.ticker == ticker).all()
        }

        new_items = []
        for it in items:
            fp = _fingerprint(ticker, it.get("title") or "")
            if fp in existing_fps or not it.get("title"):
                continue
            existing_fps.add(fp)
            new_items.append((fp, it))

        if not new_items:
            return 0

        # Score only the new ones — this is the expensive step
        texts = [
            f"{it['title']}. {it.get('summary','') or ''}".strip()
            for _, it in new_items
        ]
        try:
            scored = sentiment.score_texts(texts)
        except Exception as e:
            print(f"[news_repo] sentiment scoring failed for {ticker}: {e}")
            scored = [{"label": "neutral", "confidence": 0.0, "signed": 0.0}] * len(new_items)

        for (fp, it), s in zip(new_items, scored):
            ts = it.get("ts") or 0
            try:
                pub = datetime.fromtimestamp(float(ts), tz=timezone.utc).replace(tzinfo=None) if ts else None
            except Exception:
                pub = None
            db.add(NewsArticle(
                ticker=ticker,
                title=it.get("title") or "",
                summary=it.get("summary") or "",
                publisher=it.get("publisher") or "",
                url=it.get("url") or "",
                thumbnail=it.get("thumbnail") or "",
                published_at=pub,
                fetched_at=datetime.utcnow(),
                sentiment_label=s.get("label", "neutral"),
                sentiment_confidence=float(s.get("confidence", 0.0)),
                sentiment_signed=float(s.get("signed", 0.0)),
                fingerprint=fp,
            ))
            inserted += 1
        try:
            db.commit()
        except Exception as e:
            db.rollback()
            print(f"[news_repo] commit failed for {ticker}: {e}")
            inserted = 0
    finally:
        if own_session:
            db.close()
    return inserted


def serialize_for_api(rows: List[NewsArticle]) -> Dict:
    items = [_to_dict(r) for r in rows]
    if rows:
        avg = sum((r.sentiment_signed or 0.0) for r in rows) / len(rows)
    else:
        avg = 0.0
    return {"items": items, "sentiment_score": float(avg)}
