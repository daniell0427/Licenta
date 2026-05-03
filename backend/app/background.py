"""Background workers — polled refresh of news + predictions for watchlist tickers."""
import asyncio
import time
from datetime import datetime
from typing import Set, Dict

from . import news_repo, market, sentiment, model as ml, news_feed
from .db import SessionLocal
from .models import WatchlistItem, Cache, NewsArticle, PredictionRecord


# Tunables ------------------------------------------------------------------
POLL_INTERVAL_SECONDS = 60          # check for new news every 1 min
PREDICTION_RECOMPUTE_COOLDOWN = 900  # don't recompute the same ticker more than once / 15 min
MACRO_CONFIDENCE_THRESHOLD = 0.85    # FinBERT confidence required to call a macro headline impactful

# Keywords that suggest market-wide / political news that can move many stocks at once
MACRO_KEYWORDS = [
    "fed", "federal reserve", "fomc", "powell",
    "inflation", "cpi", "ppi",
    "interest rate", "rate cut", "rate hike", "rate decision",
    "tariff", "trade war", "sanctions",
    "recession", "gdp", "unemployment", "jobs report", "nonfarm",
    "election", "white house", "treasury",
    "opec", "oil prices",
    "war", "geopolitical",
]

# Track when we last recomputed predictions per-ticker to avoid burning CPU
_last_predict_recompute: Dict[str, float] = {}
_seen_macro_fingerprints: Set[str] = set()


def _invalidate_prediction_cache(ticker: str):
    """Drop the prediction cache so the next /api/predict call recomputes,
    or so the background recompute writes fresh data through."""
    from .main import _MEM
    key = f"predict:{ticker.upper()}"
    _MEM.pop(key, None)
    try:
        with SessionLocal() as db:
            db.query(Cache).filter(Cache.key == key).delete()
            db.commit()
    except Exception:
        pass


def _recompute_prediction(ticker: str) -> bool:
    """Run the LSTM + sentiment fusion and persist the result. Returns True on success."""
    last = _last_predict_recompute.get(ticker, 0)
    if time.time() - last < PREDICTION_RECOMPUTE_COOLDOWN:
        return False
    _last_predict_recompute[ticker] = time.time()

    try:
        df = market.fetch_history(ticker, period="3y")
        if df.empty:
            return False
        df = market.add_indicators(df)
    except Exception as e:
        print(f"[bg-predict] history failed for {ticker}: {e}")
        return False

    # Pull pre-scored news from DB (fast — already computed)
    sent_by_date: Dict[str, float] = {}
    try:
        with SessionLocal() as db:
            rows = (
                db.query(NewsArticle)
                .filter(NewsArticle.ticker == ticker.upper())
                .order_by(NewsArticle.published_at.desc())
                .limit(120)
                .all()
            )
            buckets: Dict[str, list] = {}
            for r in rows:
                if not r.published_at:
                    continue
                d = r.published_at.date().isoformat()
                buckets.setdefault(d, []).append(r.sentiment_signed or 0.0)
            sent_by_date = {d: sum(v) / len(v) for d, v in buckets.items() if v}
    except Exception as e:
        print(f"[bg-predict] reading news failed for {ticker}: {e}")

    try:
        short_term = ml.train_and_predict(df, sent_by_date, horizon=1)
        long_term = ml.train_and_predict(df, sent_by_date, horizon=5)
    except Exception as e:
        print(f"[bg-predict] training failed for {ticker}: {e}")
        return False

    out = {"ticker": ticker.upper(), "short_term": short_term, "long_term": long_term}

    # Write through the cache + log for research
    from .main import _put
    try:
        with SessionLocal() as db:
            _put(f"predict:{ticker.upper()}", out, db)
            avg_sent = (sum(sent_by_date.values()) / len(sent_by_date)) if sent_by_date else 0.0
            last_close = float(df["close"].iloc[-1]) if not df.empty else 0.0
            for h, p in (("1", short_term), ("5", long_term)):
                db.add(PredictionRecord(
                    ticker=ticker.upper(),
                    horizon_days=int(h),
                    direction=p.get("direction", ""),
                    confidence=float(p.get("confidence", 0.0)),
                    train_acc=float(p.get("train_acc", 0.0)),
                    val_acc=float(p.get("val_acc", 0.0)),
                    sentiment_score=float(avg_sent),
                    last_close=last_close,
                    generated_at=datetime.utcnow(),
                ))
            db.commit()
    except Exception as e:
        print(f"[bg-predict] persist failed for {ticker}: {e}")

    print(f"[bg-predict] {ticker}: short={short_term.get('direction')} long={long_term.get('direction')}")
    return True


def _detect_macro_news() -> bool:
    """Pull market-wide news, score it, return True if any new high-confidence
    macro headline appeared since last poll."""
    try:
        items = news_feed.fetch_market_news(limit=20)
    except Exception:
        return False
    new_macro = []
    for it in items:
        title = (it.get("title") or "").lower()
        fp = (it.get("title") or "")[:140]
        if fp in _seen_macro_fingerprints:
            continue
        _seen_macro_fingerprints.add(fp)
        if any(k in title for k in MACRO_KEYWORDS):
            new_macro.append(it)
    if not new_macro:
        return False

    try:
        scored = sentiment.score_texts(
            [f"{it['title']}. {it.get('summary','')}".strip() for it in new_macro]
        )
    except Exception:
        return False
    for it, s in zip(new_macro, scored):
        if s["label"] in ("positive", "negative") and s["confidence"] >= MACRO_CONFIDENCE_THRESHOLD:
            print(f"[bg-macro] {s['label']} ({int(s['confidence']*100)}%): {it['title'][:90]}")
            return True
    return False


def _watchlist_tickers() -> Set[str]:
    try:
        with SessionLocal() as db:
            return {wl.ticker.upper() for wl in db.query(WatchlistItem).all()}
    except Exception:
        return set()


async def refresh_loop():
    """Main background loop. Runs every POLL_INTERVAL_SECONDS forever."""
    loop = asyncio.get_event_loop()
    print(f"[bg] refresh loop started — polling every {POLL_INTERVAL_SECONDS}s")
    # Wait a bit before first scan so startup is responsive
    await asyncio.sleep(15)
    while True:
        try:
            tickers = _watchlist_tickers()
            if not tickers:
                await asyncio.sleep(POLL_INTERVAL_SECONDS)
                continue

            macro_hit = await loop.run_in_executor(None, _detect_macro_news)
            tickers_to_recompute: Set[str] = set()

            for t in sorted(tickers):
                try:
                    inserted = await loop.run_in_executor(None, news_repo.refresh_news_for_ticker, t, None)
                except Exception as e:
                    print(f"[bg-news] {t}: {e}")
                    inserted = 0
                if inserted > 0:
                    print(f"[bg-news] {t}: +{inserted} new articles")
                    _invalidate_prediction_cache(t)
                    tickers_to_recompute.add(t)

            # Macro event: invalidate every watchlist ticker
            if macro_hit:
                for t in tickers:
                    _invalidate_prediction_cache(t)
                tickers_to_recompute.update(tickers)

            # Recompute predictions one at a time (LSTM training is heavy)
            for t in sorted(tickers_to_recompute):
                await loop.run_in_executor(None, _recompute_prediction, t)
                # Yield to other coroutines between heavy tasks
                await asyncio.sleep(0)

        except Exception as e:
            print(f"[bg-loop] error: {e}")

        await asyncio.sleep(POLL_INTERVAL_SECONDS)
