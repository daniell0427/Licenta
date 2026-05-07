from typing import Dict, List, Optional
import time
import json
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import or_
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from . import market, sentiment, model as ml, news_feed, auth, schemas, news_repo
from .db import Base, engine, get_db, SessionLocal
from .models import (
    User, WatchlistItem, Cache, Notification,
    NewsArticle, PriceBar, PredictionRecord, CorpEvent,
)


Base.metadata.create_all(bind=engine)


@asynccontextmanager
async def lifespan(app):
    # Lazy-import to avoid circular deps (background imports from main for _MEM/_put)
    from . import background
    task = asyncio.create_task(background.refresh_loop())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="SentiTrade", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Default TTLs per cache namespace (in seconds).
TTL_DEFAULT = 300
TTL_BY_PREFIX = {
    "quote": 60,         # quotes go stale fast
    "price": 300,        # 5 min for price series
    "news": 600,         # 10 min for news
    "predict": 1800,     # 30 min — predictions are expensive
    "home": 120,         # 2 min for home dashboard
    "notif_scan": 120,   # last-scan timestamp per user
    "events": 3600,      # 1 h — earnings/dividends/splits don't change minute-to-minute
}

# L1 in-memory cache (fast); L2 is the DB Cache table (persists across restarts).
_MEM: Dict[str, tuple] = {}


def _ttl_for(key: str) -> int:
    prefix = key.split(":", 1)[0]
    return TTL_BY_PREFIX.get(prefix, TTL_DEFAULT)


def _cached(key: str, db: Optional[Session] = None):
    """Two-tier read: memory first, then DB. Returns None if missing/stale."""
    ttl = _ttl_for(key)
    now = time.time()
    item = _MEM.get(key)
    if item and (now - item[0]) <= ttl:
        return item[1]
    if db is None:
        return None
    row = db.query(Cache).filter(Cache.key == key).first()
    if row is None:
        return None
    age = (datetime.utcnow() - row.updated_at).total_seconds()
    if age > ttl:
        return None
    val = json.loads(row.payload)
    _MEM[key] = (now, val)  # warm L1
    return val


def _put(key: str, val, db: Optional[Session] = None):
    _MEM[key] = (time.time(), val)
    if db is None:
        return
    payload = json.dumps(val, default=str)
    row = db.query(Cache).filter(Cache.key == key).first()
    if row:
        row.payload = payload
        row.updated_at = datetime.utcnow()
    else:
        db.add(Cache(key=key, payload=payload, updated_at=datetime.utcnow()))
    try:
        db.commit()
    except Exception:
        db.rollback()


@app.get("/health")
def health():
    return {"ok": True}


# ---------- Auth ----------

@app.post("/api/auth/register", response_model=schemas.TokenOut)
def register(payload: schemas.UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(
        or_(User.email == payload.email, User.username == payload.username)
    ).first()
    if existing:
        raise HTTPException(400, "Email or username already registered")
    user = User(
        email=payload.email,
        username=payload.username,
        password_hash=auth.hash_password(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    token = auth.create_access_token(user.id)
    return {"access_token": token, "token_type": "bearer", "user": user}


@app.post("/api/auth/login", response_model=schemas.TokenOut)
def login(payload: schemas.UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(
        or_(User.email == payload.email_or_username, User.username == payload.email_or_username)
    ).first()
    if not user or not auth.verify_password(payload.password, user.password_hash):
        raise HTTPException(401, "Invalid credentials")
    token = auth.create_access_token(user.id)
    return {"access_token": token, "token_type": "bearer", "user": user}


@app.get("/api/auth/me", response_model=schemas.UserOut)
def me(user: User = Depends(auth.get_current_user)):
    return user


# ---------- Per-user watchlist ----------

@app.get("/api/watchlist", response_model=schemas.WatchlistOut)
def get_watchlist(
    user: User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    items = (
        db.query(WatchlistItem)
        .filter(WatchlistItem.user_id == user.id)
        .order_by(WatchlistItem.position, WatchlistItem.id)
        .all()
    )
    if not items:
        return {"items": []}
    # One batched call instead of N — much faster than per-ticker fetch
    tickers = [it.ticker for it in items]
    cached_key = f"quote_batch:{','.join(sorted(tickers))}"
    quotes = _cached(cached_key, db)
    if quotes is None:
        quotes = market.batch_quotes(tickers)
        _put(cached_key, quotes, db)
    rows = []
    for it in items:
        q = quotes.get(it.ticker)
        if q is None:
            rows.append({"ticker": it.ticker, "price": None, "change_pct": None})
        else:
            rows.append(q)
    return {"items": rows}


@app.post("/api/watchlist", response_model=schemas.WatchlistOut)
def add_watchlist(
    payload: schemas.WatchlistAdd,
    user: User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    ticker = payload.ticker.strip().upper()
    existing = db.query(WatchlistItem).filter(
        WatchlistItem.user_id == user.id, WatchlistItem.ticker == ticker
    ).first()
    if not existing:
        max_pos = (
            db.query(WatchlistItem)
            .filter(WatchlistItem.user_id == user.id)
            .count()
        )
        db.add(WatchlistItem(user_id=user.id, ticker=ticker, position=max_pos))
        db.commit()
    return get_watchlist(user=user, db=db)


# ---------- Notifications ----------

def _serialize_notification(n: Notification) -> dict:
    return {
        "id": n.id,
        "type": n.type,
        "ticker": n.ticker,
        "title": n.title,
        "subtitle": n.subtitle,
        "severity": n.severity,
        "direction": n.direction,
        "label": n.label,
        "url": n.url,
        "created_at": n.created_at.isoformat() if n.created_at else None,
        "read": n.read_at is not None,
    }


def _scan_notifications(user: User, db: Session):
    """Look at the user's watchlist and INSERT new Notification rows for any
    detected price move (>=5%) or very-high-confidence (>=92%) news from last 48h.
    Idempotent via the `fingerprint` unique constraint."""
    last_scan_key = f"notif_scan:{user.id}"
    if _cached(last_scan_key, db) is not None:
        return  # we scanned within the TTL window already

    items = db.query(WatchlistItem).filter(WatchlistItem.user_id == user.id).all()
    tickers = [it.ticker for it in items]
    if not tickers:
        _put(last_scan_key, {"ts": time.time()}, db)
        return

    today = datetime.utcnow().strftime("%Y-%m-%d")

    # Price moves
    quotes = market.batch_quotes(tickers)
    for t in tickers:
        q = quotes.get(t)
        if not q:
            continue
        chg = q.get("change_pct") or 0.0
        if abs(chg) >= 5:
            fp = f"price:{t}:{today}:{round(chg, 1)}"
            try:
                db.add(Notification(
                    user_id=user.id,
                    type="price_move",
                    ticker=t,
                    title=f"{t} moved {'+' if chg >= 0 else ''}{chg:.2f}% today",
                    subtitle=f"Now ${q['price']:.2f} (prev close ${q.get('previous_close') or 0:.2f})",
                    severity="high" if abs(chg) >= 10 else "medium",
                    direction="up" if chg >= 0 else "down",
                    label="",
                    url="",
                    fingerprint=fp,
                ))
                db.commit()
            except Exception:
                db.rollback()

    # Very-high-confidence news
    cutoff = time.time() - 48 * 3600
    for t in tickers:
        try:
            news_items = market.fetch_news(t, limit=10)
        except Exception:
            continue
        recent = [n for n in news_items if (n.get("ts") or 0) >= cutoff]
        if not recent:
            continue
        scored = sentiment.score_texts(
            [f"{n['title']}. {n.get('summary','')}".strip() for n in recent]
        )
        for n, s in zip(recent, scored):
            if s["label"] in ("positive", "negative") and s["confidence"] >= 0.92:
                fp = f"news:{t}:{(n.get('title') or '')[:80]}"
                try:
                    db.add(Notification(
                        user_id=user.id,
                        type="news",
                        ticker=t,
                        title=n["title"],
                        subtitle=f"{n.get('publisher') or 'News'} · {s['label'].capitalize()} · {int(s['confidence']*100)}% confidence",
                        severity="high",
                        direction="",
                        label=s["label"],
                        url=n.get("url") or "",
                        fingerprint=fp,
                    ))
                    db.commit()
                except Exception:
                    db.rollback()

    _put(last_scan_key, {"ts": time.time()}, db)


@app.get("/api/notifications")
def list_notifications(
    user: User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    _scan_notifications(user, db)
    rows = (
        db.query(Notification)
        .filter(Notification.user_id == user.id, Notification.dismissed == False)  # noqa: E712
        .order_by(Notification.created_at.desc())
        .limit(50)
        .all()
    )
    items = [_serialize_notification(n) for n in rows]
    unread = sum(1 for n in rows if n.read_at is None)
    return {"items": items, "unread": unread, "count": len(items)}


@app.post("/api/notifications/{notif_id}/read")
def mark_notification_read(
    notif_id: int,
    user: User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    n = db.query(Notification).filter(
        Notification.id == notif_id, Notification.user_id == user.id
    ).first()
    if not n:
        raise HTTPException(404, "Not found")
    if n.read_at is None:
        n.read_at = datetime.utcnow()
        db.commit()
    return {"ok": True}


@app.post("/api/notifications/read-all")
def mark_all_read(
    user: User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    db.query(Notification).filter(
        Notification.user_id == user.id, Notification.read_at.is_(None)
    ).update({"read_at": datetime.utcnow()})
    db.commit()
    return {"ok": True}


@app.delete("/api/notifications/{notif_id}")
def delete_notification(
    notif_id: int,
    user: User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    n = db.query(Notification).filter(
        Notification.id == notif_id, Notification.user_id == user.id
    ).first()
    if not n:
        raise HTTPException(404, "Not found")
    n.dismissed = True
    db.commit()
    return {"ok": True}


@app.delete("/api/notifications")
def delete_all_notifications(
    user: User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    db.query(Notification).filter(
        Notification.user_id == user.id
    ).update({"dismissed": True})
    db.commit()
    return {"ok": True}


@app.delete("/api/watchlist/{ticker}", response_model=schemas.WatchlistOut)
def remove_watchlist(
    ticker: str,
    user: User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    db.query(WatchlistItem).filter(
        WatchlistItem.user_id == user.id, WatchlistItem.ticker == ticker.upper()
    ).delete()
    db.commit()
    return get_watchlist(user=user, db=db)


# ---------- Market data ----------

def _persist_price_bars(db: Session, ticker: str, interval: str, candles: list):
    """Upsert OHLCV bars into the PriceBar archive.
    Uses INSERT OR IGNORE on (ticker, interval, bar_time) — duplicates skipped."""
    if not candles:
        return
    rows = []
    for c in candles:
        rows.append({
            "ticker": ticker,
            "interval": interval,
            "bar_time": datetime.utcfromtimestamp(c["time"]),
            "open": c["open"], "high": c["high"], "low": c["low"],
            "close": c["close"], "volume": c["volume"],
            "fetched_at": datetime.utcnow(),
        })
    try:
        stmt = sqlite_insert(PriceBar).values(rows).on_conflict_do_nothing(
            index_elements=["ticker", "interval", "bar_time"]
        )
        db.execute(stmt)
        db.commit()
    except Exception:
        db.rollback()


@app.get("/api/price/{ticker}")
def price(ticker: str, period: str = "1y", interval: str = "1d",
          db: Session = Depends(get_db)):
    key = f"price:{ticker}:{period}:{interval}"
    if (c := _cached(key, db)) is not None:
        return c
    df = market.fetch_history(ticker, period=period, interval=interval)
    if df.empty:
        raise HTTPException(404, f"No price data for {ticker}")
    if interval == "1d":
        df = market.add_indicators(df)
    candles = market.to_chart_records(df)
    out = {
        "ticker": ticker.upper(),
        "period": period,
        "interval": interval,
        "candles": candles,
    }
    _put(key, out, db)
    # Archive bars for research (doesn't affect response)
    _persist_price_bars(db, ticker.upper(), interval, candles)
    return out


@app.get("/api/quote/{ticker}")
def quote(ticker: str):
    """Live quote (regular + pre/post-market). Not cached — meant for polling."""
    q = market.fetch_live_quote(ticker)
    if q is None:
        raise HTTPException(404, f"No quote for {ticker}")
    return q


@app.get("/api/news/{ticker}")
def news(
    ticker: str,
    limit: int = 50,
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db),
):
    """News are persisted in the DB. Strategy:
       - read existing rows immediately (instant after first fetch)
       - if no data yet OR data is stale (>10 min old), refresh
         - synchronously on first ever load (no rows)
         - in the background otherwise (user gets fresh-ish data fast)"""
    ticker = ticker.upper()
    rows = news_repo.get_news_from_db(db, ticker, limit)
    stale = news_repo.needs_refresh(rows)

    if not rows:
        # First-time fetch — must do it synchronously so the user sees something
        news_repo.refresh_news_for_ticker(ticker, db)
        rows = news_repo.get_news_from_db(db, ticker, limit)
    elif stale and background_tasks is not None:
        # Have data; refresh in the background so the next call is fresh
        background_tasks.add_task(news_repo.refresh_news_for_ticker, ticker, None)

    payload = news_repo.serialize_for_api(rows)
    return {"ticker": ticker, **payload}


@app.get("/api/predict/{ticker}")
def predict(ticker: str, db: Session = Depends(get_db)):
    key = f"predict:{ticker}"
    if (c := _cached(key, db)) is not None:
        return c
    df = market.fetch_history(ticker, period="3y")
    if df.empty:
        raise HTTPException(404, f"No price data for {ticker}")
    df = market.add_indicators(df)
    # Wider news pool for prediction context — 80 ticker headlines plus 30
    # market-wide headlines so the model is anchored beyond just today's news.
    news_items = market.fetch_news(ticker, limit=80)
    market_context = news_feed.fetch_market_news(limit=30)
    news_items = news_items + [
        m for m in market_context
        if m.get("title", "").strip().lower()
        not in {(n.get("title") or "").strip().lower() for n in news_items}
    ]
    sent_by_date: Dict[str, float] = {}
    if news_items:
        scored = sentiment.score_texts(
            [f"{it['title']}. {it['summary']}".strip() for it in news_items]
        )
        for it, s in zip(news_items, scored):
            it["signed"] = s["signed"]
        sent_by_date = sentiment.aggregate_daily(
            [{"date": it["date"], "signed": it["signed"]} for it in news_items]
        )
    short_term = ml.train_and_predict(df, sent_by_date, horizon=1)
    long_term = ml.train_and_predict(df, sent_by_date, horizon=5)
    out = {"ticker": ticker.upper(), "short_term": short_term, "long_term": long_term}
    _put(key, out, db)

    # Archive every prediction so we can later evaluate accuracy vs realized returns.
    last_close = float(df["close"].iloc[-1]) if not df.empty else 0.0
    avg_sent = (sum(sent_by_date.values()) / len(sent_by_date)) if sent_by_date else 0.0
    try:
        for horizon, p in (("1", short_term), ("5", long_term)):
            db.add(PredictionRecord(
                ticker=ticker.upper(),
                horizon_days=int(horizon),
                direction=p.get("direction", ""),
                confidence=float(p.get("confidence", 0.0)),
                train_acc=float(p.get("train_acc", 0.0)),
                val_acc=float(p.get("val_acc", 0.0)),
                sentiment_score=float(avg_sent),
                last_close=last_close,
                generated_at=datetime.utcnow(),
            ))
        db.commit()
    except Exception:
        db.rollback()

    return out


def _events_from_db(db: Session, ticker: str):
    """Return all CorpEvent rows for ticker, or None if the data is stale/missing."""
    rows = (
        db.query(CorpEvent)
        .filter(CorpEvent.ticker == ticker)
        .order_by(CorpEvent.ts.desc())
        .all()
    )
    if not rows:
        return None
    # Refresh if oldest fetch is more than 12 h ago
    oldest_fetch = min(r.fetched_at for r in rows if r.fetched_at)
    if (datetime.now(timezone.utc).replace(tzinfo=None) - oldest_fetch).total_seconds() > 43200:
        return None
    return rows


def _persist_events(db: Session, ticker: str, data: dict):
    """Upsert every event into corp_events."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    all_events = data.get("past", []) + data.get("upcoming", [])
    if not all_events:
        return
    rows = [
        {
            "ticker": ticker,
            "event_type": ev["type"],
            "event_date": ev["date"],
            "ts": ev["ts"],
            "label": ev.get("label", ""),
            "value": ev.get("value"),
            "eps_actual": ev.get("eps_actual"),
            "eps_estimate": ev.get("eps_estimate"),
            "fetched_at": now,
        }
        for ev in all_events
    ]
    try:
        stmt = sqlite_insert(CorpEvent).values(rows).on_conflict_do_update(
            index_elements=["ticker", "event_type", "event_date"],
            set_={"value": sqlite_insert(CorpEvent).excluded.value,
                  "eps_actual": sqlite_insert(CorpEvent).excluded.eps_actual,
                  "eps_estimate": sqlite_insert(CorpEvent).excluded.eps_estimate,
                  "fetched_at": sqlite_insert(CorpEvent).excluded.fetched_at},
        )
        db.execute(stmt)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[events] persist failed for {ticker}: {e}")


def _serialize_events(rows, ticker: str) -> dict:
    now_ts = datetime.now(timezone.utc).timestamp()
    past, upcoming = [], []
    for r in rows:
        ev = {
            "date": r.event_date,
            "ts": r.ts,
            "type": r.event_type,
            "label": r.label,
            "value": r.value,
            "eps_actual": r.eps_actual,
            "eps_estimate": r.eps_estimate,
        }
        (upcoming if r.ts > now_ts else past).append(ev)
    past.sort(key=lambda x: x["ts"], reverse=True)
    upcoming.sort(key=lambda x: x["ts"])
    return {"ticker": ticker, "past": past, "upcoming": upcoming}


@app.get("/api/events/{ticker}")
def events(ticker: str, db: Session = Depends(get_db)):
    ticker = ticker.upper()
    # Serve from DB if fresh enough
    rows = _events_from_db(db, ticker)
    if rows is not None:
        return _serialize_events(rows, ticker)
    # Live fetch → persist → serve
    try:
        data = market.fetch_events(ticker)
    except Exception as e:
        raise HTTPException(500, str(e))
    _persist_events(db, ticker, data)
    return {"ticker": ticker, **data}


# ---------- Home page ----------

# Tickers that populate the home dashboard, grouped by category.
TRENDING_TICKERS = [
    "AAPL", "MSFT", "NVDA", "TSLA", "GOOGL", "AMZN", "META", "AMD", "NFLX", "JPM",
    "AVGO", "ORCL", "CRM", "ADBE", "INTC", "DIS", "BAC", "WMT", "COST", "XOM",
]
INDICES = [
    ("^GSPC", "S&P 500"),
    ("^IXIC", "NASDAQ"),
    ("^DJI", "Dow Jones"),
    ("^RUT", "Russell 2000"),
    ("^VIX", "VIX"),
]
# SPDR sector ETFs — the standard sector decomposition
SECTORS = [
    ("XLK", "Technology"),
    ("XLF", "Financials"),
    ("XLV", "Health Care"),
    ("XLY", "Consumer Disc."),
    ("XLP", "Consumer Staples"),
    ("XLE", "Energy"),
    ("XLI", "Industrials"),
    ("XLU", "Utilities"),
    ("XLB", "Materials"),
    ("XLRE", "Real Estate"),
    ("XLC", "Comm. Services"),
]
COMMODITIES = [
    ("GC=F", "Gold"),
    ("SI=F", "Silver"),
    ("CL=F", "Crude Oil"),
    ("NG=F", "Natural Gas"),
]
def _enrich(quotes: Dict[str, Dict], pairs):
    """Map a list of (ticker, display_name) pairs to enriched quote dicts."""
    out = []
    for sym, name in pairs:
        q = quotes.get(sym)
        if not q:
            continue
        out.append({**q, "name": name})
    return out


@app.get("/api/home")
def home(db: Session = Depends(get_db)):
    key = "home"
    if (c := _cached(key, db)) is not None:
        return c

    # Single batch call for every ticker on the page
    all_syms = (
        TRENDING_TICKERS
        + [t for t, _ in INDICES]
        + [t for t, _ in SECTORS]
        + [t for t, _ in COMMODITIES]
    )
    quotes = market.batch_quotes(all_syms)

    indices = _enrich(quotes, INDICES)
    sectors = _enrich(quotes, SECTORS)
    commodities = _enrich(quotes, COMMODITIES)

    trending_quotes = [quotes[t] for t in TRENDING_TICKERS if t in quotes]
    # Top gainers + losers split
    sorted_by_change = sorted(trending_quotes, key=lambda x: x.get("change_pct", 0))
    losers = sorted_by_change[:5]
    gainers = list(reversed(sorted_by_change[-5:]))

    market_news = news_feed.fetch_market_news(limit=20)
    if market_news:
        scored = sentiment.score_texts(
            [f"{it['title']}. {it.get('summary','')}".strip() for it in market_news]
        )
        for it, s in zip(market_news, scored):
            it.update(s)

    out = {
        "indices": indices,
        "sectors": sectors,
        "commodities": commodities,
        "gainers": gainers,
        "losers": losers,
        "news": market_news,
    }
    _put(key, out, db)
    return out
