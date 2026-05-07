from datetime import datetime, timezone
from typing import List, Dict, Optional
import io
import re
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands

try:
    from curl_cffi import requests as curl_requests
    _SESSION = curl_requests.Session(impersonate="chrome")
except Exception:
    _SESSION = None


def _ticker(sym: str):
    if _SESSION is not None:
        return yf.Ticker(sym, session=_SESSION)
    return yf.Ticker(sym)


_PERIOD_RANGE = {
    "1d": "1d", "5d": "5d", "30d": "1mo", "60d": "1mo",
    "1mo": "1mo", "3mo": "3mo", "6mo": "6mo",
    "1y": "1y", "2y": "2y", "5y": "5y", "10y": "10y", "ytd": "ytd", "max": "max",
}


def _fetch_history_yahoo_chart(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Hit Yahoo's chart API directly (what yfinance wraps) — avoids yfinance parser bugs."""
    if _SESSION is None:
        return pd.DataFrame()
    rng = _PERIOD_RANGE.get(period.lower(), period)
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"range": rng, "interval": interval, "includePrePost": "false", "events": "div,splits"}
    try:
        r = _SESSION.get(url, params=params, timeout=15)
        if r.status_code != 200:
            print(f"[yahoo-chart] {ticker}: HTTP {r.status_code}")
            return pd.DataFrame()
        data = r.json()
    except Exception as e:
        print(f"[yahoo-chart] {ticker}: {e}")
        return pd.DataFrame()
    try:
        result = (data.get("chart") or {}).get("result") or []
        if not result:
            err = (data.get("chart") or {}).get("error")
            print(f"[yahoo-chart] {ticker}: no result (error={err})")
            return pd.DataFrame()
        res = result[0]
        timestamps = res.get("timestamp") or []
        quote = ((res.get("indicators") or {}).get("quote") or [{}])[0]
        adjclose_arr = (((res.get("indicators") or {}).get("adjclose") or [{}])[0]).get("adjclose")
        if not timestamps or not quote:
            return pd.DataFrame()
        opens = quote.get("open") or []
        highs = quote.get("high") or []
        lows = quote.get("low") or []
        closes = quote.get("close") or []
        volumes = quote.get("volume") or []
        # Prefer adjusted close if present (mirrors auto_adjust=True behavior)
        if adjclose_arr and len(adjclose_arr) == len(closes):
            closes = adjclose_arr
        df = pd.DataFrame({
            "open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes,
        }, index=pd.to_datetime(timestamps, unit="s", utc=True))
        df = df.dropna(subset=["close"])
        # Dedup by index (Yahoo occasionally returns duplicate timestamps on weekly/monthly)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return df
    except Exception as e:
        print(f"[yahoo-chart] {ticker}: parse error: {e}")
        return pd.DataFrame()


def _fetch_history_yfinance(ticker: str, period: str, interval: str) -> pd.DataFrame:
    try:
        df = _ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
        if not df.empty:
            return df
    except Exception as e:
        print(f"[yfinance Ticker] {ticker}: {e}")
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True,
                         progress=False, threads=False)
        return df
    except Exception as e:
        print(f"[yfinance download] {ticker}: {e}")
        return pd.DataFrame()


def _period_to_days(period: str) -> int:
    period = period.lower().strip()
    if period.endswith("y"):
        return int(period[:-1]) * 365
    if period.endswith("mo"):
        return int(period[:-2]) * 30
    if period.endswith("d"):
        return int(period[:-1])
    if period == "max":
        return 365 * 10
    return 365


def _fetch_history_stooq(ticker: str, period: str) -> pd.DataFrame:
    """CSV from Stooq — reliable free fallback for US tickers."""
    import io
    import requests
    sym = ticker.lower()
    if "." not in sym:
        sym = f"{sym}.us"
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f"[stooq] {ticker}: request failed: {e}")
        return pd.DataFrame()
    text = r.text.strip()
    if not text or text.lower().startswith("<"):
        print(f"[stooq] {ticker}: no data (empty or HTML response)")
        return pd.DataFrame()
    # Stooq sometimes returns a plain-text error instead of CSV
    first_line = text.split("\n")[0].strip()
    if "Date" not in first_line or "Close" not in first_line:
        print(f"[stooq] {ticker}: unexpected response (not CSV): {first_line[:120]}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception as e:
        print(f"[stooq] {ticker}: CSV parse error: {e}")
        return pd.DataFrame()
    if df.empty or "Date" not in df.columns:
        print(f"[stooq] {ticker}: empty dataframe or missing 'Date' column")
        return pd.DataFrame()
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    df = df.set_index("Date").sort_index()
    days = _period_to_days(period)
    cutoff = df.index.max() - pd.Timedelta(days=days)
    df = df[df.index >= cutoff]
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    return df[["open", "high", "low", "close", "volume"]]


def fetch_history(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    # Try direct Yahoo chart API first (bypasses yfinance bugs)
    df = _fetch_history_yahoo_chart(ticker, period=period, interval=interval)
    if not df.empty:
        return df[["open", "high", "low", "close", "volume"]]
    # Fall back to yfinance library
    df = _fetch_history_yfinance(ticker, period=period, interval=interval)
    if df.empty:
        # Stooq is daily-only; only use as fallback for daily intervals
        if interval == "1d":
            return _fetch_history_stooq(ticker, period=period)
        return df
    df.index = df.index.tz_convert("UTC") if df.index.tz else df.index.tz_localize("UTC")
    df = df.rename(columns=str.lower)
    return df[["open", "high", "low", "close", "volume"]]


def fetch_live_quote(ticker: str) -> Optional[Dict]:
    """Live quote with pre/post-market price from Yahoo chart meta."""
    if _SESSION is None:
        return None
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"range": "1d", "interval": "1m", "includePrePost": "true"}
    try:
        r = _SESSION.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
    except Exception as e:
        print(f"[quote] {ticker}: {e}")
        return None
    result = (data.get("chart") or {}).get("result") or []
    if not result:
        return None
    meta = result[0].get("meta") or {}
    regular = meta.get("regularMarketPrice")
    prev_close = meta.get("chartPreviousClose") or meta.get("previousClose")
    pre = meta.get("preMarketPrice")
    post = meta.get("postMarketPrice")
    state = meta.get("marketState")  # PRE, REGULAR, POST, POSTPOST, CLOSED
    if regular is None:
        return None
    # Pick the "current" price based on market state
    if state == "PRE" and pre is not None:
        current, session = float(pre), "pre"
    elif state in ("POST", "POSTPOST", "CLOSED") and post is not None:
        current, session = float(post), "post"
    else:
        current, session = float(regular), "regular"
    change = (current - prev_close) if prev_close else 0.0
    change_pct = (change / prev_close * 100) if prev_close else 0.0
    return {
        "ticker": ticker.upper(),
        "price": current,
        "regular_price": float(regular),
        "previous_close": float(prev_close) if prev_close else None,
        "change": float(change),
        "change_pct": float(change_pct),
        "market_state": state,
        "session": session,
        "pre_market": float(pre) if pre is not None else None,
        "post_market": float(post) if post is not None else None,
        "currency": meta.get("currency"),
        "ts": meta.get("regularMarketTime"),
    }


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["rsi_14"] = RSIIndicator(close=out["close"], window=14).rsi()
    macd = MACD(close=out["close"])
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["ema_12"] = EMAIndicator(close=out["close"], window=12).ema_indicator()
    out["ema_26"] = EMAIndicator(close=out["close"], window=26).ema_indicator()
    bb = BollingerBands(close=out["close"], window=20)
    out["bb_high"] = bb.bollinger_hband()
    out["bb_low"] = bb.bollinger_lband()
    out["ret_1d"] = out["close"].pct_change()
    return out


def _norm_news_item(n: dict) -> Optional[dict]:
    content = n.get("content") if isinstance(n, dict) else None
    if content:
        title = content.get("title")
        summary = content.get("summary") or ""
        pub = content.get("pubDate") or content.get("displayTime")
        link = (content.get("canonicalUrl") or {}).get("url") or (content.get("clickThroughUrl") or {}).get("url")
        publisher = (content.get("provider") or {}).get("displayName")
        thumb = ((content.get("thumbnail") or {}).get("resolutions") or [{}])[0].get("url")
    else:
        title = n.get("title")
        summary = n.get("summary", "")
        pub = n.get("providerPublishTime")
        link = n.get("link")
        publisher = n.get("publisher")
        thumb = ((n.get("thumbnail") or {}).get("resolutions") or [{}])[0].get("url") if n.get("thumbnail") else None
    if not title:
        return None
    if isinstance(pub, (int, float)):
        date_str = datetime.fromtimestamp(pub, tz=timezone.utc).date().isoformat()
        ts = float(pub)
    elif isinstance(pub, str):
        try:
            d = datetime.fromisoformat(pub.replace("Z", "+00:00"))
            date_str = d.date().isoformat()
            ts = d.timestamp()
        except Exception:
            date_str = datetime.now(timezone.utc).date().isoformat()
            ts = datetime.now(timezone.utc).timestamp()
    else:
        date_str = datetime.now(timezone.utc).date().isoformat()
        ts = datetime.now(timezone.utc).timestamp()
    return {
        "title": title,
        "summary": summary,
        "publisher": publisher,
        "url": link,
        "date": date_str,
        "ts": ts,
        "thumbnail": thumb,
    }


def _fetch_news_yfinance(ticker: str, limit: int) -> List[Dict]:
    try:
        raw = _ticker(ticker).news or []
    except Exception:
        raw = []
    items = []
    for n in raw[:limit]:
        norm = _norm_news_item(n)
        if norm:
            items.append(norm)
    return items


def _fetch_news_rss(ticker: str, limit: int) -> List[Dict]:
    """Yahoo Finance RSS by ticker."""
    import feedparser
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    try:
        feed = feedparser.parse(url)
    except Exception:
        return []
    items = []
    for entry in feed.entries[:limit]:
        try:
            ts = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc).timestamp()
            date_str = datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()
        except Exception:
            ts = datetime.now(timezone.utc).timestamp()
            date_str = datetime.now(timezone.utc).date().isoformat()
        items.append({
            "title": entry.get("title"),
            "summary": entry.get("summary", ""),
            "publisher": "Yahoo Finance",
            "url": entry.get("link"),
            "date": date_str,
            "ts": ts,
            "thumbnail": None,
        })
    return items


def _fetch_news_google(ticker: str, limit: int) -> List[Dict]:
    """Google News RSS — wider coverage and longer history for context."""
    import feedparser
    from urllib.parse import quote_plus
    query = quote_plus(f"{ticker} stock")
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    try:
        feed = feedparser.parse(url)
    except Exception:
        return []
    items = []
    for entry in feed.entries[:limit]:
        try:
            ts = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc).timestamp()
            date_str = datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()
        except Exception:
            ts = datetime.now(timezone.utc).timestamp()
            date_str = datetime.now(timezone.utc).date().isoformat()
        publisher = ""
        try:
            publisher = (entry.get("source") or {}).get("title") or "Google News"
        except Exception:
            publisher = "Google News"
        items.append({
            "title": entry.get("title"),
            "summary": entry.get("summary", ""),
            "publisher": publisher,
            "url": entry.get("link"),
            "date": date_str,
            "ts": ts,
            "thumbnail": None,
        })
    return items


_ALIAS_CACHE: Dict[str, List[str]] = {}

# Common stop-words that shouldn't count as a "company name match" on their own
_STOP_TOKENS = {
    "inc", "inc.", "corp", "corp.", "corporation", "co", "co.", "company", "ltd",
    "ltd.", "plc", "the", "&", "and", "holdings", "group", "international",
    "industries", "technologies", "technology", "systems", "global",
}


def _company_aliases(ticker: str) -> List[str]:
    """Return a list of strings (lowercased) that signal an article is about this ticker.
    Always includes the ticker symbol itself; tries to add the company name."""
    if ticker in _ALIAS_CACHE:
        return _ALIAS_CACHE[ticker]
    aliases = {ticker.lower()}
    # Pull shortName/longName from Yahoo quoteSummary
    if _SESSION is not None:
        try:
            url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
            r = _SESSION.get(url, params={"modules": "quoteType,price"}, timeout=8)
            if r.status_code == 200:
                data = r.json()
                result = ((data.get("quoteSummary") or {}).get("result") or [])
                if result:
                    qt = result[0].get("quoteType") or {}
                    pr = result[0].get("price") or {}
                    for raw in [qt.get("shortName"), qt.get("longName"),
                                pr.get("shortName"), pr.get("longName")]:
                        if not raw:
                            continue
                        # Strip corporate suffixes; first 1-2 meaningful tokens are the brand
                        tokens = [t for t in raw.replace(",", " ").split()
                                  if t.lower() not in _STOP_TOKENS]
                        if tokens:
                            # First token alone is usually the brand ("Apple", "Microsoft")
                            aliases.add(tokens[0].lower())
                            # Two-token form covers things like "Bank of America"
                            if len(tokens) >= 2:
                                aliases.add(" ".join(tokens[:2]).lower())
        except Exception as e:
            print(f"[aliases] {ticker}: {e}")
    out = sorted(aliases, key=len, reverse=True)
    _ALIAS_CACHE[ticker] = out
    return out


def _matches_alias(text: str, aliases: List[str]) -> bool:
    import re
    if not text:
        return False
    t = text.lower()
    for a in aliases:
        if len(a) <= 5 and a.isalnum():
            if re.search(rf"\b{re.escape(a)}\b", t):
                return True
        else:
            if a in t:
                return True
    return False


def _is_relevant(title: str, summary: str, aliases: List[str]) -> bool:
    """An article is relevant if the ticker/company name appears in:
      - the title (strongest signal), OR
      - the lede — first ~200 chars of the summary (real news fronts the subject in
        the opening sentence; passing mentions live deeper in the body).
    """
    if _matches_alias(title, aliases):
        return True
    lede = (summary or "")[:200]
    return _matches_alias(lede, aliases)


def fetch_news(ticker: str, limit: int = 50) -> List[Dict]:
    """Aggregate ticker news from yfinance + Yahoo RSS + Google News, dedup by title,
    then drop items whose title doesn't reference the ticker or company name."""
    items: List[Dict] = []
    items.extend(_fetch_news_yfinance(ticker, limit))
    items.extend(_fetch_news_rss(ticker, limit))
    items.extend(_fetch_news_google(ticker, limit))
    seen = set()
    deduped = []
    for it in items:
        title = (it.get("title") or "").strip()
        key = title.lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(it)
    aliases = _company_aliases(ticker)
    relevant = [
        it for it in deduped
        if _is_relevant(it.get("title", ""), it.get("summary", ""), aliases)
    ]
    relevant.sort(key=lambda x: x.get("ts", 0), reverse=True)
    return relevant[:limit]


def to_chart_records(df: pd.DataFrame) -> List[Dict]:
    if df.empty:
        return []
    # Defensive dedup + sort so lightweight-charts never sees duplicate / unsorted times
    df = df[~df.index.duplicated(keep="last")].sort_index()
    out = []
    seen = set()
    for ts, row in df.iterrows():
        t = int(ts.timestamp())
        if t in seen:
            continue
        seen.add(t)
        out.append({
            "date": ts.date().isoformat(),
            "time": t,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        })
    return out


def _single_quote_via_chart(ticker: str) -> Optional[Dict]:
    """Pull just price + previous close from the chart endpoint's `meta` block —
    same endpoint that powers the watchlist, so we know it works."""
    if _SESSION is None:
        return None
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    try:
        r = _SESSION.get(url, params={"range": "5d", "interval": "1d"}, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
    except Exception:
        return None
    result = (data.get("chart") or {}).get("result") or []
    if not result:
        return None
    meta = result[0].get("meta") or {}
    price = meta.get("regularMarketPrice")
    prev = meta.get("chartPreviousClose") or meta.get("previousClose")
    if price is None:
        return None
    change = (price - prev) if prev else 0.0
    change_pct = (change / prev * 100) if prev else 0.0
    return {
        "ticker": ticker,
        "price": float(price),
        "previous_close": float(prev) if prev is not None else None,
        "change": float(change),
        "change_pct": float(change_pct),
        "currency": meta.get("currency"),
    }


def batch_quotes(tickers: List[str]) -> Dict[str, Dict]:
    """Fast multi-ticker snapshot via parallel chart-endpoint calls.
    Yahoo's batch quote endpoint now requires a crumb/auth, so we parallelize
    the per-ticker chart endpoint instead — same approach the watchlist uses."""
    if not tickers:
        return {}
    from concurrent.futures import ThreadPoolExecutor
    out: Dict[str, Dict] = {}
    with ThreadPoolExecutor(max_workers=12) as ex:
        results = list(ex.map(_single_quote_via_chart, tickers))
    for t, q in zip(tickers, results):
        if q is not None:
            out[t] = q
    return out


def _parse_yahoo_earnings_date(s: str) -> Optional[datetime]:
    """Yahoo earnings calendar date strings: 'Jan 28, 2025, 5 PMEST',
    'Jan 28, 2025, Time Not Supplied', or 'Jan 28, 2025'."""
    if not isinstance(s, str) or not s.strip():
        return None
    cleaned = re.sub(r",\s*\d+(?::\d+)?\s*[AP]M[A-Z]*\s*$", "", s).strip()
    cleaned = re.sub(r",\s*Time\s*Not\s*Supplied\s*$", "", cleaned, flags=re.I).strip()
    for fmt in ("%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    return None


def _fetch_earnings_yahoo_html(ticker: str, max_results: int = 200) -> List[Dict]:
    """Fetch earnings from yfinance (works without curl_cffi session issues)."""
    sym = ticker.upper()
    out: List[Dict] = []
    try:
        t = _ticker(sym)
        earnings_df = t.earnings_dates
        if earnings_df is None or earnings_df.empty:
            return []
        for idx, row in earnings_df.iterrows():
            if len(out) >= max_results:
                break
            try:
                eps_estimate = row.get("EPS Estimate")
                eps_actual = row.get("Reported EPS")
                eps_estimate = float(eps_estimate) if pd.notna(eps_estimate) else None
                eps_actual = float(eps_actual) if pd.notna(eps_actual) else None
                ts = idx.timestamp()
                out.append({
                    "ts": ts,
                    "date": idx.date().isoformat(),
                    "eps_actual": eps_actual,
                    "eps_estimate": eps_estimate,
                })
            except Exception:
                pass
    except Exception as e:
        print(f"[events] earnings fetch {sym}: {e}")
    return out


def fetch_events(ticker: str) -> Dict:
    """Fetch earnings, earnings calls, dividends, and stock splits via direct
    Yahoo Finance APIs. Returns {"past": [...], "upcoming": [...]}."""
    sym = ticker.upper()
    now = datetime.now(timezone.utc)
    past: List[Dict] = []
    upcoming: List[Dict] = []

    if _SESSION is None:
        return {"past": past, "upcoming": upcoming}

    # ── Earnings: full historical + recent via HTML scrape ──────────────────────
    seen_earn_dates = set()
    try:
        for item in _fetch_earnings_yahoo_html(sym, max_results=200):
            if item["date"] in seen_earn_dates:
                continue
            seen_earn_dates.add(item["date"])
            dt = datetime.fromtimestamp(item["ts"], tz=timezone.utc)
            ev = {
                "date": item["date"],
                "ts": item["ts"],
                "type": "earnings",
                "label": "Earnings",
                "eps_actual": item.get("eps_actual"),
                "eps_estimate": item.get("eps_estimate"),
            }
            (upcoming if dt > now else past).append(ev)
    except Exception as e:
        print(f"[events] earnings scrape {sym}: {e}")

    # ── Calendar events: upcoming earnings + earnings call ──────────────────────
    try:
        url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{sym}"
        r = _SESSION.get(url, params={"modules": "calendarEvents"}, timeout=12)
        if r.status_code == 200:
            result = ((r.json().get("quoteSummary") or {}).get("result") or [])
            if result:
                cal_earn = ((result[0].get("calendarEvents") or {}).get("earnings") or {})
                # Upcoming earnings dates (in case scrape missed them)
                for ed in (cal_earn.get("earningsDate") or []):
                    ts = ed.get("raw")
                    if not ts:
                        continue
                    dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
                    if dt.date().isoformat() in seen_earn_dates:
                        continue
                    seen_earn_dates.add(dt.date().isoformat())
                    ev = {
                        "date": dt.date().isoformat(),
                        "ts": float(ts),
                        "type": "earnings",
                        "label": "Earnings",
                        "eps_actual": None,
                        "eps_estimate": None,
                    }
                    (upcoming if dt > now else past).append(ev)
                # Earnings call date — distinct event if different from earnings date
                for ed in (cal_earn.get("earningsCallDate") or []):
                    ts = ed.get("raw")
                    if not ts:
                        continue
                    dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
                    ev = {
                        "date": dt.date().isoformat(),
                        "ts": float(ts),
                        "type": "earnings_call",
                        "label": "Earnings Call",
                        "eps_actual": None,
                        "eps_estimate": None,
                    }
                    (upcoming if dt > now else past).append(ev)
    except Exception as e:
        print(f"[events] calendarEvents {sym}: {e}")

    # ── Dividends + splits: chart API with events=div,splits ────────────────────
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
        r = _SESSION.get(url, params={"range": "max", "interval": "1mo",
                                       "events": "div,splits"}, timeout=15)
        if r.status_code == 200:
            result = ((r.json().get("chart") or {}).get("result") or [])
            if result:
                ev_data = result[0].get("events") or {}

                for div in (ev_data.get("dividends") or {}).values():
                    try:
                        ts = float(div.get("date", 0))
                        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                        past.append({
                            "date": dt.date().isoformat(),
                            "ts": ts,
                            "type": "dividend",
                            "label": "Dividend",
                            "value": round(float(div.get("amount", 0)), 4),
                            "eps_actual": None,
                            "eps_estimate": None,
                        })
                    except Exception:
                        pass

                for split in (ev_data.get("splits") or {}).values():
                    try:
                        ts = float(split.get("date", 0))
                        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                        num = float(split.get("numerator", 1))
                        den = float(split.get("denominator", 1))
                        past.append({
                            "date": dt.date().isoformat(),
                            "ts": ts,
                            "type": "split",
                            "label": "Stock Split",
                            "value": round(num / den, 4) if den else 1.0,
                            "eps_actual": None,
                            "eps_estimate": None,
                        })
                    except Exception:
                        pass
    except Exception as e:
        print(f"[events] chart events {sym}: {e}")

    past.sort(key=lambda x: x["ts"], reverse=True)
    upcoming.sort(key=lambda x: x["ts"])
    return {"past": past, "upcoming": upcoming}


def quick_quote(ticker: str) -> Optional[Dict]:
    """Fast 2-day price snapshot for watchlist rows."""
    df = _fetch_history_yahoo_chart(ticker, period="5d", interval="1d")
    if df.empty:
        df = _fetch_history_yfinance(ticker, period="5d", interval="1d")
    if df.empty:
        df = _fetch_history_stooq(ticker, period="30d")
    if df.empty or len(df) < 2:
        return None
    close_col = "Close" if "Close" in df.columns else "close"
    last = float(df[close_col].iloc[-1])
    prev = float(df[close_col].iloc[-2])
    change_pct = (last - prev) / prev * 100 if prev else 0.0
    return {"ticker": ticker.upper(), "price": last, "change_pct": change_pct}
