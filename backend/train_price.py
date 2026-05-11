"""Train the FusionLSTM on a multi-ticker panel with walk-forward validation.

Usage:
    python train_price.py --tickers SP500 --years 15 --horizon 5 --out artifacts/

Produces artifacts/price_model.pt and artifacts/price_scaler.pkl, which
backend/app/model.py loads at inference time.

Design choices (deliberate):
- Multi-ticker pooling — a single model sees ~500 tickers so it learns generic
  patterns rather than overfitting one symbol's idiosyncrasies.
- Walk-forward split — chronological 70/15/15 train/val/test. Random splits leak
  future info via feature autocorrelation and inflate accuracy by 5-10pp.
- Cost-aware label — predict sign of next-period return AFTER subtracting a
  per-trade cost. Otherwise the model learns to chase tiny moves that vanish
  after fees.
- Early stopping on val accuracy with patience=5.
- Saves only the StandardScaler + state_dict; the model architecture lives
  in app/model.py so train and serve stay in sync.
"""
from __future__ import annotations
import argparse
import gc
import os
import pickle
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).parent))
from app import market
from app.model import FusionLSTM, FEATURES


# A small but diverse universe. Replace --tickers SP500 with a CSV file path
# (one ticker per line) to use broader universes like S&P 1500.
DEFAULT_UNIVERSE = [
    # Mega-cap tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO",
    # Other tech / semis
    "AMD", "INTC", "ORCL", "CRM", "ADBE", "CSCO", "QCOM", "TXN",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "AXP",
    # Healthcare
    "JNJ", "UNH", "PFE", "MRK", "ABBV", "LLY", "TMO", "ABT",
    # Consumer
    "WMT", "COST", "PG", "KO", "PEP", "MCD", "NKE", "DIS",
    # Industrials / energy
    "CAT", "BA", "GE", "HON", "XOM", "CVX", "COP",
    # Indices
    "^GSPC", "^IXIC", "^DJI",
]


def _resolve_wiki_column(tbl: pd.DataFrame, *candidates: str) -> Optional[str]:
    """Return the first matching column name from candidates, case-insensitive."""
    if not candidates:
        return None
    lookup = {str(col).strip().lower(): col for col in tbl.columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in lookup:
            return lookup[key]
    return None


def _wiki_tickers(url: str, symbol_col: str, name_col: str = "", sector_col: str = "GICS Sector") -> List[str]:
    """Generic Wikipedia table scraper — returns normalized ticker list."""
    import io, requests
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    r.raise_for_status()
    tables = pd.read_html(io.StringIO(r.text))
    # Find the table that contains the symbol column
    for tbl in tables:
        resolved_symbol_col = _resolve_wiki_column(tbl, symbol_col)
        if resolved_symbol_col:
            syms = [str(s).strip().replace(".", "-") for s in tbl[resolved_symbol_col].tolist()]
            return [s for s in syms if s and s != "nan"]
    raise RuntimeError(f"Column '{symbol_col}' not found in any table at {url}")


def _wiki_tickers_with_meta(url: str, symbol_col: str, name_col: str = "", sector_col: str = "GICS Sector") -> dict:
    """Generic Wikipedia table scraper — returns {ticker: {name, sector}} dict."""
    import io, requests
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    r.raise_for_status()
    tables = pd.read_html(io.StringIO(r.text))
    for tbl in tables:
        resolved_symbol_col = _resolve_wiki_column(tbl, symbol_col)
        resolved_name_col = _resolve_wiki_column(tbl, name_col) if name_col else None
        resolved_sector_col = _resolve_wiki_column(tbl, sector_col) if sector_col else None
        if resolved_symbol_col:
            result = {}
            for _, row in tbl.iterrows():
                sym = str(row[resolved_symbol_col]).strip().replace(".", "-")
                if not sym or sym == "nan":
                    continue
                meta = {}
                if resolved_name_col:
                    name = str(row[resolved_name_col]).strip()
                    if name and name != "nan":
                        meta["name"] = name
                if resolved_sector_col:
                    sector = str(row[resolved_sector_col]).strip()
                    if sector and sector != "nan":
                        meta["sector"] = sector
                result[sym] = meta
            return result
    raise RuntimeError(f"Column '{symbol_col}' not found in any table at {url}")


def save_ticker_metadata(metadata_dict: dict):
    """Write ticker metadata to backend/ticker_metadata.json."""
    import json as _json
    out_path = Path(__file__).parent / "ticker_metadata.json"
    out_path.write_text(_json.dumps(metadata_dict, indent=2, ensure_ascii=False))
    print(f"Saved ticker metadata ({len(metadata_dict)} tickers) to {out_path}")


def fetch_sp500_tickers() -> List[str]:
    return _wiki_tickers(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Symbol"
    )


def fetch_nasdaq100_tickers() -> List[str]:
    return _wiki_tickers(
        "https://en.wikipedia.org/wiki/Nasdaq-100", "Ticker"
    )


def fetch_sp600_tickers() -> List[str]:
    return _wiki_tickers(
        "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", "Symbol"
    )


def fetch_russell1000_tickers() -> List[str]:
    """Russell 1000 = S&P 500 + S&P 400 mid-caps (approximate).
    Wikipedia doesn't list all 1000, so we combine S&P 500 + S&P 400."""
    sp500 = fetch_sp500_tickers()
    # Wikipedia S&P 400 page uses "Symbol" not "Ticker"
    for col in ("Symbol", "Ticker", "Ticker symbol"):
        try:
            sp400 = _wiki_tickers(
                "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", col
            )
            break
        except Exception:
            sp400 = []
    combined = list(dict.fromkeys(sp500 + sp400))
    return combined


def fetch_sp1500_tickers() -> List[str]:
    """Approximate S&P 1500 = S&P 500 + S&P 400 + S&P 600."""
    sp500 = fetch_sp500_tickers()
    sp400 = _wiki_tickers("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", "Symbol")
    sp600 = fetch_sp600_tickers()
    return list(dict.fromkeys(sp500 + sp400 + sp600))


def _load_or_fetch(cache_file: Path, fetch_fn, label: str) -> List[str]:
    if cache_file.exists():
        tickers = [l.strip() for l in cache_file.read_text().splitlines() if l.strip()]
        if tickers:
            print(f"Loaded {len(tickers)} {label} tickers from {cache_file.name}")
            return tickers
        print(f"Cached {label} list is empty; refetching...")
    print(f"Fetching {label} constituents from Wikipedia...")
    # Fetch with metadata and save both txt cache and metadata
    meta = _fetch_meta_for_label(label)
    if not meta and fetch_fn is not None:
        try:
            tickers = fetch_fn()
            meta = {t: {} for t in tickers}
        except Exception:
            meta = {}
    tickers = list(meta.keys())
    cache_file.write_text("\n".join(tickers) + "\n")
    print(f"Cached {len(tickers)} tickers to {cache_file.name}")
    # Merge into global metadata file
    _merge_metadata(meta)
    return tickers


def _fetch_meta_for_label(label: str) -> dict:
    """Fetch metadata dict {ticker: {name, sector}} from Wikipedia for the given universe label."""
    label_upper = label.upper()
    if "S&P 500" in label_upper or label_upper == "S&P 500":
        return _wiki_tickers_with_meta(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            symbol_col="Symbol", name_col="Security", sector_col="GICS Sector",
        )
    if "NASDAQ" in label_upper:
        return _wiki_tickers_with_meta(
            "https://en.wikipedia.org/wiki/Nasdaq-100",
            symbol_col="Ticker", name_col="Company", sector_col="GICS Sector",
        )
    if "S&P 600" in label_upper or "SMALLCAP" in label_upper:
        return _wiki_tickers_with_meta(
            "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
            symbol_col="Symbol", name_col="Security", sector_col="GICS Sector",
        )
    if "S&P 1500" in label_upper:
        meta = {}
        try:
            meta.update(_wiki_tickers_with_meta(
                "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                symbol_col="Symbol", name_col="Security", sector_col="GICS Sector",
            ))
        except Exception as e:
            print(f"  S&P 500 meta fetch failed: {e}")
        try:
            meta.update(_wiki_tickers_with_meta(
                "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
                symbol_col="Symbol", name_col="Company", sector_col="GICS Sector",
            ))
        except Exception as e:
            print(f"  S&P 400 meta fetch failed: {e}")
        try:
            meta.update(_wiki_tickers_with_meta(
                "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
                symbol_col="Symbol", name_col="Security", sector_col="GICS Sector",
            ))
        except Exception as e:
            print(f"  S&P 600 meta fetch failed: {e}")
        return meta
    if "S&P 400" in label_upper or "RUSSELL" in label_upper:
        meta = {}
        try:
            meta.update(_wiki_tickers_with_meta(
                "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                symbol_col="Symbol", name_col="Security", sector_col="GICS Sector",
            ))
        except Exception as e:
            print(f"  S&P 500 meta fetch failed: {e}")
        # Wikipedia S&P 400 page uses "Symbol" not "Ticker"
        for sym_col in ("Symbol", "Ticker", "Ticker symbol"):
            try:
                meta.update(_wiki_tickers_with_meta(
                    "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
                    symbol_col=sym_col, name_col="Company", sector_col="GICS Sector",
                ))
                break
            except Exception as e:
                last_err = e
        else:
            print(f"  S&P 400 meta fetch failed: {last_err}")
        return meta
    return {}


def _merge_metadata(new_meta: dict):
    """Merge new_meta into the existing ticker_metadata.json (non-destructively)."""
    import json as _json
    out_path = Path(__file__).parent / "ticker_metadata.json"
    existing = {}
    if out_path.exists():
        try:
            existing = _json.loads(out_path.read_text())
        except Exception:
            pass
    existing.update(new_meta)
    save_ticker_metadata(existing)


def generate_metadata_from_wikipedia():
    """Fetch fresh ticker metadata from Wikipedia and write ticker_metadata.json.
    Combines S&P 500, NASDAQ 100, S&P 400, and S&P 600 sources."""
    import json as _json
    print("Fetching S&P 500 metadata from Wikipedia...")
    meta: dict = {}
    try:
        sp500_meta = _wiki_tickers_with_meta(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            symbol_col="Symbol", name_col="Security", sector_col="GICS Sector",
        )
        meta.update(sp500_meta)
        print(f"  S&P 500: {len(sp500_meta)} tickers")
    except Exception as e:
        print(f"  S&P 500 fetch failed: {e}")

    print("Fetching NASDAQ 100 metadata from Wikipedia...")
    try:
        ndx_meta = _wiki_tickers_with_meta(
            "https://en.wikipedia.org/wiki/Nasdaq-100",
            symbol_col="Ticker", name_col="Company", sector_col="GICS Sector",
        )
        ndx_new = sum(1 for t in ndx_meta if t not in meta)
        for t, info in ndx_meta.items():
            if t not in meta:
                meta[t] = info
        print(f"  NASDAQ 100: {len(ndx_meta)} tickers ({ndx_new} new)")
    except Exception as e:
        print(f"  NASDAQ 100 fetch failed: {e}")

    print("Fetching S&P 400 metadata from Wikipedia...")
    try:
        sp400_meta = {}
        for sym_col in ("Symbol", "Ticker", "Ticker symbol"):
            try:
                sp400_meta = _wiki_tickers_with_meta(
                    "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
                    symbol_col=sym_col, name_col="Company", sector_col="GICS Sector",
                )
                break
            except Exception:
                continue
        if not sp400_meta:
            raise RuntimeError("Could not resolve a ticker column for the S&P 400 table")
        sp400_new = sum(1 for t in sp400_meta if t not in meta)
        for t, info in sp400_meta.items():
            if t not in meta:
                meta[t] = info
        print(f"  S&P 400: {len(sp400_meta)} tickers ({sp400_new} new)")
    except Exception as e:
        print(f"  S&P 400 fetch failed: {e}")

    print("Fetching S&P 600 metadata from Wikipedia...")
    try:
        sp600_meta = _wiki_tickers_with_meta(
            "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
            symbol_col="Symbol", name_col="Security", sector_col="GICS Sector",
        )
        sp600_new = sum(1 for t in sp600_meta if t not in meta)
        for t, info in sp600_meta.items():
            if t not in meta:
                meta[t] = info
        print(f"  S&P 600: {len(sp600_meta)} tickers ({sp600_new} new)")
    except Exception as e:
        print(f"  S&P 600 fetch failed: {e}")

    save_ticker_metadata(meta)
    return meta


def load_universe(spec: str) -> List[str]:
    base = Path(__file__).parent
    spec_upper = spec.upper()

    if spec_upper == "DEFAULT":
        return DEFAULT_UNIVERSE

    if spec_upper == "SP500":
        return _load_or_fetch(base / "sp500.txt", fetch_sp500_tickers, "S&P 500")

    if spec_upper == "NASDAQ100":
        return _load_or_fetch(base / "nasdaq100.txt", fetch_nasdaq100_tickers, "NASDAQ 100")

    if spec_upper == "RUSSELL1000":
        return _load_or_fetch(base / "russell1000.txt", fetch_russell1000_tickers, "Russell 1000")

    if spec_upper == "SP600":
        return _load_or_fetch(base / "sp600.txt", fetch_sp600_tickers, "S&P 600")

    if spec_upper == "SP1500":
        return _load_or_fetch(base / "sp1500.txt", fetch_sp1500_tickers, "S&P 1500")

    if spec_upper in ("SP500+NASDAQ100", "NASDAQ100+SP500"):
        sp500 = _load_or_fetch(base / "sp500.txt", fetch_sp500_tickers, "S&P 500")
        ndx = _load_or_fetch(base / "nasdaq100.txt", fetch_nasdaq100_tickers, "NASDAQ 100")
        combined = list(dict.fromkeys(sp500 + ndx))
        print(f"Combined universe: {len(combined)} unique tickers")
        return combined

    if spec_upper in ("SP500+RUSSELL1000", "RUSSELL1000+SP500"):
        return _load_or_fetch(base / "russell1000.txt", fetch_russell1000_tickers, "Russell 1000")

    if spec_upper in ("SP500+SP400+SP600", "SP1500+SP500", "SP500+SP400+SP600+NASDAQ100"):
        return _load_or_fetch(base / "sp1500.txt", fetch_sp1500_tickers, "S&P 1500")

    path = Path(spec)
    if path.exists():
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return [t.strip() for t in spec.split(",") if t.strip()]


def fetch_long_panel(
    tickers: List[str],
    years: int,
    cache_path: Optional[Path] = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch OHLCV+indicators for every ticker, return one long DataFrame
    indexed by (date, ticker) with the per-ticker feature columns.

    Cached to parquet — subsequent runs load in seconds instead of refetching.
    """
    if cache_path and cache_path.exists() and not refresh:
        print(f"Loading cached panel from {cache_path}")
        df = pd.read_parquet(cache_path)
        # Older caches may have un-normalized dates with time components;
        # force midnight here so joins downstream don't silently drop everything.
        date_idx = df.index.get_level_values("date")
        if date_idx.tz is not None:
            date_idx = date_idx.tz_localize(None)
        date_idx = date_idx.normalize()
        df.index = pd.MultiIndex.from_arrays(
            [date_idx, df.index.get_level_values("ticker")],
            names=["date", "ticker"],
        )
        cached_tickers = set(df.index.get_level_values("ticker").unique())
        missing = [t for t in tickers if t not in cached_tickers]
        if not missing:
            print(f"  {len(df):,} rows, {len(cached_tickers)} tickers")
            return df
        print(f"  cache missing {len(missing)} tickers — refetching those only")
        try:
            new_part = _fetch_long_part(missing, years)
            df = pd.concat([df, new_part]).sort_index()
            df.to_parquet(cache_path)
        except RuntimeError:
            print(f"  all {len(missing)} missing tickers failed to fetch — using cached data as-is")
        print(f"  {len(df):,} rows, {df.index.get_level_values('ticker').nunique()} tickers")
        return df

    df = _fetch_long_part(tickers, years)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
        print(f"Cached panel to {cache_path}")
    return df


def _fetch_long_part(tickers: List[str], years: int) -> pd.DataFrame:
    period = f"{years}y"
    parts = []
    skipped = 0
    for i, t in enumerate(tickers):
        try:
            df = market.fetch_history(t, period=period, interval="1d")
        except Exception as e:
            print(f"  skip {t}: fetch failed ({e})")
            skipped += 1
            continue
        if df.empty or len(df) < 60:
            skipped += 1
            continue
        df = market.add_indicators(df).dropna()
        if df.empty:
            skipped += 1
            continue
        df = df.copy()
        idx = pd.to_datetime(df.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        df.index = idx.normalize()  # force every date to midnight so all joins align
        df["ticker"] = t
        df = df.set_index("ticker", append=True)
        df.index.names = ["date", "ticker"]
        parts.append(df)
        if (i + 1) % 25 == 0 or i == len(tickers) - 1:
            print(f"  [{i+1}/{len(tickers)}] fetched (skipped {skipped})")
    if not parts:
        if len(tickers) <= 5:
            # Small missing-ticker refetch — silently skip rather than crash
            raise RuntimeError(f"No data for {tickers}")
        raise RuntimeError("No tickers produced data")
    return pd.concat(parts).sort_index()


MACRO_FEATURES = ["vix", "vix_ret_5d", "yield_spread", "spy_ret_5d"]


def fetch_macro(years: int) -> pd.DataFrame:
    """Daily macro context: VIX level + 5d change, 10y-5y yield spread, SPY 5d return.

    Returned as a date-indexed DataFrame to be joined onto the long panel.
    Anything missing is forward-filled (e.g. holidays where one feed lags)."""
    period = f"{years}y"
    out = pd.DataFrame()

    def grab(sym, col):
        try:
            df = market.fetch_history(sym, period=period, interval="1d")
            if df.empty:
                return pd.Series(dtype=float, name=col)
            s = df["close"].copy()
            s.index = pd.to_datetime(s.index).tz_localize(None) if s.index.tz else pd.to_datetime(s.index)
            s.name = col
            return s
        except Exception as e:
            print(f"  macro fetch {sym}: {e}")
            return pd.Series(dtype=float, name=col)

    vix = grab("^VIX", "vix")
    tnx = grab("^TNX", "y10")  # 10y treasury yield (in tens of percent — Yahoo convention)
    fvx = grab("^FVX", "y5")   # 5y treasury yield
    spy = grab("SPY", "spy_close")

    out = pd.concat([vix, tnx, fvx, spy], axis=1).ffill()
    out["vix_ret_5d"] = out["vix"].pct_change(5)
    out["yield_spread"] = out["y10"] - out["y5"]
    out["spy_ret_5d"] = out["spy_close"].pct_change(5)
    out = out[MACRO_FEATURES].dropna()
    out.index.name = "date"
    print(f"  macro panel: {len(out)} days, {list(out.columns)}")
    return out


def join_macro(panel: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Left-join macro by date onto the (date, ticker) panel."""
    p = panel.reset_index()
    p["date"] = pd.to_datetime(p["date"])
    if p["date"].dt.tz is not None:
        p["date"] = p["date"].dt.tz_localize(None)
    p["date"] = p["date"].dt.normalize()
    m = macro.reset_index()
    m["date"] = pd.to_datetime(m["date"])
    if m["date"].dt.tz is not None:
        m["date"] = m["date"].dt.tz_localize(None)
    m["date"] = m["date"].dt.normalize()
    merged = p.merge(m, on="date", how="left")
    # Drop rows where any macro feature is NaN (early dates before macro is available)
    before = len(merged)
    merged = merged.dropna(subset=MACRO_FEATURES)
    print(f"  joined macro: dropped {before - len(merged):,} rows missing macro context")
    return merged.set_index(["date", "ticker"]).sort_index()


CROSS_SECTIONAL_BASE = ["rsi_14", "ret_1d", "volume", "macd"]
CROSS_SECTIONAL_FEATURES = [f"xs_rank_{c}" for c in CROSS_SECTIONAL_BASE]

SENTIMENT_FEATURES = ["sentiment_mean", "sentiment_std", "log_news_count", "has_news",
                      "sentiment_5d_mean", "sentiment_surge"]

EARNINGS_FEATURES = ["eps_surprise_pct", "days_since_earn_norm", "is_earnings_window"]


def fetch_earnings_features(
    tickers: List[str],
    cache_path: Optional[Path] = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch historical EPS actual/estimate for every ticker via yfinance.
    Returns (ticker, earn_date, eps_surprise_pct) — one row per earnings release."""
    if cache_path and cache_path.exists() and not refresh:
        df = pd.read_parquet(cache_path)
        print(f"Loaded earnings cache: {len(df):,} rows ({df['ticker'].nunique()} tickers)")
        return df

    import yfinance as yf
    rows: list = []
    failed = 0
    for i, t in enumerate(tickers):
        try:
            ed = yf.Ticker(t).earnings_dates
            if ed is None or len(ed) == 0:
                continue
            ed = ed.reset_index()
            date_col = ed.columns[0]
            ed["earn_date"] = pd.to_datetime(ed[date_col]).dt.tz_localize(None).dt.normalize()
            ed["ticker"] = t
            if "Surprise(%)" in ed.columns:
                ed["eps_surprise_pct"] = pd.to_numeric(ed["Surprise(%)"], errors="coerce").div(100.0)
            elif "EPS Estimate" in ed.columns and "Reported EPS" in ed.columns:
                est = pd.to_numeric(ed["EPS Estimate"], errors="coerce")
                rep = pd.to_numeric(ed["Reported EPS"], errors="coerce")
                ed["eps_surprise_pct"] = np.where(est.abs() > 0.01, (rep - est) / est.abs(), 0.0)
            else:
                continue
            valid = ed.dropna(subset=["earn_date", "eps_surprise_pct"])
            if not valid.empty:
                rows.append(valid[["ticker", "earn_date", "eps_surprise_pct"]])
        except Exception:
            failed += 1
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(tickers)}] earnings fetched (failed={failed})")

    if not rows:
        print("  [earnings] no data fetched — skipping")
        return pd.DataFrame(columns=["ticker", "earn_date", "eps_surprise_pct"])

    result = pd.concat(rows, ignore_index=True)
    result["eps_surprise_pct"] = result["eps_surprise_pct"].clip(-2.0, 2.0).astype(np.float32)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(cache_path)
    print(f"  [earnings] {len(result):,} releases, {result['ticker'].nunique()} tickers")
    return result


def join_earnings(panel: pd.DataFrame, earnings_df: pd.DataFrame) -> pd.DataFrame:
    """Backward-join most-recent earnings to every (ticker, date) row.

    eps_surprise_pct  — actual EPS surprise fraction (decays to 0 over DECAY_DAYS)
    days_since_earn_norm — 0=just happened, 1=≥DECAY_DAYS ago
    is_earnings_window   — 1 if within WINDOW_DAYS calendar days of a release
    """
    DECAY_DAYS = 90
    WINDOW_DAYS = 7

    if earnings_df.empty:
        for col, val in [("eps_surprise_pct", 0.0),
                         ("days_since_earn_norm", 1.0),
                         ("is_earnings_window", 0.0)]:
            panel[col] = val
        return panel

    p = panel.reset_index()
    p["date"] = pd.to_datetime(p["date"]).dt.normalize()

    e = (earnings_df.copy()
         .rename(columns={"earn_date": "date"})
         .sort_values("date"))   # merge_asof requires sort by the on-key only
    e["_earn_orig"] = e["date"]

    merged = pd.merge_asof(
        p.sort_values("date"),
        e[["ticker", "date", "_earn_orig", "eps_surprise_pct"]]
          .rename(columns={"eps_surprise_pct": "_earn_eps"}),
        on="date", by="ticker", direction="backward",
    )

    cal_days = (merged["date"] - merged["_earn_orig"]).dt.days.fillna(999)
    weight = ((DECAY_DAYS - cal_days.clip(WINDOW_DAYS, DECAY_DAYS))
              / (DECAY_DAYS - WINDOW_DAYS)).clip(0, 1)
    merged["eps_surprise_pct"] = (merged["_earn_eps"].fillna(0.0) * weight).astype(np.float32)
    merged["days_since_earn_norm"] = (cal_days.clip(0, DECAY_DAYS) / DECAY_DAYS).astype(np.float32)
    merged["is_earnings_window"] = (cal_days <= WINDOW_DAYS).astype(np.float32)
    merged.drop(columns=["_earn_orig", "_earn_eps"], inplace=True)

    n_window = (merged["is_earnings_window"] == 1.0).sum()
    print(f"  [earnings] {n_window:,} ticker-days within earnings window ({n_window/len(merged):.1%})")
    return merged.set_index(["date", "ticker"]).sort_index()


def join_sentiment(panel: pd.DataFrame, sentiment_path: Path) -> pd.DataFrame:
    """Left-join historical FinBERT sentiment onto the (date, ticker) panel.

    Rows with no news on a given day get neutral defaults (sentiment=0, std=0,
    count=0). Dropping them would erase 70%+ of the data — better to let the
    model learn that 'no news' is a distinct (and common) state."""
    if not sentiment_path.exists():
        print(f"  [sentiment] {sentiment_path} not found — skipping (run score_fnspid.py first)")
        return panel
    sent = pd.read_parquet(sentiment_path)
    # Normalize both sides to midnight so a time component on the panel index
    # (yfinance can return market-open timestamps) doesn't break the join.
    sent["date"] = pd.to_datetime(sent["date"]).dt.normalize()
    sent["ticker"] = sent["ticker"].astype(str).str.upper().str.strip()
    sent = sent.rename(columns={"news_count": "_count"})
    sent["log_news_count"] = np.log1p(sent["_count"])
    sent["has_news"] = 1.0
    sent = sent[["ticker", "date", "sentiment_mean", "sentiment_std", "log_news_count", "has_news"]]

    p = panel.reset_index()
    p["date"] = pd.to_datetime(p["date"]).dt.normalize()
    p["ticker"] = p["ticker"].astype(str).str.upper().str.strip()

    # Diagnostic — these tell us whether the join keys actually overlap
    common_tickers = set(p["ticker"].unique()) & set(sent["ticker"].unique())
    print(f"  [sentiment] panel tickers={p['ticker'].nunique()}, "
          f"sentiment tickers={sent['ticker'].nunique()}, common={len(common_tickers)}")
    print(f"  [sentiment] panel dates  : {p['date'].min().date()} → {p['date'].max().date()}")
    print(f"  [sentiment] sent  dates  : {sent['date'].min().date()} → {sent['date'].max().date()}")

    merged = p.merge(sent, on=["ticker", "date"], how="left")

    # Neutral fills for ticker-days without news
    merged["sentiment_mean"] = merged["sentiment_mean"].fillna(0.0)
    merged["sentiment_std"] = merged["sentiment_std"].fillna(0.0)
    merged["log_news_count"] = merged["log_news_count"].fillna(0.0)
    merged["has_news"] = merged["has_news"].fillna(0.0)

    # Rolling sentiment: sort by (ticker, date) so groupby rolling is chronological
    merged = merged.sort_values(["ticker", "date"])
    grp = merged.groupby("ticker")["sentiment_mean"]
    merged["sentiment_5d_mean"] = (
        grp.transform(lambda x: x.rolling(5, min_periods=1).mean()).fillna(0.0)
    )
    sent_20d = grp.transform(lambda x: x.rolling(20, min_periods=5).mean()).fillna(0.0)
    merged["sentiment_surge"] = (merged["sentiment_mean"] - sent_20d).fillna(0.0)

    matched = (merged["has_news"] == 1.0).sum()
    print(f"  [sentiment] {matched:,} / {len(merged):,} ticker-days had news ({matched/len(merged):.1%})")
    return merged.set_index(["date", "ticker"]).sort_index()


def add_cross_sectional(panel: pd.DataFrame) -> pd.DataFrame:
    """For each date, rank tickers against each other on key features.

    Output is in [0, 1] — pct rank within the universe on that date.
    A value of 0.95 on `xs_rank_rsi_14` means this ticker's RSI is in the
    top 5% of all ranked tickers today. This is the kind of signal that
    long-short quant strategies trade on."""
    p = panel.copy()
    print(f"  computing cross-sectional ranks for {CROSS_SECTIONAL_BASE}...")
    for col in CROSS_SECTIONAL_BASE:
        if col not in p.columns:
            print(f"    skip {col}: not in panel")
            continue
        # pct=True → rank scaled to [0,1]; method='average' handles ties
        p[f"xs_rank_{col}"] = p.groupby(level="date")[col].rank(pct=True, method="average")
    # Some early dates may have only one ticker (rank trivially 0.5 by 'average').
    # Keep them — won't hurt training.
    p = p.dropna(subset=CROSS_SECTIONAL_FEATURES)
    print(f"  panel after cross-sectional: {len(p):,} rows")
    return p


def build_windows(
    panel: pd.DataFrame,
    feature_cols: List[str],
    window: int,
    horizon: int,
    label_mode: str = "rel_spy",
    spy_close: Optional[pd.Series] = None,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Slide windows over the long panel per ticker.

    label_mode:
      - 'absolute': y = 1 if stock_ret > cost (old behavior; baseline ≈ market drift)
      - 'rel_spy':  y = 1 if stock_ret > spy_ret + cost — the model now has to find
                    *stock-specific* signal instead of memorizing market drift.
                    Baseline becomes ~50% by construction.

    stride: step between windows (stride=5 gives ~5x fewer samples, saves RAM).
            Adjacent windows with stride=1 are ~95% correlated anyway so stride=5
            loses almost no information while cutting peak RAM from ~24 GB to ~5 GB.

    Returns X (N, window, n_features), y (N,), dates (N,), tickers (N,).
    """
    if label_mode == "rel_spy" and spy_close is None:
        raise ValueError("rel_spy label requires spy_close series")

    # Two-pass to avoid the list-of-arrays + np.asarray() peak (which doubles
    # memory because the list and final array coexist briefly). For SP1500/10y
    # that spike pushed us to 40+ GB and froze the box. Pre-allocating with
    # np.empty + fill-by-index keeps peak at ~N*window*nfeat*4 bytes flat.
    n_features = len(feature_cols)
    groups = list(panel.groupby(level="ticker", sort=False))

    # --- Pass 1: count valid samples per ticker ---
    per_ticker = []  # list of (ticker, sub_df, count)
    total = 0
    skipped_no_spy = 0
    for t, sub in groups:
        sub = sub.droplevel("ticker").sort_index()
        if len(sub) < window + horizon + 5:
            continue
        close = sub["close"].values
        if label_mode == "rel_spy":
            spy_aligned = spy_close.reindex(sub.index).ffill().values
        else:
            spy_aligned = None
        count = 0
        n_steps = len(sub) - window - horizon
        for k in range(0, n_steps, stride):
            if close[k + window - 1] <= 0:
                continue
            if spy_aligned is not None:
                spy_now = spy_aligned[k + window - 1]
                spy_fut = spy_aligned[k + window + horizon - 1]
                if not (np.isfinite(spy_now) and np.isfinite(spy_fut) and spy_now > 0):
                    skipped_no_spy += 1
                    continue
            count += 1
        if count > 0:
            per_ticker.append((t, sub, spy_aligned, count))
            total += count

    if total == 0:
        raise RuntimeError("No windows produced — check window/horizon vs data length")

    # --- Pass 2: pre-allocate, fill by index ---
    X = np.empty((total, window, n_features), dtype=np.float32)
    y = np.empty(total, dtype=np.float32)
    dates = np.empty(total, dtype="datetime64[ns]")
    tkrs = np.empty(total, dtype=object)

    pos = 0
    for t, sub, spy_aligned, _count in per_ticker:
        feat = sub[feature_cols].values.astype(np.float32, copy=False)
        close = sub["close"].values
        idx = sub.index.values
        n_steps = len(sub) - window - horizon
        for k in range(0, n_steps, stride):
            present = close[k + window - 1]
            if present <= 0:
                continue
            future = close[k + window + horizon - 1]
            stock_ret = (future - present) / present
            if spy_aligned is not None:
                spy_now = spy_aligned[k + window - 1]
                spy_fut = spy_aligned[k + window + horizon - 1]
                if not (np.isfinite(spy_now) and np.isfinite(spy_fut) and spy_now > 0):
                    continue
                spy_ret = (spy_fut - spy_now) / spy_now
                y_val = stock_ret - spy_ret
            else:
                y_val = stock_ret
            X[pos] = feat[k : k + window]
            y[pos] = y_val
            dates[pos] = idx[k + window - 1]
            tkrs[pos] = t
            pos += 1

    # Defensive: pos should equal total, but truncate if a row slipped through.
    if pos != total:
        X = X[:pos]; y = y[:pos]; dates = dates[:pos]; tkrs = tkrs[:pos]
    if skipped_no_spy:
        print(f"  skipped {skipped_no_spy:,} windows missing SPY context")
    return X, y, dates, tkrs


def walk_forward_folds(
    X: np.ndarray, y: np.ndarray, dates: np.ndarray,
    n_folds: int = 5, val_frac: float = 0.15,
):
    """Expanding-window CV. Each fold trains on everything before some
    cutoff, validates on a slice, and tests on the next slice.

        |---- train ----|-val-|-test-|         (fold 1)
        |------- train -------|-val-|-test-|   (fold 2)
        ...

    Yields dict per fold with train/val/test arrays and date ranges.
    """
    order = np.argsort(dates)
    X, y, dates = X[order], y[order], dates[order]
    n = len(X)
    test_size = n // (n_folds + 1)
    val_size = max(int(test_size * val_frac / (1 - val_frac)), 1)
    for k in range(n_folds):
        test_end = n - (n_folds - 1 - k) * test_size
        test_start = test_end - test_size
        val_end = test_start
        val_start = val_end - val_size
        if val_start <= 0:
            continue
        yield {
            "fold": k + 1,
            "Xtr": X[:val_start], "ytr": y[:val_start], "dtr": dates[:val_start],
            "Xva": X[val_start:val_end], "yva": y[val_start:val_end], "dva": dates[val_start:val_end],
            "Xte": X[val_end:test_end], "yte": y[val_end:test_end], "dte": dates[val_end:test_end],
        }


def train_loop(
    Xtr: np.ndarray, ytr: np.ndarray,
    Xva: np.ndarray, yva: np.ndarray,
    epochs: int, batch: int, lr: float, hidden: int,
    device: str, patience: int = 5,
    dropout: float = 0.3, num_layers: int = 2,
    weight_decay: float = 1e-3,
) -> Tuple[FusionLSTM, dict]:
    n_features = Xtr.shape[-1]
    model = FusionLSTM(
        n_features=n_features, hidden=hidden, with_sentiment=False,
        dropout=dropout, num_layers=num_layers,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # HuberLoss is more robust than MSE for fat-tailed stock return distributions.
    # delta=0.02 means the transition from quadratic to linear happens at ±2%.
    loss_fn = nn.HuberLoss(delta=0.02)

    Xtr_t = torch.from_numpy(Xtr)
    ytr_t = torch.from_numpy(ytr)   # float32 returns (fractions)
    Xva_t = torch.from_numpy(Xva)
    yva_t = torch.from_numpy(yva)

    n = len(Xtr_t)
    best_val_mae = float("inf")
    best_state = None
    bad_epochs = 0
    history = {"train_dir_acc": [], "val_dir_acc": [], "val_mae": [], "train_loss": []}

    eval_batch = max(batch, 4096)

    @torch.no_grad()
    def batched_eval(X_cpu, y_cpu):
        preds = []
        for i in range(0, len(X_cpu), eval_batch):
            xb = X_cpu[i : i + eval_batch].to(device, non_blocking=True)
            preds.append(model(xb).cpu())
        preds = torch.cat(preds)
        mae = (preds - y_cpu).abs().mean().item()
        dir_acc = ((preds > 0) == (y_cpu > 0)).float().mean().item()
        return mae, dir_acc

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        ep_loss = 0.0
        for i in range(0, n, batch):
            sel = perm[i : i + batch]
            xb = Xtr_t[sel].to(device, non_blocking=True)
            yb = ytr_t[sel].to(device, non_blocking=True)
            opt.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * len(sel)
        ep_loss /= n

        model.eval()
        _, tr_dir = batched_eval(Xtr_t, ytr_t)
        va_mae, va_dir = batched_eval(Xva_t, yva_t)
        history["train_loss"].append(ep_loss)
        history["train_dir_acc"].append(tr_dir)
        history["val_dir_acc"].append(va_dir)
        history["val_mae"].append(va_mae)

        improved = va_mae < best_val_mae - 1e-6
        if improved:
            best_val_mae = va_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        marker = "*" if improved else " "
        print(f"  ep {ep+1:3d}{marker}  loss={ep_loss:.5f}  "
              f"tr_dir={tr_dir:.4f}  va_dir={va_dir:.4f}  va_mae={va_mae*100:.3f}%")
        if bad_epochs >= patience:
            print(f"  early stop at epoch {ep+1} (no val MAE improvement for {patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--build-metadata", action="store_true",
                   help="Fetch ticker metadata from Wikipedia and write ticker_metadata.json, then exit")
    p.add_argument("--tickers", default="DEFAULT", help="DEFAULT, comma-list, or path to file")
    p.add_argument("--years", type=int, default=15)
    p.add_argument("--horizon", type=int, default=5, help="prediction horizon in trading days")
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--out", default="artifacts")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--refresh-cache", action="store_true",
                   help="ignore the cached parquet panel and refetch from Yahoo")
    p.add_argument("--folds", type=int, default=5,
                   help="walk-forward CV folds (5 is standard)")
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--label-mode", choices=["absolute", "rel_spy"], default="absolute",
                   help="absolute: y=stock went up; rel_spy: y=stock beat SPY (real alpha)")
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--stride", type=int, default=1,
                   help="step between windows; >1 trades samples for RAM (e.g. 5 → ~5x fewer samples)")
    p.add_argument("--no-sentiment", action="store_true",
                   help="skip the sentiment feature join (useful for ablation)")
    args = p.parse_args()

    if args.build_metadata:
        generate_metadata_from_wikipedia()
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {name}  sm_{cap[0]}{cap[1]}  {vram:.1f} GB  torch={torch.__version__}  cuda={torch.version.cuda}")
        # Sanity check: try a Blackwell-sensitive op. If torch was built without
        # sm_120 kernels, this raises immediately rather than mid-training.
        try:
            _ = torch.zeros(8, device="cuda") @ torch.zeros(8, device="cuda")
        except RuntimeError:
            print(f"\nERROR: GPU op failed — likely torch built without sm_{cap[0]}{cap[1]} support.")
            print("Reinstall torch with: pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128")
            raise
    else:
        print("  WARNING: CUDA not available — falling back to CPU. Training the S&P 500 panel on CPU is impractical.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    universe = load_universe(args.tickers)
    print(f"Universe: {len(universe)} tickers")

    cache_path = out_dir / f"panel_{args.tickers.lower()}_{args.years}y.parquet"
    macro_cache = out_dir / f"macro_{args.years}y.parquet"
    t0 = time.time()
    panel = fetch_long_panel(universe, args.years, cache_path=cache_path, refresh=args.refresh_cache)
    print(f"  panel ready in {time.time()-t0:.0f}s — {len(panel):,} (date, ticker) rows")

    if macro_cache.exists() and not args.refresh_cache:
        macro = pd.read_parquet(macro_cache)
        print(f"Loaded macro cache: {len(macro)} days")
    else:
        print("Fetching macro context (VIX, yields, SPY)...")
        macro = fetch_macro(args.years)
        macro.to_parquet(macro_cache)
    panel = join_macro(panel, macro)

    print("Adding cross-sectional rank features...")
    panel = add_cross_sectional(panel)

    feature_cols = list(FEATURES) + MACRO_FEATURES + CROSS_SECTIONAL_FEATURES

    if not args.no_sentiment:
        sentiment_path = out_dir / "historical_sentiment.parquet"
        print(f"Joining historical sentiment from {sentiment_path}...")
        panel_before = len(panel)
        panel = join_sentiment(panel, sentiment_path)
        if len(panel) == panel_before and "sentiment_mean" in panel.columns:
            feature_cols += SENTIMENT_FEATURES
        elif "sentiment_mean" in panel.columns:
            feature_cols += SENTIMENT_FEATURES

    # Earnings features
    earnings_cache = out_dir / "earnings_features.parquet"
    print("Fetching historical earnings surprises...")
    earnings_df = fetch_earnings_features(universe, cache_path=earnings_cache, refresh=args.refresh_cache)
    panel = join_earnings(panel, earnings_df)
    if not earnings_df.empty:
        feature_cols += EARNINGS_FEATURES

    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Building windows — horizon={args.horizon}d, window={args.window}d, label={args.label_mode}")

    spy_close = None
    if args.label_mode == "rel_spy":
        try:
            spy_df = market.fetch_history("SPY", period=f"{args.years}y", interval="1d")
            spy_close = spy_df["close"]
            idx = pd.to_datetime(spy_close.index)
            if idx.tz is not None:
                idx = idx.tz_localize(None)
            # Normalize to midnight so SPY's index matches the panel's normalized
            # dates (the sentiment join calls .dt.normalize()). Without this,
            # reindex returns all-NaN and every window gets skipped.
            spy_close.index = idx.normalize()
            print(f"  SPY benchmark loaded: {len(spy_close)} days")
        except Exception as e:
            raise RuntimeError(f"rel_spy needs SPY history but fetch failed: {e}")

    X, y, dates, tkrs = build_windows(panel, feature_cols, args.window, args.horizon,
                                       label_mode=args.label_mode, spy_close=spy_close,
                                       stride=args.stride)
    pos_rate = (y > 0).mean()
    print(f"  {len(X):,} samples, stride={args.stride}  (positive return rate: {pos_rate:.3f})")

    print(f"\nRunning {args.folds}-fold walk-forward CV...")

    import json as _json
    ckpt_path = out_dir / "fold_checkpoint.json"
    fold_results: list = []
    final_model = None
    final_scaler = None
    final_X_shape = None

    # Resume: load results from any previously completed folds so a freeze only
    # loses the fold that was running, not all prior work.
    completed_folds: set = set()
    if ckpt_path.exists():
        try:
            saved = _json.loads(ckpt_path.read_text())
            fold_results = saved.get("fold_results", [])
            completed_folds = {r["fold"] for r in fold_results}
            print(f"[resume] found checkpoint — {len(completed_folds)} fold(s) already done: {sorted(completed_folds)}")
        except Exception as e:
            print(f"[resume] checkpoint unreadable ({e}) — starting fresh")

    for fold in walk_forward_folds(X, y, dates, n_folds=args.folds):
        k = fold["fold"]
        if k in completed_folds:
            print(f"\n--- Fold {k}/{args.folds} --- SKIPPED (already in checkpoint)")
            continue

        Xtr, ytr, dtr = fold["Xtr"], fold["ytr"], fold["dtr"]
        Xva, yva, dva = fold["Xva"], fold["yva"], fold["dva"]
        Xte, yte, dte = fold["Xte"], fold["yte"], fold["dte"]
        print(f"\n--- Fold {k}/{args.folds} ---")
        print(f"  train: {len(Xtr):>7d}  ({pd.Timestamp(dtr.min()).date()} → {pd.Timestamp(dtr.max()).date()})")
        print(f"  val:   {len(Xva):>7d}  ({pd.Timestamp(dva.min()).date()} → {pd.Timestamp(dva.max()).date()})")
        print(f"  test:  {len(Xte):>7d}  ({pd.Timestamp(dte.min()).date()} → {pd.Timestamp(dte.max()).date()})")

        scaler = StandardScaler()
        scaler.fit(Xtr.reshape(-1, Xtr.shape[-1]))
        def scale(A):
            flat = scaler.transform(A.reshape(-1, A.shape[-1])).astype(np.float32)
            return flat.reshape(A.shape)
        Xtr_s, Xva_s, Xte_s = scale(Xtr), scale(Xva), scale(Xte)

        model, hist = train_loop(Xtr_s, ytr, Xva_s, yva, args.epochs, args.batch,
                                 args.lr, args.hidden, device,
                                 dropout=args.dropout, num_layers=args.num_layers,
                                 weight_decay=args.weight_decay)

        model.eval()
        eval_bs = max(args.batch, 4096)
        Xte_cpu = torch.from_numpy(Xte_s)
        yte_cpu = torch.from_numpy(yte)   # float32 returns
        all_preds = []
        with torch.no_grad():
            for i in range(0, len(Xte_cpu), eval_bs):
                xb = Xte_cpu[i : i + eval_bs].to(device, non_blocking=True)
                all_preds.append(model(xb).cpu())
        all_preds = torch.cat(all_preds)
        test_mae = (all_preds - yte_cpu).abs().mean().item()
        test_dir_acc = ((all_preds > 0) == (yte_cpu > 0)).float().mean().item()
        baseline_mae = yte_cpu.abs().mean().item()   # always-predict-zero baseline

        print(f"  fold {k}  dir_acc={test_dir_acc:.4f}  "
              f"mae={test_mae*100:.3f}%  baseline_mae={baseline_mae*100:.3f}%")
        fold_result = {
            "fold": k,
            "test_dir_acc": test_dir_acc,
            "test_mae_pct": round(test_mae * 100, 4),
            "baseline_mae_pct": round(baseline_mae * 100, 4),
            "best_val_mae_pct": round(min(hist["val_mae"]) * 100, 4),
            "n_train": len(Xtr), "n_val": len(Xva), "n_test": len(Xte),
            "test_start": str(pd.Timestamp(dte.min()).date()),
            "test_end": str(pd.Timestamp(dte.max()).date()),
        }
        fold_results.append(fold_result)

        # Keep the LAST fold's model+scaler — it was trained on the most data
        # and is what we want to deploy for inference on today's prices.
        if k == args.folds:
            final_model = model
            final_scaler = scaler
            final_X_shape = Xtr.shape

        # Checkpoint after every completed fold so a crash only loses the fold
        # that was in-flight. Also save the model weights for this fold so we
        # always have a deployable checkpoint even if the last fold never finishes.
        fold_ckpt_model = out_dir / f"price_model_fold{k}.pt"
        fold_ckpt_scaler = out_dir / f"price_scaler_fold{k}.pkl"
        torch.save({
            "state_dict": model.state_dict(),
            "n_features": Xtr.shape[-1],
            "hidden": args.hidden,
            "with_sentiment": False,
            "dropout": args.dropout,
            "num_layers": args.num_layers,
            "window": args.window,
            "horizon": args.horizon,
            "features": feature_cols,
        }, fold_ckpt_model)
        with open(fold_ckpt_scaler, "wb") as f:
            pickle.dump(scaler, f)
        ckpt_path.write_text(_json.dumps({
            "fold_results": fold_results,
            "feature_cols": feature_cols,
            "args": vars(args),
        }, indent=2, default=str))
        print(f"  [ckpt] fold {k} saved → {fold_ckpt_model.name}")

        # Free per-fold memory aggressively. Without this, fold N's tensors
        # accumulate in the heap during fold N+1's training and a 16 GB laptop
        # OOMs around fold 3.
        del Xtr_s, Xva_s, Xte_s, Xte_cpu, yte_cpu
        del Xtr, ytr, Xva, yva, Xte, yte
        if k != args.folds:
            del model, scaler
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    dir_accs = np.array([r["test_dir_acc"] for r in fold_results])
    maes = np.array([r["test_mae_pct"] for r in fold_results])
    base_maes = np.array([r["baseline_mae_pct"] for r in fold_results])
    print(f"\n{'='*60}")
    print(f"WALK-FORWARD CV RESULTS ({len(fold_results)} folds)  [regression]")
    print(f"{'='*60}")
    print(f"Dir accuracy:   {dir_accs.mean():.4f} ± {dir_accs.std():.4f}  (per fold: {[f'{a:.4f}' for a in dir_accs]})")
    print(f"Test MAE:       {maes.mean():.3f}% ± {maes.std():.3f}%")
    print(f"Baseline MAE:   {base_maes.mean():.3f}% ± {base_maes.std():.3f}%  (always-predict-zero)")
    print(f"MAE vs baseline:{(maes - base_maes).mean():+.3f}pp")

    # Save the final-fold artifacts.  If the last fold was skipped because it
    # was already in the checkpoint, promote the per-fold file to the main path.
    model_path = out_dir / "price_model.pt"
    scaler_path = out_dir / "price_scaler.pkl"
    meta_path = out_dir / "price_meta.json"

    if final_model is None:
        # All folds were resumed from checkpoint — copy the last fold's files.
        last_k = max(r["fold"] for r in fold_results)
        src_model = out_dir / f"price_model_fold{last_k}.pt"
        src_scaler = out_dir / f"price_scaler_fold{last_k}.pkl"
        import shutil
        shutil.copy2(src_model, model_path)
        shutil.copy2(src_scaler, scaler_path)
        print(f"  [resume] promoted fold {last_k} checkpoint to {model_path.name}")
    else:
        torch.save({
            "state_dict": final_model.state_dict(),
            "n_features": final_X_shape[-1],
            "hidden": args.hidden,
            "with_sentiment": False,
            "dropout": args.dropout,
            "num_layers": args.num_layers,
            "window": args.window,
            "horizon": args.horizon,
            "features": feature_cols,
        }, model_path)
        with open(scaler_path, "wb") as f:
            pickle.dump(final_scaler, f)

    with open(meta_path, "w") as f:
        _json.dump({
            "tickers": universe,
            "feature_cols": feature_cols,
            "folds": fold_results,
            "mean_dir_acc": float(dir_accs.mean()),
            "std_dir_acc": float(dir_accs.std()),
            "mean_mae_pct": float(maes.mean()),
            "mean_baseline_mae_pct": float(base_maes.mean()),
            "args": vars(args),
        }, f, indent=2, default=str)

    # Clean up per-fold checkpoint files now that everything is saved.
    ckpt_path.unlink(missing_ok=True)
    for r in fold_results:
        (out_dir / f"price_model_fold{r['fold']}.pt").unlink(missing_ok=True)
        (out_dir / f"price_scaler_fold{r['fold']}.pkl").unlink(missing_ok=True)

    print(f"\nSaved:\n  {model_path}\n  {scaler_path}\n  {meta_path}")


if __name__ == "__main__":
    main()
