# SentiTrade

Bachelor's thesis prototype: a financial dashboard that fuses FinBERT news
sentiment with an LSTM trained on price + technical indicators to predict
short- and long-term price direction.

> **Status: prototype.** No fine-tuning yet. FinBERT is used pre-trained
> (`ProsusAI/finbert`) and the LSTM is trained on demand per ticker. Treat
> predictions as illustrative.

## What's in the box

- **FastAPI backend** — market data (yfinance + Yahoo Chart API + Stooq fallback),
  FinBERT sentiment, LSTM fusion model, JWT auth, SQLite persistence.
- **React + Vite frontend** — TradingView lightweight-charts, dark theme,
  per-user watchlist, notifications, predictions.
- **Background worker** — every 2 minutes, polls news for all watchlist tickers,
  scores only new articles, recomputes affected predictions, and detects macro
  /political news that could move many stocks at once.
- **Research archive** — every news article (with sentiment score), every
  price bar, and every prediction is stored in SQLite with timestamps so you
  can later evaluate accuracy or research news → price impact.

## Repo layout

```
backend/    FastAPI service
  app/
    main.py        routes, lifespan, cache layer
    market.py      yfinance + Yahoo + Stooq data
    sentiment.py   FinBERT
    model.py       LSTM fusion
    news_repo.py   news persistence + dedup
    background.py  async refresh loop
    models.py      SQLAlchemy schema
frontend/   React + Vite dashboard
```

## Prerequisites

- Python **3.11+**
- Node **18+** and npm
- ~2 GB free disk (FinBERT weights are ~440 MB on first run; PyTorch is ~800 MB)

## First-time setup

Clone and bootstrap both halves:

```bash
git clone <repo-url> sentitrade
cd sentitrade
```

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The first request that touches sentiment or predictions downloads the FinBERT
weights (~440 MB) and may take a minute. Subsequent calls hit the local cache.

#### Optional: production env vars

```bash
export JWT_SECRET="change-me-to-a-long-random-string"
```

If unset, a development default is used (fine for local but **must** be set in
production).

### Frontend

In a second terminal:

```bash
cd frontend
npm install
```

## Running locally

You need both processes running. Two terminals:

**Terminal 1 — backend:**
```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload
```
The API is now on http://localhost:8000. Health check: http://localhost:8000/health

**Terminal 2 — frontend:**
```bash
cd frontend
npm run dev
```
Open http://localhost:5173 in your browser. The dev server proxies `/api/*` to
the backend on port 8000, so just one URL to remember.

## First steps in the UI

1. Click "Register" and create an account (no email verification — local prototype).
2. Add a ticker to your watchlist (e.g. `AAPL`, `TSLA`, `NVDA`).
3. Click the ticker — first load fetches news and runs FinBERT (~10–20 s).
   Subsequent loads are instant because results are cached in the SQLite DB.
4. The background worker starts polling 15 seconds after server start; new
   articles will appear automatically and predictions will refresh themselves.

## API surface (selected)

| Endpoint                                | Purpose                                     |
| --------------------------------------- | ------------------------------------------- |
| `POST /api/auth/register`               | Create account, returns JWT                 |
| `POST /api/auth/login`                  | Email or username + password → JWT          |
| `GET  /api/home`                        | Indices, sectors, gainers/losers, news      |
| `GET  /api/price/{ticker}`              | OHLCV bars (`?period=1y&interval=1d`)       |
| `GET  /api/quote/{ticker}`              | Live quote incl. pre/post-market            |
| `GET  /api/news/{ticker}`               | News with FinBERT sentiment                 |
| `GET  /api/predict/{ticker}`            | LSTM 1d + 5d direction forecasts            |
| `GET  /api/watchlist`                   | Current user's watchlist + quotes           |
| `GET  /api/notifications`               | Price-move + macro alerts (read/dismiss)    |

All authenticated routes use `Authorization: Bearer <jwt>`.

## Database

A single SQLite file `backend/app.db` is created on first run via
`Base.metadata.create_all`. To start clean:

```bash
rm backend/app.db
```

Tables: `users`, `watchlist_items`, `notifications`, `cache`,
`news_articles` (with sentiment), `price_bars`, `predictions`.

## Troubleshooting

- **"No data" or empty sections** — the backend likely couldn't reach
  `query1.finance.yahoo.com`. Check the backend log; Yahoo occasionally rate-
  limits IP ranges. The local archive in `news_articles` and `price_bars` will
  still serve previously-fetched data.
- **First chart load is very slow** — FinBERT weights are downloading. The
  spinner says so. Wait it out the first time only.
- **Frontend says proxy error** — your backend isn't running on port 8000
  yet. Start it before reloading the page.
- **Predictions look the same for hours** — they're cached for 30 minutes.
  The 2-minute background loop will refresh them automatically when new news
  arrives.

