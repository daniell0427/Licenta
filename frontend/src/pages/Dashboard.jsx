import { useEffect, useState } from "react";
import { useParams, useLocation } from "react-router-dom";
import { api } from "../api.js";
import PriceChart from "../components/PriceChart.jsx";
import NewsList from "../components/NewsList.jsx";
import Predictions from "../components/Predictions.jsx";
import SentimentGauge from "../components/SentimentGauge.jsx";
import PeriodSummary from "../components/PeriodSummary.jsx";
import { Skeleton } from "../components/Loading.jsx";

// Line mode: timeframe = visible window. Chart fits to this period.
const LINE_TIMEFRAMES = [
  { id: "1D",  period: "1d",  interval: "5m"  },
  { id: "5D",  period: "5d",  interval: "30m" },
  { id: "1M",  period: "1mo", interval: "1d"  },
  { id: "3M",  period: "3mo", interval: "1d"  },
  { id: "6M",  period: "6mo", interval: "1d"  },
  { id: "1Y",  period: "1y",  interval: "1d"  },
  { id: "5Y",  period: "5y",  interval: "1wk" },
  { id: "MAX", period: "max", interval: "1mo" },
];

// Candle mode: timeframe = candle size. We load as much history as Yahoo reliably
// returns for that interval. The chart does NOT auto-fit so the user can scroll back.
const CANDLE_TIMEFRAMES = [
  { id: "5m",  period: "1mo", interval: "5m"  },
  { id: "15m", period: "1mo", interval: "15m" },
  { id: "30m", period: "1mo", interval: "30m" },
  { id: "1h",  period: "2y",  interval: "1h"  },
  { id: "1D",  period: "10y", interval: "1d"  },
  { id: "1W",  period: "10y", interval: "1wk" },
  { id: "1M",  period: "max", interval: "1mo" },
];

export default function Dashboard() {
  const { ticker } = useParams();
  const location = useLocation();
  const highlightUrl = location.state?.highlightUrl || null;
  const highlightTitle = location.state?.highlightTitle || null;
  const [tab, setTab] = useState("chart");
  const [chartType, setChartType] = useState("candle");
  const [lineTf, setLineTf] = useState("1Y");
  const [candleTf, setCandleTf] = useState("1D");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [data, setData] = useState({ price: null, news: null, predict: null });
  const [quote, setQuote] = useState(null);

  const tfList = chartType === "line" ? LINE_TIMEFRAMES : CANDLE_TIMEFRAMES;
  const currentTfId = chartType === "line" ? lineTf : candleTf;
  const setTf = chartType === "line" ? setLineTf : setCandleTf;
  const tfDef = tfList.find((t) => t.id === currentTfId) || tfList[0];
  // Intraday view = sub-day intervals; affects crosshair time format + session display
  const intraday = ["1D", "5D", "5m", "15m", "30m", "1h"].includes(currentTfId);

  // Reload news + predictions when ticker changes, then poll periodically.
  // Backend already caches both for 5 min, so polling more often than that is wasted
  // work. We refresh every 5 min so the user sees fresh predictions over a long session.
  useEffect(() => {
    let cancelled = false;
    let timer = null;

    async function load(showSpinner) {
      if (showSpinner) {
        setLoading(true);
        setError(null);
        setData((d) => ({ ...d, news: null, predict: null }));
      }
      try {
        const [news, predict] = await Promise.all([
          api(`/api/news/${ticker}?limit=50`),
          api(`/api/predict/${ticker}`).catch(() => null),
        ]);
        if (!cancelled) setData((d) => ({ ...d, news, predict }));
      } catch (e) {
        if (!cancelled && showSpinner) setError(e.message);
      } finally {
        if (!cancelled && showSpinner) setLoading(false);
      }
    }

    load(true);
    // Poll every 60s. Backend caches news in DB and the 1-min background loop
    // pulls fresh articles from upstream, so a minute is the most useful interval.
    timer = setInterval(() => load(false), 60 * 1000);
    return () => { cancelled = true; if (timer) clearInterval(timer); };
  }, [ticker]);

  // Reload price series when ticker or timeframe changes
  useEffect(() => {
    let cancelled = false;
    async function loadPrice() {
      try {
        const price = await api(
          `/api/price/${ticker}?period=${tfDef.period}&interval=${tfDef.interval}`
        );
        if (!cancelled) setData((d) => ({ ...d, price }));
      } catch (e) {
        if (!cancelled) setError(e.message);
      }
    }
    loadPrice();
    return () => { cancelled = true; };
  }, [ticker, currentTfId, chartType]);

  // When the user lands on this page from a notification, scroll the matching
  // headline into view and apply a temporary "flash" highlight class so they
  // can see exactly which article triggered the alert.
  useEffect(() => {
    if (!data.news || !data.news.items || (!highlightUrl && !highlightTitle)) return;
    const selector = highlightUrl
      ? `[data-news-url="${CSS.escape(highlightUrl)}"]`
      : `[data-news-title="${CSS.escape(highlightTitle)}"]`;
    const t = setTimeout(() => {
      const el = document.querySelector(selector);
      if (!el) return;
      el.scrollIntoView({ behavior: "smooth", block: "center" });
      el.classList.add("flash-highlight");
      setTimeout(() => el.classList.remove("flash-highlight"), 4000);
    }, 100); // wait for DOM to render
    return () => clearTimeout(t);
  }, [data.news, highlightUrl, highlightTitle]);

  // Live quote polling — every 15s, includes pre/post-market
  useEffect(() => {
    let cancelled = false;
    async function fetchQuote() {
      try {
        const q = await api(`/api/quote/${ticker}`);
        if (!cancelled) setQuote(q);
      } catch {
        /* keep last quote */
      }
    }
    setQuote(null);
    fetchQuote();
    const id = setInterval(fetchQuote, 15000);
    return () => { cancelled = true; clearInterval(id); };
  }, [ticker]);

  const sent = data.news?.sentiment_score ?? 0;
  const last = quote?.price
    ?? (data.price?.candles?.length ? data.price.candles.at(-1).close : null);
  const dayChg = quote?.change_pct ?? null;

  return (
    <>
      <header className="topbar">
        <nav className="tabs">
          {["chart", "predictions", "events", "notify"].map((t) => (
            <button key={t} className={tab === t ? "tab active" : "tab"} onClick={() => setTab(t)}>
              {t[0].toUpperCase() + t.slice(1)}
            </button>
          ))}
        </nav>
        <div className="ticker-display">
          <span className="t-symbol">{ticker}</span>
          {last != null && (
            <>
              <span className="t-price">${last.toFixed(2)}</span>
              {dayChg !== null && (
                <span className={dayChg >= 0 ? "t-change up" : "t-change down"}>
                  {dayChg >= 0 ? "+" : ""}{dayChg.toFixed(2)}%
                </span>
              )}
              {quote?.session === "pre" && <span className="session-badge pre">Pre-Market</span>}
              {quote?.session === "post" && <span className="session-badge post">After Hours</span>}
            </>
          )}
        </div>
      </header>

      {error && <div className="error">Error: {error}</div>}
      {loading && <div className="loading-bar">Loading {ticker}…</div>}

      {tab === "chart" && (
        <>
          <div className="row top-row">
            <div className="card chart-card">
              <div className="chart-controls">
                <div className="tf-group">
                  {tfList.map((t) => (
                    <button
                      key={t.id}
                      className={currentTfId === t.id ? "tf-btn active" : "tf-btn"}
                      onClick={() => setTf(t.id)}
                    >{t.id}</button>
                  ))}
                </div>
                <div className="ctype-group">
                  <button
                    className={chartType === "candle" ? "ctype-btn active" : "ctype-btn"}
                    onClick={() => setChartType("candle")}
                  >Candles</button>
                  <button
                    className={chartType === "line" ? "ctype-btn active" : "ctype-btn"}
                    onClick={() => setChartType("line")}
                  >Line</button>
                </div>
              </div>
              {data.price ? (
                <>
                  <PriceChart
                    candles={data.price.candles}
                    chartType={chartType}
                    fit={chartType === "line"}
                    intraday={intraday}
                  />
                  <PeriodSummary
                    candles={data.price.candles}
                    quote={quote}
                    chartType={chartType}
                  />
                </>
              ) : (
                <div style={{ padding: "60px 20px", textAlign: "center" }}>
                  <Skeleton height={420} style={{ marginBottom: 8 }} />
                  <div className="muted-inline" style={{ fontSize: 12 }}>
                    Loading {ticker} price history…
                  </div>
                </div>
              )}
            </div>
            <div className="card gauge-card">
              <div className="card-title">Sentiment Gauge</div>
              {data.news ? (
                <>
                  <SentimentGauge score={sent} />
                  <div className="muted small" style={{ marginTop: 8, textAlign: "center" }}>
                    Aggregated FinBERT score across {data.news?.items?.length ?? 0} headlines
                  </div>
                </>
              ) : (
                <div style={{ padding: 20, textAlign: "center" }}>
                  <Skeleton height={140} width={220} style={{ margin: "0 auto 10px" }} />
                  <div className="muted-inline" style={{ fontSize: 11 }}>Scoring headlines…</div>
                </div>
              )}
            </div>
          </div>
          <div className="card">
            <div className="card-title">News Feed</div>
            {data.news === null ? (
              <div className="loading-bar">
                Fetching latest news for {ticker}…
                <div className="muted-inline" style={{ fontSize: 11, marginTop: 4 }}>
                  First load runs FinBERT on every new headline (~10–20s).
                  Subsequent loads pull from the local archive — instant.
                </div>
              </div>
            ) : (data.news.items?.length ? (
              <NewsList
                items={data.news.items}
                highlightUrl={highlightUrl}
                highlightTitle={highlightTitle}
              />
            ) : (
              <div className="placeholder">No news yet for {ticker}.</div>
            ))}
          </div>
        </>
      )}
      {tab === "predictions" && (
        data.predict
          ? <Predictions predict={data.predict} />
          : <div className="card loading-bar">
              Training LSTM and computing forecasts for {ticker}…
              <div className="muted-inline" style={{ fontSize: 11, marginTop: 4 }}>
                First run trains the model (~30–60s). Predictions are then cached for 30 minutes.
              </div>
            </div>
      )}
      {tab === "events" && <div className="card placeholder">Earnings & macro events — coming soon.</div>}
      {tab === "notify" && <div className="card placeholder">Notification rules — coming soon.</div>}
    </>
  );
}
