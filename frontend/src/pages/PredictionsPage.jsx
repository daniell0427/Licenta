import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api.js";

export default function PredictionsPage() {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      try {
        const wl = await api("/api/watchlist");
        const tickers = (wl.items || []).map((i) => i.ticker);
        if (cancelled) return;
        // Fire predictions in parallel; backend caches each for 5 min
        const results = await Promise.all(
          tickers.map((t) =>
            api(`/api/predict/${t}`)
              .then((p) => ({ ticker: t, predict: p, error: null }))
              .catch((e) => ({ ticker: t, predict: null, error: e.message }))
          )
        );
        if (!cancelled) setRows(results);
      } catch (e) {
        if (!cancelled) setErr(e.message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => { cancelled = true; };
  }, []);

  if (err) return <div className="error">Error: {err}</div>;

  return (
    <>
      <header className="page-head">
        <h1>Predictions</h1>
        <p className="muted-inline">FinBERT-fused LSTM direction forecasts across your watchlist.</p>
      </header>

      <div className="card">
        {loading && <div className="loading-bar">Computing predictions… (this can take a minute on first run)</div>}
        <table className="trending-table">
          <thead>
            <tr>
              <th>Ticker</th>
              <th>Short-term (1d)</th>
              <th>Confidence</th>
              <th>Long-term (5d)</th>
              <th>Confidence</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const st = r.predict?.short_term;
              const lt = r.predict?.long_term;
              return (
                <tr key={r.ticker}>
                  <td><Link to={`/ticker/${r.ticker}`} className="link">{r.ticker}</Link></td>
                  <td className={st?.direction === "up" ? "up" : st?.direction === "down" ? "down" : ""}>
                    {st ? (st.direction === "up" ? "▲ UP" : "▼ DOWN") : (r.error ? "—" : "…")}
                  </td>
                  <td className="num">{st ? `${(st.confidence * 100).toFixed(1)}%` : ""}</td>
                  <td className={lt?.direction === "up" ? "up" : lt?.direction === "down" ? "down" : ""}>
                    {lt ? (lt.direction === "up" ? "▲ UP" : "▼ DOWN") : ""}
                  </td>
                  <td className="num">{lt ? `${(lt.confidence * 100).toFixed(1)}%` : ""}</td>
                </tr>
              );
            })}
            {!loading && rows.length === 0 && (
              <tr><td colSpan="5" className="placeholder">Add tickers to your watchlist to see predictions.</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </>
  );
}
