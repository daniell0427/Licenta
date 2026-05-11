import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api.js";

export default function StockBrowser() {
  const [sectors, setSectors] = useState([]);
  const [activeSector, setActiveSector] = useState("");
  const [stocks, setStocks] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [filter, setFilter] = useState("");
  const [loading, setLoading] = useState(true);
  const LIMIT = 50;

  useEffect(() => {
    api("/api/sectors").then(d => setSectors(d.sectors || [])).catch(() => {});
  }, []);

  useEffect(() => {
    setLoading(true);
    const params = new URLSearchParams({ page, limit: LIMIT });
    if (activeSector) params.set("sector", activeSector);
    api(`/api/stocks?${params}`)
      .then(d => { setStocks(d.items || []); setTotal(d.total || 0); })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [activeSector, page]);

  const displayed = filter
    ? stocks.filter(s =>
        s.ticker.includes(filter.toUpperCase()) ||
        s.name?.toLowerCase().includes(filter.toLowerCase()))
    : stocks;

  const totalPages = Math.ceil(total / LIMIT);

  return (
    <>
      <header className="page-head">
        <h1>Browse Stocks</h1>
        <p className="muted-inline">{total} stocks in the training universe</p>
      </header>

      <div className="card">
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 16 }}>
          <button
            className={`sector-btn ${activeSector === "" ? "active" : ""}`}
            onClick={() => { setActiveSector(""); setPage(1); }}
          >All</button>
          {sectors.map(s => (
            <button
              key={s}
              className={`sector-btn ${activeSector === s ? "active" : ""}`}
              onClick={() => { setActiveSector(s); setPage(1); }}
            >{s}</button>
          ))}
        </div>

        <input
          className="filter-input"
          placeholder="Filter by ticker or name..."
          value={filter}
          onChange={e => setFilter(e.target.value)}
          style={{ marginBottom: 16, width: "100%", padding: "6px 12px",
                   background: "#1e1e2e", border: "1px solid #333",
                   color: "white", borderRadius: 6 }}
        />

        {loading ? (
          <div className="muted">Loading…</div>
        ) : (
          <table className="trending-table">
            <thead>
              <tr><th>Ticker</th><th>Company</th><th>Sector</th></tr>
            </thead>
            <tbody>
              {displayed.map(s => (
                <tr key={s.ticker}>
                  <td><Link to={`/ticker/${s.ticker}`} className="link">{s.ticker}</Link></td>
                  <td>{s.name || "—"}</td>
                  <td className="muted-inline" style={{ fontSize: 12 }}>{s.sector || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}

        {totalPages > 1 && (
          <div style={{ display: "flex", gap: 8, marginTop: 16, alignItems: "center" }}>
            <button className="btn-secondary" onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1}>← Prev</button>
            <span className="muted-inline">Page {page} of {totalPages}</span>
            <button className="btn-secondary" onClick={() => setPage(p => Math.min(totalPages, p + 1))} disabled={page === totalPages}>Next →</button>
          </div>
        )}
      </div>
    </>
  );
}
