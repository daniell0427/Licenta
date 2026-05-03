import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { api } from "../api.js";

export default function Watchlist({ onSelect }) {
  const { ticker: active } = useParams();
  const [rows, setRows] = useState([]);
  const [adding, setAdding] = useState("");
  const [busy, setBusy] = useState(false);

  async function load() {
    try {
      const data = await api("/api/watchlist");
      setRows(data.items || []);
    } catch {
      setRows([]);
    }
  }

  useEffect(() => { load(); }, []);

  async function add(e) {
    e.preventDefault();
    const t = adding.trim().toUpperCase();
    if (!t) return;
    setBusy(true);
    try {
      const data = await api("/api/watchlist", {
        method: "POST",
        body: JSON.stringify({ ticker: t }),
      });
      setRows(data.items);
      setAdding("");
    } finally {
      setBusy(false);
    }
  }

  async function remove(t) {
    setBusy(true);
    try {
      const data = await api(`/api/watchlist/${t}`, { method: "DELETE" });
      setRows(data.items);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="watchlist">
      <div className="watchlist-header">Watchlist</div>
      <ul>
        {rows.map((r) => {
          const positive = (r.change_pct ?? 0) >= 0;
          return (
            <li
              key={r.ticker}
              className={r.ticker === active ? "active" : ""}
              onClick={() => onSelect(r.ticker)}
            >
              <div className="wl-ticker">{r.ticker}</div>
              <div className="wl-price">{r.price != null ? `$${r.price.toFixed(2)}` : "—"}</div>
              <div className={`wl-pct ${positive ? "up" : "down"}`}>
                {r.change_pct != null ? `${positive ? "+" : ""}${r.change_pct.toFixed(2)}%` : ""}
              </div>
              <button
                className="wl-remove"
                onClick={(e) => { e.stopPropagation(); remove(r.ticker); }}
                title="Remove"
                disabled={busy}
              >×</button>
            </li>
          );
        })}
      </ul>
      <form onSubmit={add} className="wl-add">
        <input
          value={adding}
          onChange={(e) => setAdding(e.target.value)}
          placeholder="Add ticker..."
        />
        <button type="submit" disabled={busy}>+</button>
      </form>
    </div>
  );
}
