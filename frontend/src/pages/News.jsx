import { useEffect, useState } from "react";
import { api } from "../api.js";
import NewsList from "../components/NewsList.jsx";

export default function News() {
  const [items, setItems] = useState(null);
  const [filter, setFilter] = useState("all");
  const [err, setErr] = useState(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const d = await api("/api/home");
        if (!cancelled) setItems(d.news || []);
      } catch (e) {
        if (!cancelled) setErr(e.message);
      }
    }
    load();
    const id = setInterval(load, 5 * 60 * 1000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  if (err) return <div className="error">Error: {err}</div>;
  if (!items) return <div className="loading-bar">Loading market news…</div>;

  const filtered = filter === "all"
    ? items
    : items.filter((it) => (it.label || "").toLowerCase() === filter);

  const counts = {
    all: items.length,
    positive: items.filter((it) => it.label === "positive").length,
    neutral: items.filter((it) => it.label === "neutral").length,
    negative: items.filter((it) => it.label === "negative").length,
  };

  return (
    <>
      <header className="page-head">
        <h1>Market News</h1>
        <p className="muted-inline">Latest headlines with FinBERT sentiment classification.</p>
      </header>
      <div className="card">
        <div className="news-filter">
          {["all", "positive", "neutral", "negative"].map((k) => (
            <button
              key={k}
              className={filter === k ? "tab active" : "tab"}
              onClick={() => setFilter(k)}
            >
              {k[0].toUpperCase() + k.slice(1)} <span className="muted-inline">({counts[k]})</span>
            </button>
          ))}
        </div>
        <NewsList items={filtered} />
      </div>
    </>
  );
}
