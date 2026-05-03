import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api.js";
import NewsList from "../components/NewsList.jsx";
import { Skeleton, SkeletonTable } from "../components/Loading.jsx";

function fmtPrice(v) {
  if (v == null) return "—";
  if (Math.abs(v) >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 2 });
  if (Math.abs(v) >= 1) return v.toFixed(2);
  return v.toFixed(4);
}

function fmtPct(p) {
  if (p == null) return "";
  const sign = p >= 0 ? "+" : "";
  return `${sign}${p.toFixed(2)}%`;
}

// Compact card used across markets snapshot grids
function QuoteCard({ q, prefix = "" }) {
  const up = (q.change_pct ?? 0) >= 0;
  return (
    <div className="qcard">
      <div className="qcard-name">{q.name}</div>
      <div className="qcard-price">{prefix}{fmtPrice(q.price)}</div>
      <div className={`qcard-chg ${up ? "up" : "down"}`}>
        {up ? "▲" : "▼"} {fmtPct(q.change_pct)}
      </div>
    </div>
  );
}

function MoversTable({ rows, title }) {
  return (
    <div className="card">
      <div className="card-title">{title}</div>
      <table className="trending-table">
        <thead>
          <tr><th>Ticker</th><th className="num">Price</th><th className="num">Change</th></tr>
        </thead>
        <tbody>
          {rows.map((t) => (
            <tr key={t.ticker}>
              <td><Link to={`/ticker/${t.ticker}`} className="link">{t.ticker}</Link></td>
              <td className="num">${fmtPrice(t.price)}</td>
              <td className={`num ${t.change_pct >= 0 ? "up" : "down"}`}>
                {fmtPct(t.change_pct)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function SectorBar({ sectors }) {
  // Horizontal bar sized by absolute change_pct
  const max = Math.max(...sectors.map((s) => Math.abs(s.change_pct || 0)), 1);
  return (
    <div className="card">
      <div className="card-title">Sector Performance · SPDR ETFs</div>
      <div className="sector-list">
        {sectors.map((s) => {
          const up = (s.change_pct ?? 0) >= 0;
          const w = (Math.abs(s.change_pct || 0) / max) * 100;
          return (
            <div key={s.ticker} className="sector-row">
              <div className="sector-name">{s.name}</div>
              <div className="sector-bar-wrap">
                <div
                  className={`sector-bar ${up ? "up" : "down"}`}
                  style={{ width: `${w}%` }}
                />
              </div>
              <div className={`sector-pct ${up ? "up" : "down"}`}>
                {fmtPct(s.change_pct)}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function MarketsBlock({ title, items, prefix = "" }) {
  return (
    <div className="card">
      <div className="card-title">{title}</div>
      <div className="qcard-grid">
        {items.map((q) => (
          <QuoteCard key={q.ticker} q={q} prefix={prefix} />
        ))}
      </div>
    </div>
  );
}

export default function Home() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const d = await api("/api/home");
        if (!cancelled) setData(d);
      } catch (e) {
        if (!cancelled) setErr(e.message);
      }
    }
    load();
    const id = setInterval(load, 60_000); // refresh every minute
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  if (err) return <div className="error">Error: {err}</div>;
  if (!data) {
    // Skeleton layout that mirrors the real one so the page doesn't shift in
    return (
      <>
        <header className="page-head">
          <h1>Market Overview</h1>
          <p className="muted-inline">Loading indices, sectors, and global markets…</p>
        </header>
        <section className="indices-row">
          {Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="card index-card">
              <Skeleton height={11} width={60} style={{ marginBottom: 8 }} />
              <Skeleton height={22} width={100} style={{ marginBottom: 6 }} />
              <Skeleton height={13} width={70} />
            </div>
          ))}
        </section>
        <div className="row two-col">
          <div className="card">
            <div className="card-title">Sector Performance</div>
            {Array.from({ length: 8 }).map((_, i) => (
              <Skeleton key={i} height={18} style={{ marginBottom: 6 }} />
            ))}
          </div>
          <div>
            <div className="card">
              <div className="card-title">Top Gainers</div>
              <SkeletonTable rows={5} cols={3} />
            </div>
            <div className="card">
              <div className="card-title">Top Losers</div>
              <SkeletonTable rows={5} cols={3} />
            </div>
          </div>
        </div>
      </>
    );
  }

  return (
    <>
      <header className="page-head">
        <h1>Market Overview</h1>
        <p className="muted-inline">Indices, sectors, and global markets at a glance.</p>
      </header>

      {/* Indices ribbon */}
      <section className="indices-row">
        {data.indices.map((idx) => (
          <div key={idx.ticker} className="card index-card">
            <div className="idx-name">{idx.name}</div>
            <div className="idx-price">{fmtPrice(idx.price)}</div>
            <div className={idx.change_pct >= 0 ? "idx-change up" : "idx-change down"}>
              {idx.change_pct >= 0 ? "▲" : "▼"} {Math.abs(idx.change_pct).toFixed(2)}%
            </div>
          </div>
        ))}
      </section>

      {/* Sectors + Movers */}
      <div className="row two-col">
        <SectorBar sectors={data.sectors} />
        <div>
          <MoversTable rows={data.gainers} title="Top Gainers · S&P movers" />
          <MoversTable rows={data.losers} title="Top Losers · S&P movers" />
        </div>
      </div>

      {/* Commodities + Market News */}
      <div className="row two-col">
        <MarketsBlock title="Commodities" items={data.commodities} prefix="$" />
        <div className="card">
          <div className="card-title">Market News</div>
          <NewsList items={data.news} />
        </div>
      </div>
    </>
  );
}
