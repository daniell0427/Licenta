import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api.js";

function fmtLocalDateTime(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  if (isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    year: "numeric", month: "short", day: "2-digit",
    hour: "2-digit", minute: "2-digit",
  });
}

function NotifGlyph({ n }) {
  const isUp = n.type === "price_move" ? n.direction === "up" : n.label === "positive";
  const isDown = n.type === "price_move" ? n.direction === "down" : n.label === "negative";
  const cls = isUp ? "up" : isDown ? "down" : "neutral";
  if (n.type === "price_move") {
    return <div className={`notif-glyph price big ${cls}`}>{isUp ? "▲" : "▼"}</div>;
  }
  return <div className={`notif-glyph news big ${cls}`}>NEWS</div>;
}

export default function NotificationsPage() {
  const [data, setData] = useState(null);
  const [filter, setFilter] = useState("all"); // all | unread | price_move | news

  async function load() {
    try {
      const d = await api("/api/notifications");
      setData(d);
    } catch {
      setData({ items: [], unread: 0 });
    }
  }

  useEffect(() => {
    load();
    const id = setInterval(load, 2 * 60 * 1000);
    return () => clearInterval(id);
  }, []);

  async function markRead(id) {
    await api(`/api/notifications/${id}/read`, { method: "POST" }).catch(() => {});
    load();
  }
  async function dismiss(id) {
    await api(`/api/notifications/${id}`, { method: "DELETE" }).catch(() => {});
    load();
  }
  async function markAllRead() {
    await api("/api/notifications/read-all", { method: "POST" }).catch(() => {});
    load();
  }
  async function clearAll() {
    if (!confirm("Dismiss all notifications?")) return;
    await api("/api/notifications", { method: "DELETE" }).catch(() => {});
    load();
  }

  if (!data) return <div className="loading-bar">Loading notifications…</div>;

  const items = data.items || [];
  const filtered = items.filter((n) => {
    if (filter === "all") return true;
    if (filter === "unread") return !n.read;
    return n.type === filter;
  });

  return (
    <>
      <header className="page-head">
        <h1>Notifications</h1>
        <p className="muted-inline">
          Alerts for your watchlist — price moves ≥5% and very-high-confidence news.
        </p>
      </header>

      <div className="card">
        <div className="notif-toolbar">
          <div className="news-filter">
            {[
              ["all", "All"],
              ["unread", "Unread"],
              ["price_move", "Price moves"],
              ["news", "News"],
            ].map(([k, label]) => (
              <button
                key={k}
                className={filter === k ? "tab active" : "tab"}
                onClick={() => setFilter(k)}
              >
                {label}
              </button>
            ))}
          </div>
          <div className="notif-bulk-actions">
            <button className="popover-action" onClick={markAllRead} disabled={data.unread === 0}>
              Mark all read
            </button>
            <button className="popover-action danger" onClick={clearAll} disabled={items.length === 0}>
              Clear all
            </button>
          </div>
        </div>

        {filtered.length === 0 && (
          <div className="placeholder">No notifications match the current filter.</div>
        )}

        <div className="notif-page-list">
          {filtered.map((n) => (
            <div key={n.id} className={`notif-page-item ${n.severity || ""} ${n.read ? "read" : "unread"}`}>
              <NotifGlyph n={n} />
              <div className="notif-body">
                <div className="notif-title">
                  <Link
                    to={`/ticker/${n.ticker}`}
                    state={n.type === "news" ? { highlightUrl: n.url, highlightTitle: n.title } : undefined}
                    className="link"
                  >
                    {n.ticker}
                  </Link>
                  {" — "}
                  {n.title}
                </div>
                <div className="notif-sub">{n.subtitle}</div>
                <div className="notif-meta">
                  {fmtLocalDateTime(n.created_at)}
                  {n.url && (
                    <>
                      {" · "}
                      <a href={n.url} target="_blank" rel="noreferrer" className="link">
                        Read article ↗
                      </a>
                    </>
                  )}
                </div>
              </div>
              <div className="notif-actions">
                {!n.read && (
                  <button className="popover-action" onClick={() => markRead(n.id)}>
                    Mark read
                  </button>
                )}
                <button className="popover-action danger" onClick={() => dismiss(n.id)}>
                  Dismiss
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </>
  );
}
