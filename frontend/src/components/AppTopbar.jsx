import { useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../api.js";
import { useAuth } from "../auth.jsx";

function fmtLocalTs(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  if (isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit",
  });
}

// Clean, monochrome alert glyph: vertical bar + arrow / dot variant
function NotifGlyph({ n }) {
  const isUp = n.type === "price_move" ? n.direction === "up" : n.label === "positive";
  const isDown = n.type === "price_move" ? n.direction === "down" : n.label === "negative";
  const cls = isUp ? "up" : isDown ? "down" : "neutral";
  if (n.type === "price_move") {
    return <div className={`notif-glyph price ${cls}`}>{isUp ? "▲" : "▼"}</div>;
  }
  // News: small uppercase label chip
  return <div className={`notif-glyph news ${cls}`}>NEWS</div>;
}

export default function AppTopbar() {
  const { user, logout } = useAuth();
  const nav = useNavigate();
  const [notifs, setNotifs] = useState([]);
  const [unread, setUnread] = useState(0);
  const [showNotifs, setShowNotifs] = useState(false);
  const [showAccount, setShowAccount] = useState(false);
  const notifRef = useRef(null);
  const acctRef = useRef(null);

  async function load() {
    try {
      const d = await api("/api/notifications");
      setNotifs(d.items || []);
      setUnread(d.unread || 0);
    } catch {
      setNotifs([]);
      setUnread(0);
    }
  }

  useEffect(() => {
    load();
    const id = setInterval(load, 2 * 60 * 1000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    function onDoc(e) {
      if (notifRef.current && !notifRef.current.contains(e.target)) setShowNotifs(false);
      if (acctRef.current && !acctRef.current.contains(e.target)) setShowAccount(false);
    }
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, []);

  async function markRead(id, e) {
    if (e) { e.preventDefault(); e.stopPropagation(); }
    await api(`/api/notifications/${id}/read`, { method: "POST" }).catch(() => {});
    load();
  }
  async function dismiss(id, e) {
    e.preventDefault(); e.stopPropagation();
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

  function onLogout() {
    logout();
    nav("/login");
  }

  return (
    <header className="app-topbar">
      <Link to="/" className="topbar-logo">📊 SentiTrade</Link>

      <div className="topbar-right">
        <div className="topbar-popover-wrap" ref={notifRef}>
          <button
            className="topbar-icon-btn"
            onClick={() => { setShowNotifs((s) => !s); setShowAccount(false); }}
            title="Notifications"
          >
            🔔
            {unread > 0 && <span className="notif-dot">{unread}</span>}
          </button>
          {showNotifs && (
            <div className="popover">
              <div className="popover-head">
                <span>Notifications</span>
                <span className="muted-inline" style={{ fontSize: 11 }}>
                  {unread} unread · {notifs.length} total
                </span>
              </div>
              {notifs.length > 0 && (
                <div className="popover-actions">
                  <button className="popover-action" onClick={markAllRead} disabled={unread === 0}>
                    Mark all read
                  </button>
                  <button className="popover-action danger" onClick={clearAll}>
                    Clear all
                  </button>
                </div>
              )}
              {notifs.length === 0 && (
                <div className="popover-empty">
                  No alerts. Watchlist tickers with ≥5% moves or breaking news will appear here.
                </div>
              )}
              {notifs.map((n) => (
                <Link
                  key={n.id}
                  to={`/ticker/${n.ticker}`}
                  state={n.type === "news" ? { highlightUrl: n.url, highlightTitle: n.title } : undefined}
                  className={`notif-item ${n.severity || ""} ${n.read ? "read" : "unread"}`}
                  onClick={() => { setShowNotifs(false); if (!n.read) markRead(n.id); }}
                >
                  <NotifGlyph n={n} />
                  <div className="notif-body">
                    <div className="notif-title">{n.title}</div>
                    <div className="notif-sub">{n.subtitle} · {fmtLocalTs(n.created_at)}</div>
                  </div>
                  <div className="notif-actions">
                    {!n.read && (
                      <button
                        className="notif-action-btn"
                        onClick={(e) => markRead(n.id, e)}
                        title="Mark read"
                      >✓</button>
                    )}
                    <button
                      className="notif-action-btn danger"
                      onClick={(e) => dismiss(n.id, e)}
                      title="Dismiss"
                    >×</button>
                  </div>
                </Link>
              ))}
              {notifs.length > 0 && (
                <Link to="/notifications" className="popover-link" onClick={() => setShowNotifs(false)}>
                  View all notifications →
                </Link>
              )}
            </div>
          )}
        </div>

        <div className="topbar-popover-wrap" ref={acctRef}>
          <button
            className="topbar-account"
            onClick={() => { setShowAccount((s) => !s); setShowNotifs(false); }}
          >
            <div className="avatar small">{user?.username?.[0]?.toUpperCase() ?? "?"}</div>
            <span className="acct-name">{user?.username}</span>
            <span className="caret">▾</span>
          </button>
          {showAccount && (
            <div className="popover small">
              <div className="popover-head">
                <div>
                  <div className="user-name">{user?.username}</div>
                  <div className="user-mail">{user?.email}</div>
                </div>
              </div>
              <Link to="/settings" className="popover-link" onClick={() => setShowAccount(false)}>
                ⚙️ Settings
              </Link>
              <button className="popover-link danger" onClick={onLogout}>
                ↪ Log out
              </button>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
