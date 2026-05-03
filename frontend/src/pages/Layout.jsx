import { useState } from "react";
import { NavLink, Outlet, useNavigate } from "react-router-dom";
import Watchlist from "../components/Watchlist.jsx";
import AppTopbar from "../components/AppTopbar.jsx";

export default function Layout() {
  const nav = useNavigate();
  const [sidebarOpen, setSidebarOpen] = useState(true);

  return (
    <div className={`app ${sidebarOpen ? "sb-open" : "sb-closed"}`}>
      <aside className="sidebar">
        <div className="sidebar-brand">SentiTrade</div>
        <div className="sidebar-section-label">Navigation</div>
        <nav className="side-nav">
          <NavLink to="/" end className={({ isActive }) => isActive ? "side-link active" : "side-link"}>
            Dashboard
          </NavLink>
          <NavLink to="/news" className={({ isActive }) => isActive ? "side-link active" : "side-link"}>
            Market News
          </NavLink>
          <NavLink to="/predictions" className={({ isActive }) => isActive ? "side-link active" : "side-link"}>
            Predictions
          </NavLink>
          <NavLink to="/notifications" className={({ isActive }) => isActive ? "side-link active" : "side-link"}>
            Notifications
          </NavLink>
          <NavLink to="/settings" className={({ isActive }) => isActive ? "side-link active" : "side-link"}>
            Settings
          </NavLink>
        </nav>
        <Watchlist onSelect={(t) => nav(`/ticker/${t}`)} />
      </aside>
      <button
        className={`sidebar-toggle ${sidebarOpen ? "open" : "closed"}`}
        onClick={() => setSidebarOpen((s) => !s)}
        title={sidebarOpen ? "Hide sidebar" : "Show sidebar"}
      >
        {sidebarOpen ? "‹" : "›"}
      </button>
      <div className="main-wrap">
        <AppTopbar />
        <main className="main">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
