import { useNavigate } from "react-router-dom";
import { useAuth } from "../auth.jsx";

export default function Settings() {
  const { user, logout } = useAuth();
  const nav = useNavigate();

  function onLogout() {
    logout();
    nav("/login");
  }

  return (
    <>
      <header className="page-head">
        <h1>Settings</h1>
        <p className="muted-inline">Account and application preferences.</p>
      </header>

      <div className="card">
        <div className="card-title">Account</div>
        <div className="settings-row">
          <div className="settings-label">Username</div>
          <div className="settings-value">{user?.username}</div>
        </div>
        <div className="settings-row">
          <div className="settings-label">Email</div>
          <div className="settings-value">{user?.email}</div>
        </div>
        <div className="settings-row">
          <div className="settings-label">Member since</div>
          <div className="settings-value">
            {user?.created_at ? new Date(user.created_at).toLocaleDateString() : "—"}
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-title">About</div>
        <p className="muted-inline" style={{ lineHeight: 1.6 }}>
          SentiTrade combines FinBERT sentiment analysis of financial news with an LSTM
          model trained on price and technical indicators (RSI, MACD, EMA, Bollinger Bands).
          Predictions are short-term (1 day) and long-term (5 day) directional forecasts.
        </p>
        <p className="muted-inline" style={{ marginTop: 8 }}>
          This is a research prototype — not financial advice.
        </p>
      </div>

      <div className="card">
        <div className="card-title">Danger Zone</div>
        <button className="logout-btn" onClick={onLogout} style={{ maxWidth: 200 }}>
          Log out
        </button>
      </div>
    </>
  );
}
