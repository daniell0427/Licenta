import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../auth.jsx";

export default function Login() {
  const { login } = useAuth();
  const nav = useNavigate();
  const [form, setForm] = useState({ id: "", password: "" });
  const [err, setErr] = useState(null);
  const [loading, setLoading] = useState(false);

  async function submit(e) {
    e.preventDefault();
    setErr(null);
    setLoading(true);
    try {
      await login(form.id, form.password);
      nav("/");
    } catch (e) {
      setErr(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="auth-page">
      <div className="auth-card">
        <h1 className="auth-title">📊 SentiTrade</h1>
        <p className="auth-sub">Sign in to your account</p>
        <form onSubmit={submit} className="auth-form">
          <label>Email or username</label>
          <input
            value={form.id}
            onChange={(e) => setForm({ ...form, id: e.target.value })}
            required
            autoFocus
          />
          <label>Password</label>
          <input
            type="password"
            value={form.password}
            onChange={(e) => setForm({ ...form, password: e.target.value })}
            required
          />
          {err && <div className="auth-error">{err}</div>}
          <button disabled={loading}>{loading ? "Signing in…" : "Sign in"}</button>
        </form>
        <div className="auth-foot">
          No account? <Link to="/register">Create one</Link>
        </div>
      </div>
    </div>
  );
}
