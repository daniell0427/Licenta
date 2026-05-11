import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider, useAuth } from "./auth.jsx";
import Login from "./pages/Login.jsx";
import Register from "./pages/Register.jsx";
import Home from "./pages/Home.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import Layout from "./pages/Layout.jsx";
import News from "./pages/News.jsx";
import PredictionsPage from "./pages/PredictionsPage.jsx";
import Settings from "./pages/Settings.jsx";
import NotificationsPage from "./pages/NotificationsPage.jsx";
import StockBrowser from "./pages/StockBrowser.jsx";
import "./styles.css";

function Protected({ children }) {
  const { user, loading } = useAuth();
  if (loading) return <div className="full-center muted">Loading…</div>;
  if (!user) return <Navigate to="/login" replace />;
  return children;
}

function GuestOnly({ children }) {
  const { user, loading } = useAuth();
  if (loading) return <div className="full-center muted">Loading…</div>;
  if (user) return <Navigate to="/" replace />;
  return children;
}

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<GuestOnly><Login /></GuestOnly>} />
          <Route path="/register" element={<GuestOnly><Register /></GuestOnly>} />
          <Route element={<Protected><Layout /></Protected>}>
            <Route path="/" element={<Home />} />
            <Route path="/news" element={<News />} />
            <Route path="/predictions" element={<PredictionsPage />} />
            <Route path="/notifications" element={<NotificationsPage />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="/ticker/:ticker" element={<Dashboard />} />
            <Route path="/stocks" element={<StockBrowser />} />
          </Route>
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  </React.StrictMode>
);
