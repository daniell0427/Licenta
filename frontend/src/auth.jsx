import { createContext, useContext, useEffect, useState } from "react";
import { api, getToken, setToken } from "./api.js";

const AuthCtx = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!getToken()) { setLoading(false); return; }
    api("/api/auth/me")
      .then(setUser)
      .catch(() => setToken(null))
      .finally(() => setLoading(false));
  }, []);

  async function login(email_or_username, password) {
    const data = await api("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({ email_or_username, password }),
    });
    setToken(data.access_token);
    setUser(data.user);
  }

  async function register(email, username, password) {
    const data = await api("/api/auth/register", {
      method: "POST",
      body: JSON.stringify({ email, username, password }),
    });
    setToken(data.access_token);
    setUser(data.user);
  }

  function logout() {
    setToken(null);
    setUser(null);
  }

  return (
    <AuthCtx.Provider value={{ user, loading, login, register, logout }}>
      {children}
    </AuthCtx.Provider>
  );
}

export function useAuth() {
  return useContext(AuthCtx);
}
