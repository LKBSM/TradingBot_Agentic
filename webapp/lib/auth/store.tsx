'use client';

import * as React from 'react';
import * as api from './api-client';
import type { Account, LoginInput, RegisterInput } from './types';

interface AuthContextValue {
  account: Account | null;
  /** True until the initial /me probe resolves. */
  loading: boolean;
  isAuthenticated: boolean;
  isOwner: boolean;
  login: (input: LoginInput) => Promise<Account>;
  register: (input: RegisterInput) => Promise<Account>;
  logout: () => Promise<void>;
  refresh: () => Promise<void>;
}

const AuthContext = React.createContext<AuthContextValue | null>(null);

/**
 * Session provider. Probes `/api/auth/me` once on mount to hydrate the current
 * account from the HttpOnly session cookie, then exposes the auth actions. The
 * provider never stores the session itself — the cookie is the source of truth.
 */
export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [account, setAccount] = React.useState<Account | null>(null);
  const [loading, setLoading] = React.useState(true);

  const refresh = React.useCallback(async () => {
    try {
      setAccount(await api.fetchMe());
    } catch {
      // Network/5xx during probe → treat as logged out, don't crash the tree.
      setAccount(null);
    }
  }, []);

  React.useEffect(() => {
    let active = true;
    (async () => {
      try {
        const me = await api.fetchMe();
        if (active) setAccount(me);
      } catch {
        if (active) setAccount(null);
      } finally {
        if (active) setLoading(false);
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  const login = React.useCallback(async (input: LoginInput) => {
    const acc = await api.login(input);
    setAccount(acc);
    return acc;
  }, []);

  const register = React.useCallback(async (input: RegisterInput) => {
    const acc = await api.register(input);
    setAccount(acc);
    return acc;
  }, []);

  const logout = React.useCallback(async () => {
    try {
      await api.logout();
    } finally {
      setAccount(null);
    }
  }, []);

  const value = React.useMemo<AuthContextValue>(
    () => ({
      account,
      loading,
      isAuthenticated: account !== null,
      isOwner: account?.role === 'owner',
      login,
      register,
      logout,
      refresh,
    }),
    [account, loading, login, register, logout, refresh],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = React.useContext(AuthContext);
  if (ctx === null) {
    throw new Error('useAuth must be used within an <AuthProvider>');
  }
  return ctx;
}
