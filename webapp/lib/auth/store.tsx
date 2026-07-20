'use client';

import * as React from 'react';
import * as api from './api-client';
import type { Account, LoginInput, RegisterInput } from './types';

interface AuthContextValue {
  account: Account | null;
  /** True until the initial /me probe resolves. */
  loading: boolean;
  /**
   * True when the LAST /me probe failed for a NON-401 reason (network, 5xx).
   * Distinct from "logged out": we couldn't reach the server, so `account` is
   * left as-is and callers should offer a retry rather than bounce a possibly
   * valid user to login (AUTH-03).
   */
  probeFailed: boolean;
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
  const [probeFailed, setProbeFailed] = React.useState(false);

  const refresh = React.useCallback(async () => {
    try {
      // fetchMe returns null ONLY on a real 401 (anonymous); network/5xx throw.
      setAccount(await api.fetchMe());
      setProbeFailed(false);
    } catch {
      // We couldn't reach the server — do NOT null out the account (that would
      // log a valid user out on a transient blip). Flag it so the UI can retry.
      setProbeFailed(true);
    }
  }, []);

  React.useEffect(() => {
    let active = true;
    (async () => {
      try {
        const me = await api.fetchMe();
        if (active) {
          setAccount(me);
          setProbeFailed(false);
        }
      } catch {
        if (active) setProbeFailed(true);
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
    setProbeFailed(false);
    return acc;
  }, []);

  const register = React.useCallback(async (input: RegisterInput) => {
    const acc = await api.register(input);
    setAccount(acc);
    setProbeFailed(false);
    return acc;
  }, []);

  const logout = React.useCallback(async () => {
    try {
      await api.logout();
    } finally {
      setAccount(null);
      setProbeFailed(false);
    }
  }, []);

  const value = React.useMemo<AuthContextValue>(
    () => ({
      account,
      loading,
      probeFailed,
      isAuthenticated: account !== null,
      isOwner: account?.role === 'owner',
      login,
      register,
      logout,
      refresh,
    }),
    [account, loading, probeFailed, login, register, logout, refresh],
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
