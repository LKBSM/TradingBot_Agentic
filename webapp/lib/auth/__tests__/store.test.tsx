import { render, screen, waitFor } from '@/components/test-utils';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import * as api from '../api-client';
import { AuthProvider, useAuth } from '../store';
import type { Account } from '../types';

vi.mock('../api-client');

const ACCOUNT: Account = {
  id: 1,
  username: 'alice',
  email: 'alice@example.com',
  role: 'user',
  age_confirmed: true,
  created_at: '2026-01-01T00:00:00',
  consents: [],
};

function Probe() {
  const { loading, account, probeFailed, isAuthenticated } = useAuth();
  return (
    <div>
      <span data-testid="loading">{String(loading)}</span>
      <span data-testid="account">{account ? account.username : 'null'}</span>
      <span data-testid="probeFailed">{String(probeFailed)}</span>
      <span data-testid="authed">{String(isAuthenticated)}</span>
    </div>
  );
}

describe('AuthProvider — AUTH-03 probe state', () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it('a network/5xx probe failure does NOT log the user out (probeFailed)', async () => {
    vi.mocked(api.fetchMe).mockRejectedValue(new Error('network'));
    render(
      <AuthProvider>
        <Probe />
      </AuthProvider>,
    );
    await waitFor(() =>
      expect(screen.getByTestId('loading').textContent).toBe('false'),
    );
    expect(screen.getByTestId('probeFailed').textContent).toBe('true');
    // account stays null here (nothing to preserve on first load) but the key
    // point is it is NOT treated as a confirmed logout.
    expect(screen.getByTestId('authed').textContent).toBe('false');
  });

  it('a real 401 (fetchMe → null) is a clean logged-out state', async () => {
    vi.mocked(api.fetchMe).mockResolvedValue(null);
    render(
      <AuthProvider>
        <Probe />
      </AuthProvider>,
    );
    await waitFor(() =>
      expect(screen.getByTestId('loading').textContent).toBe('false'),
    );
    expect(screen.getByTestId('probeFailed').textContent).toBe('false');
    expect(screen.getByTestId('account').textContent).toBe('null');
  });

  it('a successful probe hydrates the account', async () => {
    vi.mocked(api.fetchMe).mockResolvedValue(ACCOUNT);
    render(
      <AuthProvider>
        <Probe />
      </AuthProvider>,
    );
    await waitFor(() =>
      expect(screen.getByTestId('account').textContent).toBe('alice'),
    );
    expect(screen.getByTestId('probeFailed').textContent).toBe('false');
    expect(screen.getByTestId('authed').textContent).toBe('true');
  });
});
