import { render, screen, fireEvent, waitFor } from '@/components/test-utils';
import { afterEach, describe, expect, it, vi } from 'vitest';

// The account panel's logout must invalidate the Router Cache before navigating
// home, so a cached authenticated page never survives the logout (inverse of
// the first-login reliability fix — see LoginForm).
const h = vi.hoisted(() => ({
  push: vi.fn(),
  replace: vi.fn(),
  refresh: vi.fn(),
  logout: vi.fn(),
  refreshAuth: vi.fn(),
}));

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: h.push, replace: h.replace, refresh: h.refresh }),
}));
vi.mock('@/lib/auth/store', () => ({
  useAuth: () => ({
    account: { id: 1, username: 'me', email: 'me@x.io', role: 'user', consents: [] },
    loading: false,
    probeFailed: false,
    logout: h.logout,
    refresh: h.refreshAuth,
  }),
}));
vi.mock('@/lib/i18n/href', () => ({
  useLocalizedHref: () => (path: string) => path,
}));

import { AccountPanel } from '../AccountPanel';

afterEach(() => {
  h.push.mockReset();
  h.replace.mockReset();
  h.refresh.mockReset();
  h.logout.mockReset();
  h.refreshAuth.mockReset();
});

describe('AccountPanel — logout invalidates the Router Cache', () => {
  it('refreshes the Router Cache BEFORE navigating home', async () => {
    h.logout.mockResolvedValue(undefined);
    render(<AccountPanel />);

    fireEvent.click(screen.getByRole('button', { name: 'Se déconnecter' }));

    await waitFor(() => expect(h.push).toHaveBeenCalledWith('/'));
    expect(h.logout).toHaveBeenCalledTimes(1);
    expect(h.refresh).toHaveBeenCalledTimes(1);
    expect(h.refresh.mock.invocationCallOrder[0]!).toBeLessThan(
      h.push.mock.invocationCallOrder[0]!,
    );
    // A logged-in panel must never trigger the logged-out redirect.
    expect(h.replace).not.toHaveBeenCalled();
  });
});
