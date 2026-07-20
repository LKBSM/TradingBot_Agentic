import { render, screen, fireEvent, waitFor } from '@/components/test-utils';
import { afterEach, describe, expect, it, vi } from 'vitest';

// Logout is the inverse of the first-login fix: after the session cookie is
// cleared, cached authenticated RSC entries must not survive in the Next.js
// Router Cache. The handler must call router.refresh() (invalidate) BEFORE
// navigating home.
const h = vi.hoisted(() => ({
  push: vi.fn(),
  refresh: vi.fn(),
  logout: vi.fn(),
}));

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: h.push, refresh: h.refresh }),
}));
vi.mock('@/lib/auth/store', () => ({
  useAuth: () => ({
    account: { id: 1, username: 'me', role: 'user' },
    isAuthenticated: true,
    logout: h.logout,
  }),
}));
vi.mock('@/lib/i18n/href', () => ({
  useLocalizedHref: () => (path: string) => path,
}));
// LocaleToggle pulls in navigation/pathname plumbing irrelevant here.
vi.mock('@/components/LocaleToggle', () => ({ LocaleToggle: () => null }));

import { AccountMenu } from '../AccountMenu';

afterEach(() => {
  h.push.mockReset();
  h.refresh.mockReset();
  h.logout.mockReset();
});

describe('AccountMenu — logout invalidates the Router Cache', () => {
  it('refreshes the Router Cache BEFORE navigating home', async () => {
    h.logout.mockResolvedValue(undefined);
    render(<AccountMenu />);

    // Open the menu (trigger is the only button until the panel renders).
    fireEvent.click(screen.getByRole('button', { name: 'Menu du compte' }));
    fireEvent.click(screen.getByRole('menuitem', { name: 'Se déconnecter' }));

    await waitFor(() => expect(h.push).toHaveBeenCalledWith('/'));
    expect(h.logout).toHaveBeenCalledTimes(1);
    expect(h.refresh).toHaveBeenCalledTimes(1);
    // Cache invalidation must precede navigation, otherwise a cached
    // authenticated page could be served to the now-logged-out user.
    expect(h.refresh.mock.invocationCallOrder[0]!).toBeLessThan(
      h.push.mock.invocationCallOrder[0]!,
    );
  });
});
