import { render, screen, fireEvent, waitFor } from '@/components/test-utils';
import { afterEach, describe, expect, it, vi } from 'vitest';

// Router + auth store are mocked so the test drives just the form's post-login
// navigation contract. The bug under test: on a cookieless device the FIRST
// login attempt bounced back to /connexion (a stale pre-login Router Cache
// redirect was served) and only a refresh let the user in. The fix invalidates
// the Router Cache (router.refresh) BEFORE navigating (router.replace).
const h = vi.hoisted(() => ({
  push: vi.fn(),
  replace: vi.fn(),
  refresh: vi.fn(),
  login: vi.fn(),
}));

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: h.push, replace: h.replace, refresh: h.refresh }),
}));
vi.mock('@/lib/auth/store', () => ({
  useAuth: () => ({ login: h.login }),
}));
vi.mock('@/lib/i18n/href', () => ({
  useLocalizedHref: () => (path: string) => path,
}));

import { LoginForm } from '../LoginForm';
import { AuthError } from '@/lib/auth/api-client';

function fill(container: HTMLElement, identifier: string, password: string) {
  const idInput = container.querySelector('input[name="identifier"]') as HTMLInputElement;
  const pwInput = container.querySelector('input[name="password"]') as HTMLInputElement;
  fireEvent.change(idInput, { target: { value: identifier } });
  fireEvent.change(pwInput, { target: { value: password } });
}

afterEach(() => {
  h.push.mockReset();
  h.replace.mockReset();
  h.refresh.mockReset();
  h.login.mockReset();
  window.history.replaceState({}, '', '/connexion');
});

describe('LoginForm — first-attempt reliability', () => {
  it('on success, refreshes the Router Cache BEFORE navigating (never push)', async () => {
    h.login.mockResolvedValue({ id: 1, username: 'me', role: 'user' });
    const { container } = render(<LoginForm />);
    fill(container, '  me  ', 'secret-pass');
    fireEvent.click(screen.getByRole('button'));

    await waitFor(() => expect(h.replace).toHaveBeenCalledWith('/app'));
    // Identifier is trimmed before submit.
    expect(h.login).toHaveBeenCalledWith({ identifier: 'me', password: 'secret-pass' });
    expect(h.refresh).toHaveBeenCalledTimes(1);
    // Soft push is never used — it can serve the stale cached redirect.
    expect(h.push).not.toHaveBeenCalled();
    // Cache invalidation MUST precede navigation, otherwise a stale pre-login
    // redirect can be served and bounce a freshly-authenticated user to login.
    expect(h.refresh.mock.invocationCallOrder[0]!).toBeLessThan(
      h.replace.mock.invocationCallOrder[0]!,
    );
  });

  it('honors a safe internal ?next= return path', async () => {
    h.login.mockResolvedValue({ id: 1, username: 'me', role: 'user' });
    window.history.replaceState({}, '', '/connexion?next=/scanner');
    const { container } = render(<LoginForm />);
    fill(container, 'me', 'secret-pass');
    fireEvent.click(screen.getByRole('button'));
    await waitFor(() => expect(h.replace).toHaveBeenCalledWith('/scanner'));
  });

  it('ignores an unsafe (off-site) ?next= and falls back to /app (AUTH-06)', async () => {
    h.login.mockResolvedValue({ id: 1, username: 'me', role: 'user' });
    window.history.replaceState({}, '', '/connexion?next=//evil.com');
    const { container } = render(<LoginForm />);
    fill(container, 'me', 'secret-pass');
    fireEvent.click(screen.getByRole('button'));
    await waitFor(() => expect(h.replace).toHaveBeenCalledWith('/app'));
  });

  it('on bad credentials, shows an error and does NOT navigate or refresh', async () => {
    h.login.mockRejectedValue(
      new AuthError(401, 'Identifiant ou mot de passe incorrect.'),
    );
    const { container } = render(<LoginForm />);
    fill(container, 'me', 'wrong');
    fireEvent.click(screen.getByRole('button'));

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'Identifiant ou mot de passe incorrect.',
    );
    expect(h.replace).not.toHaveBeenCalled();
    expect(h.refresh).not.toHaveBeenCalled();
    expect(h.push).not.toHaveBeenCalled();
  });
});
