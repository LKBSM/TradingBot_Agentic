import { afterEach, describe, expect, it, vi } from 'vitest';

/**
 * DS-1 §2C — the gallery route is DEV-ONLY. In production it must 404 (calls
 * notFound()); in development it must NOT. We mock the heavy client component so
 * the test only exercises the route guard, and assert purely on whether
 * notFound() fires (the JSX return value is irrelevant to the guard).
 */
vi.mock('@/components/gallery/DesignGallery', () => ({
  DesignGallery: () => null,
}));

const notFoundSpy = vi.fn(() => {
  throw new Error('NEXT_NOT_FOUND');
});
vi.mock('next/navigation', () => ({
  notFound: notFoundSpy,
}));

afterEach(() => {
  vi.unstubAllEnvs();
  vi.resetModules();
  notFoundSpy.mockClear();
});

async function callPage() {
  const mod = await import('../page');
  try {
    mod.default();
  } catch {
    // Under stubbed NODE_ENV + resetModules the JSX dev-runtime binding can be
    // stale; the guard's notFound() call (or lack of it) is what we assert on.
  }
}

describe('galerie route guard', () => {
  it('calls notFound() in production', async () => {
    vi.stubEnv('NODE_ENV', 'production');
    await callPage();
    expect(notFoundSpy).toHaveBeenCalledTimes(1);
  });

  it('does not call notFound() in development', async () => {
    vi.stubEnv('NODE_ENV', 'development');
    await callPage();
    expect(notFoundSpy).not.toHaveBeenCalled();
  });
});
