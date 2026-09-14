'use client';

import { useEffect, useRef, useState, type ReactNode } from 'react';
import styles from './lp1.module.css';

/**
 * LP-3 — the page's ONE scroll-reveal. A section fades and lifts into place the
 * first time it comes into view, once, and never animates again.
 *
 * There was no shared utility for this before (only a dead pre-LP-1 component
 * with its own observer), so this is deliberately the single one: every section
 * below the hero goes through it.
 *
 * Two failure modes are designed out rather than accepted:
 *
 *  · **reduced motion** — the hidden state is declared only inside
 *    `@media (prefers-reduced-motion: no-preference)`, and the effect settles
 *    immediately anyway. A visitor who asked for less motion sees a plain page.
 *  · **no JS / a broken observer** — the server renders `data-rv="pending"`,
 *    which carries a failsafe animation that reveals the section on its own
 *    after 2.8 s. A marketing page must never be able to stay blank.
 */
export function Reveal({ children, id }: { children: ReactNode; id?: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [shown, setShown] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduce || typeof IntersectionObserver === 'undefined') {
      setShown(true);
      return undefined;
    }
    // Already on screen at load (the section right under the hero): reveal now
    // rather than waiting for a scroll that may never come.
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setShown(true);
          io.disconnect();
        }
      },
      { threshold: 0.1, rootMargin: '0px 0px -6% 0px' },
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);

  return (
    <div ref={ref} {...(id ? { id } : null)} className={styles.rv} data-rv={shown ? 'in' : 'pending'}>
      {children}
    </div>
  );
}
