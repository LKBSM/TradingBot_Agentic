'use client';

import { useEffect, useRef } from 'react';
import styles from './lp1.module.css';

/**
 * LP-3 — a hairline at the top of the page showing how far down it you are.
 *
 * It reports position, it does not decorate: the bar is driven by `scaleX` on a
 * composited transform and updated inside one `requestAnimationFrame` per
 * scroll burst, so scrolling this long page costs no layout. Purely decorative
 * to assistive technology — the real structure is the sections themselves.
 */
export function ScrollProgress() {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let frame = 0;
    const paint = () => {
      frame = 0;
      const el = ref.current;
      if (!el) return;
      const max = document.documentElement.scrollHeight - window.innerHeight;
      const ratio = max > 0 ? Math.min(1, Math.max(0, window.scrollY / max)) : 0;
      el.style.transform = `scaleX(${ratio})`;
    };
    const schedule = () => {
      if (!frame) frame = window.requestAnimationFrame(paint);
    };
    paint();
    window.addEventListener('scroll', schedule, { passive: true });
    window.addEventListener('resize', schedule, { passive: true });
    return () => {
      if (frame) window.cancelAnimationFrame(frame);
      window.removeEventListener('scroll', schedule);
      window.removeEventListener('resize', schedule);
    };
  }, []);

  return <div ref={ref} className={styles.vprog} aria-hidden="true" />;
}
