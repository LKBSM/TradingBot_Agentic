import type { CSSProperties, ReactNode } from 'react';
import styles from './lp1.module.css';

/**
 * Colour pair per zone type — the product's OWN tokens (app/globals.css),
 * theme-aware across the four themes: rose for an Order Block, violet for a
 * Fair Value Gap. Nothing hard-coded here.
 *
 * Liquidity (amber `--liq`) is deliberately absent: the engine draws a pocket
 * as a LEVEL LINE, not a band, so it has no rectangle to share. Adding an
 * unused variant would be dead code.
 */
const KIND_COLORS = {
  ob: { band: 'var(--ob)', line: 'var(--ob-l)', text: 'var(--bear)' },
  fvg: { band: 'var(--fvg)', line: 'var(--fvg-l)', text: 'var(--fvg-l)' },
} as const;

export type ZoneKind = keyof typeof KIND_COLORS;

export interface ZoneRectProps {
  kind: ZoneKind;
  /** Absolute placement of the band over its chart box (caller owns geometry). */
  style: CSSProperties;
  label: ReactNode;
  /** Absolute placement of the label chip. */
  labelStyle: CSSProperties;
  /** 0-100 — the share of the band price has already eaten. Omitted = untouched. */
  fillPct?: number;
  /** A zone whose life is over: drawn dimmed and solid-edged, never hidden. */
  spent?: boolean;
}

/**
 * THE renderer for a detected zone drawn over a candle box — the demo's shared
 * visual language. Used by the Structure tab (one band per layer, over the full
 * chart) and by the Zones tab (the selected zone over a compact chart). Callers
 * own the geometry; this component owns the colours, the dashed band, the fill
 * and the label chip, so the two tabs can never drift apart.
 */
export function ZoneRect({ kind, style, label, labelStyle, fillPct, spent }: ZoneRectProps) {
  const c = KIND_COLORS[kind];
  return (
    <>
      <div
        className={styles.dz}
        style={{
          background: c.band,
          border: `1px ${spent ? 'solid' : 'dashed'} ${c.line}`,
          ...(spent ? { opacity: 0.55 } : null),
          ...style,
        }}
      >
        {fillPct != null && (
          <div
            className={styles.dzfill}
            style={{ height: `${fillPct}%`, background: c.line }}
          />
        )}
      </div>
      <div
        className={styles.dl}
        style={{ background: c.band, color: c.text, ...labelStyle }}
      >
        {label}
      </div>
    </>
  );
}
