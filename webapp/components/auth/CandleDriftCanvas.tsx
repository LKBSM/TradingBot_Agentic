'use client';

import * as React from 'react';

/**
 * Animated connexion backdrop — a faithful port of the `#fx` IIFE in
 * docs/design/reference-desktop.html: random-walk candlesticks drifting LEFT
 * behind the login card, a faint grid, and one dashed liquidity line. Purely
 * decorative (aria-hidden), self-contained, no external deps. Colours are read
 * from the live design tokens on <body> so it follows all four themes.
 *
 * Honours `prefers-reduced-motion: reduce` → a single static frame, no rAF loop.
 */
export function CandleDriftCanvas() {
  const ref = React.useRef<HTMLCanvasElement>(null);

  React.useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const ctx = cv.getContext('2d');
    if (!ctx) return;

    const DPR = Math.min(window.devicePixelRatio || 1, 2);
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    let W = 0;
    let H = 0;

    function resize() {
      const host = cv!.parentElement ?? cv!;
      const r = host.getBoundingClientRect();
      W = r.width;
      H = r.height;
      cv!.width = W * DPR;
      cv!.height = H * DPR;
      ctx!.setTransform(DPR, 0, 0, DPR, 0, 0);
    }
    resize();

    const css = getComputedStyle(document.body);
    const BULL = css.getPropertyValue('--bull').trim() || '#37b98c';
    const BEAR = css.getPropertyValue('--bear').trim() || '#dd6b7a';
    const LIQ = css.getPropertyValue('--liq').trim() || '#d6a24a';
    const GRID = 'rgba(255,255,255,.05)';
    const CW = 13;
    const GAP = 9;
    const STEP = CW + GAP;
    const SPEED = 0.22;

    type Candle = { x: number; o: number; c: number; h: number; l: number };
    let candles: Candle[] = [];
    let last = 0.5;

    function mk(x: number): Candle {
      const drift = (Math.random() - 0.5) * 0.07;
      const o = last;
      const c = Math.max(0.1, Math.min(0.9, o + drift));
      last = c;
      return {
        x,
        o,
        c,
        h: Math.max(o, c) + Math.random() * 0.035,
        l: Math.min(o, c) - Math.random() * 0.035,
      };
    }
    function fill() {
      candles = [];
      last = 0.5;
      for (let x = -STEP; x < W + STEP * 2; x += STEP) candles.push(mk(x));
    }
    fill();

    const yv = (v: number) => H * (0.12 + 0.76 * (1 - v));

    function draw() {
      ctx!.clearRect(0, 0, W, H);
      ctx!.strokeStyle = GRID;
      ctx!.lineWidth = 1;
      for (let g = 0; g <= 4; g++) {
        const yy = H * (0.12 + (0.76 * g) / 4);
        ctx!.beginPath();
        ctx!.moveTo(0, yy);
        ctx!.lineTo(W, yy);
        ctx!.stroke();
      }
      ctx!.save();
      ctx!.globalAlpha = 0.3;
      ctx!.strokeStyle = LIQ;
      ctx!.setLineDash([6, 5]);
      const ly = H * 0.26;
      ctx!.beginPath();
      ctx!.moveTo(0, ly);
      ctx!.lineTo(W, ly);
      ctx!.stroke();
      ctx!.restore();
      ctx!.globalAlpha = 0.13;
      candles.forEach((cd) => {
        const up = cd.c >= cd.o;
        const col = up ? BULL : BEAR;
        const cx = cd.x + CW / 2;
        ctx!.strokeStyle = col;
        ctx!.fillStyle = col;
        ctx!.lineWidth = 1.2;
        ctx!.beginPath();
        ctx!.moveTo(cx, yv(cd.h));
        ctx!.lineTo(cx, yv(cd.l));
        ctx!.stroke();
        const top = yv(Math.max(cd.o, cd.c));
        const bh = Math.max(2, Math.abs(yv(cd.o) - yv(cd.c)));
        ctx!.fillRect(cd.x, top, CW, bh);
      });
      ctx!.globalAlpha = 1;
    }

    let raf = 0;
    function tick() {
      candles.forEach((cd) => (cd.x -= SPEED));
      if (candles.length && candles[0]!.x < -STEP) {
        candles.shift();
        const lx = candles[candles.length - 1]!.x;
        candles.push(mk(lx + STEP));
      }
      draw();
      raf = requestAnimationFrame(tick);
    }

    function onResize() {
      resize();
      fill();
      draw();
    }
    window.addEventListener('resize', onResize);

    if (reduce) draw();
    else tick();

    return () => {
      window.removeEventListener('resize', onResize);
      if (raf) cancelAnimationFrame(raf);
    };
  }, []);

  return <canvas ref={ref} className="fx" aria-hidden="true" />;
}
