'use client';

import * as React from 'react';
import { Search as SearchIcon, ShieldCheck } from 'lucide-react';
import { cn } from '@/lib/utils';

/**
 * Base UI primitives for the product shell (mission UI-1). Each is a thin wrapper
 * that emits the reference class names (see components/shell/shell.css), so the
 * validated reference markup transposes one-to-one. They only carry visual style
 * + a11y affordances — no data, no detection. Colour comes from the literal
 * design tokens, so all four themes are covered for free.
 */

/* ── Card ─────────────────────────────────────────────────────────────── */
export function Card({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn('card', className)} {...props} />;
}

export function CardHeader({
  icon,
  title,
  badge,
}: {
  icon?: React.ReactNode;
  title: React.ReactNode;
  /** Small uppercase note pinned to the right (the reference's `.badge2`). */
  badge?: React.ReactNode;
}) {
  return (
    <div className="card-h">
      {icon}
      <h3>{title}</h3>
      {badge != null && <span className="badge2">{badge}</span>}
    </div>
  );
}

/* ── Button (default / primary / acc / danger) ────────────────────────── */
type BtnVariant = 'default' | 'primary' | 'acc' | 'danger';

export const Btn = React.forwardRef<
  HTMLButtonElement,
  React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: BtnVariant }
>(function Btn({ variant = 'default', className, type = 'button', ...props }, ref) {
  return (
    <button
      ref={ref}
      type={type}
      className={cn('btn', variant !== 'default' && variant, className)}
      {...props}
    />
  );
});

/* ── Live dot + LiveBadge ─────────────────────────────────────────────── */
/** Pulsing "live" dot (a11y-hidden; meaning is carried by adjacent text). */
export function Dot({ className }: { className?: string }) {
  return <span className={cn('dot', className)} aria-hidden />;
}

export function LiveBadge({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <span className={cn('livebadge', className)}>
      <Dot />
      <span className="mono">{children}</span>
    </span>
  );
}

/* ── Early-access badge ───────────────────────────────────────────────── */
export function EarlyAccessBadge({ children }: { children: React.ReactNode }) {
  return (
    <span className="ea">
      <span className="d" aria-hidden />
      {children}
    </span>
  );
}

/* ── Inline legal mention (shield + one line) ─────────────────────────── */
export function LegalInline({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <span className={cn('legal-inline', className)}>
      <ShieldCheck aria-hidden />
      {children}
    </span>
  );
}

/* ── Tagx (bear / bull / neutral) ─────────────────────────────────────── */
export function Tagx({
  tone,
  children,
  className,
}: {
  tone: 'bear' | 'bull' | 'neu';
  children: React.ReactNode;
  className?: string;
}) {
  return <span className={cn('tagx', tone, 'mono', className)}>{children}</span>;
}

/* ── Timeframe pill (segmented control member) ────────────────────────── */
export const TfPill = React.forwardRef<
  HTMLButtonElement,
  React.ButtonHTMLAttributes<HTMLButtonElement> & { active?: boolean }
>(function TfPill({ active = false, className, type = 'button', ...props }, ref) {
  return (
    <button
      ref={ref}
      type={type}
      aria-pressed={active}
      className={cn('tf', active && 'on', className)}
      {...props}
    />
  );
});

/* ── Search field ─────────────────────────────────────────────────────── */
export function SearchField({
  className,
  ...props
}: React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <div className={cn('search', className)}>
      <SearchIcon aria-hidden />
      <input type="search" {...props} />
    </div>
  );
}

/* ── Text input + field label ─────────────────────────────────────────── */
export const Input = React.forwardRef<
  HTMLInputElement,
  React.InputHTMLAttributes<HTMLInputElement>
>(function Input({ className, ...props }, ref) {
  return <input ref={ref} className={cn('input', className)} {...props} />;
});

export function FieldLabel({
  className,
  ...props
}: React.LabelHTMLAttributes<HTMLLabelElement>) {
  return <label className={cn('flabel', className)} {...props} />;
}

/* ── Freshbox (live-reading footer block) ─────────────────────────────── */
export function Freshbox({
  line1,
  line2,
  className,
}: {
  line1: React.ReactNode;
  line2: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn('freshbox', className)}>
      <Dot />
      <div>
        <div className="l1">{line1}</div>
        <div className="l2 mono">{line2}</div>
      </div>
    </div>
  );
}

/* ── Chart layer chip (swatch + label, toggleable) ────────────────────── */
export function LayerChip({
  swatch,
  off = false,
  children,
  className,
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  /** CSS colour for the swatch square (usually a `var(--…)` token). */
  swatch: string;
  off?: boolean;
}) {
  return (
    <button
      type="button"
      aria-pressed={!off}
      className={cn('layer', off && 'off', className)}
      {...props}
    >
      <span className="sw" style={{ background: swatch }} aria-hidden />
      {children}
    </button>
  );
}
