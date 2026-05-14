'use client';

import { useState } from 'react';
import { GLOSSARY_BY_TERM } from '@/lib/glossary';

export function Tooltip({ term, children }: { term: string; children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  const definition = GLOSSARY_BY_TERM[term.toLowerCase()];
  if (!definition) return <>{children}</>;
  return (
    <span
      className="relative cursor-help border-b border-dotted border-slate-500"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onClick={() => setOpen((o) => !o)}
    >
      {children}
      {open && (
        <span className="absolute bottom-full left-1/2 z-10 mb-2 w-64 -translate-x-1/2 rounded-md bg-sentinel-ink p-2 text-xs text-white shadow-lg">
          <strong className="block">{definition.term}</strong>
          <span>{definition.definition_fr}</span>
        </span>
      )}
    </span>
  );
}
