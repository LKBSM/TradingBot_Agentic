'use client';

import dynamic from 'next/dynamic';

// Idem ChatPanelLazy : le bandeau cookies n'a aucune valeur SSR (il s'affiche
// uniquement après lecture du localStorage). On retarde son code pour
// alléger le first paint mobile (Lighthouse 2026-05-27).
const CookieBanner = dynamic(
  () => import('./CookieBanner').then((m) => m.CookieBanner),
  { ssr: false },
);

export function CookieBannerLazy() {
  return <CookieBanner />;
}
