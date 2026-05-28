'use client';

import dynamic from 'next/dynamic';

// Client wrapper qui permet d'utiliser ssr:false. Sans ce wrapper, Next 15
// refuse `ssr: false` quand l'import dynamique est posé dans un Server
// Component (cf. erreur build 2026-05-27). Le ChatPanel n'a aucune valeur
// SSR (il n'apparaît que sur action utilisateur), on peut le retarder
// jusqu'à l'hydratation pour économiser ~30 ko de bundle initial.
const ChatPanel = dynamic(
  () => import('./ChatPanel').then((m) => m.ChatPanel),
  { ssr: false },
);

export function ChatPanelLazy() {
  return <ChatPanel />;
}
