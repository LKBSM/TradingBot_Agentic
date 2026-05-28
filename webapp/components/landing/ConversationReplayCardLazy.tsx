'use client';

import dynamic from 'next/dynamic';

// Carte rejouable avec IntersectionObserver + typing effect, rendue 3× sur la
// page. Pas critique pour le first paint mobile — on retarde l'hydratation.
const ConversationReplayCard = dynamic(
  () => import('./ConversationReplayCard').then((m) => m.ConversationReplayCard),
  { ssr: false },
);

interface ConversationReplayCardLazyProps {
  title: string;
  kicker: string;
  question: string;
  answer: string;
  instrument: string;
  highlight?: 'refusal' | 'normal';
}

export function ConversationReplayCardLazy(props: ConversationReplayCardLazyProps) {
  return <ConversationReplayCard {...props} />;
}
