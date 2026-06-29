'use client';

import * as React from 'react';

/** Top padding (py-4 = 1rem) of both chat scroll areas — keeps the anchored
 *  question a hair below the top edge instead of flush against it. */
const TOP_GAP_PX = 16;

interface TurnLike {
  role: 'user' | 'assistant';
}

/**
 * Assistant-style chat scrolling.
 *
 * Instead of yanking the viewport to the absolute bottom on every change
 * (which buries the start of a long reply below the fold and forces the reader
 * to scroll back up), this anchors the *latest user message* near the top of
 * the scroll area:
 *
 * - When a new question is sent, the question is smooth-scrolled to the top so
 *   the question and the beginning of the answer are both visible.
 * - While "following", the same question stays pinned at the top as the answer
 *   lands / grows — so the reader keeps reading from the start, never teleported
 *   to the bottom. (Streaming-safe: incremental growth re-pins to the same spot,
 *   which is a no-op once reached.)
 * - The first wheel / touch gesture disengages following, so the user can scroll
 *   freely during generation without being re-aspirated.
 *
 * The hook only reads layout and scrolls a single container — it touches no chat
 * logic, security, or view-control.
 *
 * Anchoring relies on the latest `[data-chat-role="user"]` element inside the
 * container (set by <ChatMessage />).
 */
export function useChatAnchorScroll(
  turns: ReadonlyArray<TurnLike>,
  isLoading: boolean,
) {
  const scrollRef = React.useRef<HTMLDivElement>(null);
  const prevUserCount = React.useRef(0);
  const followingRef = React.useRef(false);

  const userCount = React.useMemo(
    () => turns.reduce((n, t) => (t.role === 'user' ? n + 1 : n), 0),
    [turns],
  );

  const anchorToLastUser = React.useCallback(() => {
    const container = scrollRef.current;
    if (!container) return;
    const users = container.querySelectorAll<HTMLElement>(
      '[data-chat-role="user"]',
    );
    const anchor = users[users.length - 1];
    if (!anchor) return;
    const delta =
      anchor.getBoundingClientRect().top -
      container.getBoundingClientRect().top;
    const top = Math.max(0, container.scrollTop + delta - TOP_GAP_PX);
    container.scrollTo({ top, behavior: 'smooth' });
  }, []);

  // A new question was sent → engage following and pin it to the top.
  React.useEffect(() => {
    if (userCount > prevUserCount.current) {
      followingRef.current = true;
      anchorToLastUser();
    }
    prevUserCount.current = userCount;
  }, [userCount, anchorToLastUser]);

  // The answer landed / grew (or the loader toggled) → keep the question pinned
  // while following, so the start of the reply stays in view. No-op once the
  // question already sits at the top.
  React.useEffect(() => {
    if (followingRef.current) anchorToLastUser();
  }, [turns, isLoading, anchorToLastUser]);

  // Any deliberate scroll gesture disengages following — the user is now in
  // control and must not be re-aspirated toward the anchor.
  React.useEffect(() => {
    const container = scrollRef.current;
    if (!container) return;
    const disengage = () => {
      followingRef.current = false;
    };
    container.addEventListener('wheel', disengage, { passive: true });
    container.addEventListener('touchmove', disengage, { passive: true });
    return () => {
      container.removeEventListener('wheel', disengage);
      container.removeEventListener('touchmove', disengage);
    };
  }, []);

  return scrollRef;
}
