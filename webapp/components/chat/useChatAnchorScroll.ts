'use client';

import * as React from 'react';

/** Top padding (py-4 = 1rem) of both chat scroll areas — keeps the anchored
 *  question a hair below the top edge instead of flush against it. */
const TOP_GAP_PX = 16;

interface TurnLike {
  role: 'user' | 'assistant';
}

type AnchorTarget = 'user' | 'assistant';

interface AnchorOptions {
  /**
   * Which message to pin near the top once a new exchange starts.
   * - `'user'` (default): the latest question — keeps the question and the
   *   beginning of the reply both visible (landing slide-over behaviour).
   * - `'assistant'`: the latest reply — pins the *first word of M.I.A's answer*
   *   to the top (docked sidebar). Falls back to the question while the reply
   *   has not rendered yet, then re-pins to the reply the moment it appears.
   */
  anchor?: AnchorTarget;
}

/**
 * Assistant-style chat scrolling.
 *
 * Instead of yanking the viewport to the absolute bottom on every change
 * (which buries the start of a long reply below the fold and forces the reader
 * to scroll back up), this anchors a chosen message near the top of the scroll
 * area:
 *
 * - When a new question is sent, following engages and the anchor is
 *   smooth-scrolled to the top.
 * - While "following", the anchor stays pinned at the top as the answer lands /
 *   grows — so the reader keeps reading from the start, never teleported to the
 *   bottom. (Streaming-safe: incremental growth re-pins to the same spot, which
 *   is a no-op once reached.) With `anchor: 'assistant'` the pin hands off from
 *   the question to the reply the moment the reply mounts.
 * - The first wheel / touch gesture disengages following, so the user can scroll
 *   freely during generation without being re-aspirated.
 *
 * The hook only reads layout and scrolls a single container — it touches no chat
 * logic, security, or view-control.
 *
 * Anchoring relies on the latest `[data-chat-role="…"]` element inside the
 * container (set by <ChatMessage />).
 */
export function useChatAnchorScroll(
  turns: ReadonlyArray<TurnLike>,
  isLoading: boolean,
  options: AnchorOptions = {},
) {
  const anchor = options.anchor ?? 'user';
  const scrollRef = React.useRef<HTMLDivElement>(null);
  const prevUserCount = React.useRef(0);
  const followingRef = React.useRef(false);

  const userCount = React.useMemo(
    () => turns.reduce((n, t) => (t.role === 'user' ? n + 1 : n), 0),
    [turns],
  );

  const anchorToTop = React.useCallback(() => {
    const container = scrollRef.current;
    if (!container) return;
    const pick = (role: AnchorTarget) => {
      const nodes = container.querySelectorAll<HTMLElement>(
        `[data-chat-role="${role}"]`,
      );
      return nodes[nodes.length - 1] ?? null;
    };
    // Prefer the requested target; fall back to the latest question so the view
    // stays anchored while the assistant reply has not mounted yet.
    const target =
      (anchor === 'assistant' ? pick('assistant') : null) ?? pick('user');
    if (!target) return;
    const delta =
      target.getBoundingClientRect().top -
      container.getBoundingClientRect().top;
    const top = Math.max(0, container.scrollTop + delta - TOP_GAP_PX);
    // Already at the anchor (within a few px)? Don't re-issue a smooth scroll.
    // The streaming re-pin fires on every chunk; without this guard each chunk
    // launches a fresh smooth-scroll that fights the previous one → visible
    // jitter (UI-07). This makes the "no-op once reached" comment actually true.
    if (Math.abs(top - container.scrollTop) < 4) return;
    container.scrollTo({ top, behavior: 'smooth' });
  }, [anchor]);

  // A new question was sent → engage following and pin the anchor to the top.
  React.useEffect(() => {
    if (userCount > prevUserCount.current) {
      followingRef.current = true;
      anchorToTop();
    }
    prevUserCount.current = userCount;
  }, [userCount, anchorToTop]);

  // The answer landed / grew (or the loader toggled) → keep the anchor pinned
  // while following, so the start of the reply stays in view. No-op once the
  // anchor already sits at the top; this is also where an assistant anchor hands
  // the pin off from the question to the freshly-mounted reply.
  React.useEffect(() => {
    if (followingRef.current) anchorToTop();
  }, [turns, isLoading, anchorToTop]);

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
