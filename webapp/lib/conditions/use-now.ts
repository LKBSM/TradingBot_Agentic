'use client';

import * as React from 'react';

/**
 * A clock that re-renders the caller every `intervalMs`, so relative-time
 * labels ("il y a 3 min") stay honest while the page sits open — WITHOUT
 * re-fetching anything. Returns the current epoch ms.
 */
export function useNow(intervalMs: number): number {
  const [now, setNow] = React.useState(() => Date.now());
  React.useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), intervalMs);
    return () => clearInterval(id);
  }, [intervalMs]);
  return now;
}
