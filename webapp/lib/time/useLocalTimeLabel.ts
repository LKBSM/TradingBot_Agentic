import * as React from 'react';
import { localTimeLabel } from './localTime';

/**
 * The « Heure locale · UTC−X » indicator for the reader's OWN timezone, kept in
 * sync with the browser.
 *
 * Resolved on the client only: the initial state is '' so the server render and
 * the first client render match (no hydration mismatch), then the real label is
 * computed on mount. It is recomputed whenever the tab regains focus or becomes
 * visible again so a mid-session timezone change — VPN, OS timezone edit,
 * travel, or a DST crossing — is reflected WITHOUT a reload. The browser fires
 * no native timezone-change event, and such a change almost always coincides
 * with the reader leaving and returning to the tab. setState with an unchanged
 * string is a no-op (React bails on identical primitive state), so re-syncing on
 * every focus is free.
 */
export function useLocalTimeLabel(): string {
  const [label, setLabel] = React.useState('');
  React.useEffect(() => {
    const sync = () => setLabel(localTimeLabel());
    sync();
    const onVisibility = () => {
      if (document.visibilityState === 'visible') sync();
    };
    window.addEventListener('focus', sync);
    document.addEventListener('visibilitychange', onVisibility);
    return () => {
      window.removeEventListener('focus', sync);
      document.removeEventListener('visibilitychange', onVisibility);
    };
  }, []);
  return label;
}
