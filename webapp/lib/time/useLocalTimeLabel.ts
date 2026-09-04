import * as React from 'react';
import { useTranslations } from 'next-intl';
import { utcOffsetLabel } from './localTime';

/**
 * The « Heure locale · UTC−X » indicator for the reader's OWN timezone, kept in
 * sync with the browser. The « Heure locale »/« Local time »/« Hora local »
 * prefix is translated (i18n key `app.chart.localTime`, offset injected); the
 * offset itself is locale-independent.
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
  const t = useTranslations('app.chart');
  const [label, setLabel] = React.useState('');
  React.useEffect(() => {
    const sync = () => setLabel(t('localTime', { offset: utcOffsetLabel() }));
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
  }, [t]);
  return label;
}
