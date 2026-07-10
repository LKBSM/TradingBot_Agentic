import * as React from 'react';
import { render as rtlRender, type RenderOptions } from '@testing-library/react';
import { NextIntlClientProvider } from 'next-intl';
import messages from '@/messages/fr.json';

/**
 * Test render that wraps the tree in `NextIntlClientProvider` (fr messages) so
 * components consuming `useTranslations` (market-reading card, scanner, etc.)
 * render under test. Re-exports the rest of @testing-library/react so a test
 * file can simply import from here instead of the library. fr messages keep any
 * asserted French strings intact.
 */
export * from '@testing-library/react';

export function render(ui: React.ReactElement, options?: Omit<RenderOptions, 'wrapper'>) {
  return rtlRender(ui, {
    wrapper: ({ children }) => (
      <NextIntlClientProvider locale="fr" messages={messages}>
        {children}
      </NextIntlClientProvider>
    ),
    ...options,
  });
}
