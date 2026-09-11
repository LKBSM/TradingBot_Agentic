import * as React from 'react';
import { render as rtlRender, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { NextIntlClientProvider } from 'next-intl';
import { TooltipProvider } from '@/components/ui/tooltip';
import messages from '@/messages/fr.json';

import { MarketReadingHeader } from '@/components/market-reading/MarketReadingHeader';
import { MarketReadingCard } from '@/components/market-reading/MarketReadingCard';
import { ZoneLifecycleCard } from '@/components/zones/ZoneLifecycleCard';
import { ScanResults } from '@/components/scanner/ScanResults';
import { ChatMessage } from '@/components/chat/ChatMessage';
import { ChatComposer } from '@/components/chat/ChatComposer';

import {
  SAMPLE_READING_XAU_H4,
  SAMPLE_CANDLES_XAU_H4,
  SAMPLE_ZONES_XAU_H4,
  SAMPLE_LIQUIDITY_XAU_H4,
  SAMPLE_ZONE_REFERENCE_PRICE,
  SAMPLE_ZONE_PRICE_INSIDE,
  SAMPLE_SCAN_RESPONSE,
  SAMPLE_SCAN_CONFIG,
  SAMPLE_CHAT_TURNS,
} from '@/lib/ds-samples';

/**
 * DS-1 — the presentation components render from the frozen samples with NO
 * network. Any fetch attempt throws and fails the test, so a component that
 * secretly self-fetches cannot pass. (ReadingChart needs a real canvas/
 * ResizeObserver and so is proven no-network by code + Playwright, not jsdom.)
 */

let fetchSpy: ReturnType<typeof vi.fn>;

beforeEach(() => {
  fetchSpy = vi.fn(() => {
    throw new Error('DS-1: a presentation component attempted a network request');
  });
  vi.stubGlobal('fetch', fetchSpy);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

const noop = () => {};

function render(ui: React.ReactElement) {
  return rtlRender(
    <NextIntlClientProvider locale="fr" messages={messages}>
      <TooltipProvider>{ui}</TooltipProvider>
    </NextIntlClientProvider>,
  );
}

describe('DS-1 presentation components — no network', () => {
  it('MarketReadingHeader renders from a frozen reading', () => {
    render(<MarketReadingHeader header={SAMPLE_READING_XAU_H4.header} />);
    expect(screen.getAllByText(/XAU/).length).toBeGreaterThan(0);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('MarketReadingCard renders from a frozen reading', () => {
    render(<MarketReadingCard reading={SAMPLE_READING_XAU_H4} />);
    expect(screen.getAllByText(/XAU/).length).toBeGreaterThan(0);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('ZoneLifecycleCard renders from a frozen zone (price inside band)', () => {
    expect(SAMPLE_ZONE_PRICE_INSIDE).toBeDefined();
    render(
      <ZoneLifecycleCard
        zone={SAMPLE_ZONE_PRICE_INSIDE!}
        instrument="XAUUSD"
        referencePrice={SAMPLE_ZONE_REFERENCE_PRICE}
        candles={[...SAMPLE_CANDLES_XAU_H4]}
        sameTfZones={SAMPLE_ZONES_XAU_H4}
        siblingZones={[]}
        liquidityPools={SAMPLE_LIQUIDITY_XAU_H4}
        isHidden={false}
        onToggleHide={noop}
        onShowOnChart={noop}
        onSelect={noop}
        detailHref="/zones/z1"
        onNavigateToZone={noop}
      />,
    );
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('ScanResults renders a frozen scan response', () => {
    render(
      <ScanResults
        response={SAMPLE_SCAN_RESPONSE}
        config={SAMPLE_SCAN_CONFIG}
        locale="fr"
        onEdit={noop}
        onRefresh={noop}
        isRefreshing={false}
        autoRefreshEnabled={false}
        onToggleAutoRefresh={noop}
      />,
    );
    // the full-match combo card is present
    expect(screen.getAllByText(/XAUUSD|XAU\/USD/).length).toBeGreaterThan(0);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('ChatMessage renders every frozen turn', () => {
    render(
      <>
        {SAMPLE_CHAT_TURNS.map((t, i) => (
          <ChatMessage key={i} role={t.role} text={t.text} blockedReason={t.blockedReason} viewUpdated={t.viewUpdated} />
        ))}
      </>,
    );
    expect(screen.getByText(/CHOCH/)).toBeInTheDocument();
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('ChatComposer renders without network', () => {
    render(
      <ChatComposer
        onSubmit={noop}
        placeholder="Pose ta question…"
        ariaLabel="Message"
        sendAria="Envoyer"
        privacyNote="Note de confidentialité"
      />,
    );
    expect(screen.getByPlaceholderText('Pose ta question…')).toBeInTheDocument();
    expect(fetchSpy).not.toHaveBeenCalled();
  });
});
