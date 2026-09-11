'use client';

import * as React from 'react';
import { MarketReadingHeader } from '@/components/market-reading/MarketReadingHeader';
import { MarketReadingCard } from '@/components/market-reading/MarketReadingCard';
import { ZoneLifecycleCard } from '@/components/zones/ZoneLifecycleCard';
import { ReadingChart } from '@/components/app/ReadingChart';
import { ScanResults } from '@/components/scanner/ScanResults';
import { ChatMessage } from '@/components/chat/ChatMessage';
import { ChatComposer } from '@/components/chat/ChatComposer';
import { MiaSection } from '@/components/landing/lp1/MiaSection';
import { ReadingCarousel } from '@/components/landing/lp1/ReadingCarousel';
import type { ZoneLifecycle } from '@/lib/zones/lifecycle';
import {
  SAMPLE_READING_XAU_H4,
  SAMPLE_READING_EUR_M15,
  SAMPLE_READING_XAU_M15_SPARSE,
  SAMPLE_CANDLES_XAU_H4,
  SAMPLE_ZONES_XAU_H4,
  SAMPLE_LIQUIDITY_XAU_H4,
  SAMPLE_ZONE_REFERENCE_PRICE,
  SAMPLE_ZONE_PRICE_INSIDE,
  SAMPLE_ZONE_ABOVE,
  SAMPLE_ZONE_BELOW,
  SAMPLE_ZONE_TESTED,
  SAMPLE_ZONE_UNTOUCHED,
  SAMPLE_SCAN_RESPONSE,
  SAMPLE_SCAN_NO_MATCH,
  SAMPLE_SCAN_CONFIG,
  SAMPLE_CHAT_TURNS,
} from '@/lib/ds-samples';

/**
 * DS-1 — the design gallery. Renders every Palier-0 PRESENTATION component in
 * all its states, fed entirely by the frozen real-data samples (lib/ds-samples).
 * No network: nothing here fetches. It is the surface captured for Claude Design
 * and the fastest way to see a component without a backend + a loaded market.
 *
 * This is a dev-only surface — the route that mounts it returns notFound() in
 * production.
 */

const noop = () => {};

function Surface({ id, title, children }: { id: string; title: string; children: React.ReactNode }) {
  return (
    <section id={id} data-testid={`ds-surface-${id}`} className="mb-16">
      <h2 className="mb-1 fs-section font-semibold text-foreground">{title}</h2>
      <div className="mb-6 h-px w-full bg-border" />
      <div className="flex flex-col gap-10">{children}</div>
    </section>
  );
}

function State({
  name,
  note,
  width,
  children,
}: {
  name: string;
  note?: string;
  width?: number;
  children: React.ReactNode;
}) {
  return (
    <div data-testid={`ds-state-${name}`} className="flex flex-col gap-2">
      <div className="flex items-baseline gap-3">
        <span className="fs-label font-mono uppercase tracking-wide text-primary">{name}</span>
        {note ? <span className="fs-legal text-muted-foreground">{note}</span> : null}
      </div>
      <div style={width ? { maxWidth: width } : undefined}>{children}</div>
    </div>
  );
}

function Zone({ name, note, zone }: { name: string; note?: string; zone: ZoneLifecycle | undefined }) {
  if (!zone) return null;
  return (
    <State name={name} note={note} width={520}>
      <ZoneLifecycleCard
        zone={zone}
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
      />
    </State>
  );
}

export function DesignGallery() {
  const [mounted, setMounted] = React.useState(false);
  React.useEffect(() => setMounted(true), []);

  return (
    <div data-testid="ds-gallery" className="min-h-screen bg-background px-6 py-10 text-foreground">
      <header className="mb-12">
        <p className="fs-label font-mono uppercase tracking-wide text-primary">DS-1 · galerie interne</p>
        <h1 className="fs-title font-semibold">Composants affichables — données figées réelles</h1>
        <p className="mt-2 max-w-2xl fs-secondary text-muted-foreground">
          Chaque composant de présentation, dans tous ses états, alimenté par des données réelles
          figées (issues de market_readings.db). Aucun appel réseau. Surface non accessible en
          production.
        </p>
      </header>

      <Surface id="market-reading" title="Lecture de marché">
        <State name="en-tête" note="XAU/USD H4 — prix de clôture réel">
          <MarketReadingHeader header={SAMPLE_READING_XAU_H4.header} />
        </State>
        <State name="carte-complète" note="XAU/USD H4 — structure riche" width={720}>
          <MarketReadingCard reading={SAMPLE_READING_XAU_H4} />
        </State>
        <State name="carte-eur" note="EUR/USD M15" width={720}>
          <MarketReadingCard reading={SAMPLE_READING_EUR_M15} />
        </State>
        <State
          name="carte-champs-absents"
          note="Aucune cassure en cours, aucun événement, aucun retest → aucun élément rendu pour ces champs (jamais un tiret)"
          width={720}
        >
          <MarketReadingCard reading={SAMPLE_READING_XAU_M15_SPARSE} />
        </State>
      </Surface>

      <Surface id="zones" title="Zones (/zones)">
        <Zone
          name="prix-à-l’intérieur"
          note="Le prix est DANS la bande — le cas limite de la jauge de proximité"
          zone={SAMPLE_ZONE_PRICE_INSIDE}
        />
        <Zone name="au-dessus-du-prix" zone={SAMPLE_ZONE_ABOVE} />
        <Zone name="en-dessous-du-prix" zone={SAMPLE_ZONE_BELOW} />
        <Zone name="testée" note="Zone déjà touchée au moins une fois" zone={SAMPLE_ZONE_TESTED} />
        <Zone name="jamais-touchée" zone={SAMPLE_ZONE_UNTOUCHED} />
      </Surface>

      <Surface id="chart" title="Graphique de lecture">
        <State
          name="bougies-figées"
          note="XAU/USD H4 — 260 bougies réelles + structure, sans tick live (livePrice=null)"
        >
          <div className="rounded-lg border border-border">
            {mounted ? (
              <ReadingChart
                candles={[...SAMPLE_CANDLES_XAU_H4]}
                structure={SAMPLE_READING_XAU_H4.structure}
                instrument="XAUUSD"
                timeframe="H4"
                livePrice={null}
                heightClassName="h-[420px]"
              />
            ) : (
              <div className="h-[420px]" />
            )}
          </div>
        </State>
      </Surface>

      <Surface id="scanner" title="Scanner de conditions">
        <State name="avec-correspondances" note="Correspondance complète + presque + non évaluable" width={960}>
          <ScanResults
            response={SAMPLE_SCAN_RESPONSE}
            config={SAMPLE_SCAN_CONFIG}
            locale="fr"
            onEdit={noop}
            onRefresh={noop}
            isRefreshing={false}
            autoRefreshEnabled={false}
            onToggleAutoRefresh={noop}
          />
        </State>
        <State
          name="aucune-correspondance"
          note="Aucun combo ne satisfait tout → état explicite, jamais une erreur"
          width={960}
        >
          <ScanResults
            response={SAMPLE_SCAN_NO_MATCH}
            config={SAMPLE_SCAN_CONFIG}
            locale="fr"
            onEdit={noop}
            onRefresh={noop}
            isRefreshing={false}
            autoRefreshEnabled={false}
            onToggleAutoRefresh={noop}
          />
        </State>
      </Surface>

      <Surface id="mia" title="Conversation M.I.A">
        <State name="conversation" note="5 messages — dernier réponse redirigée (posture de lecture)" width={640}>
          <div className="flex flex-col gap-4 rounded-lg border border-border p-4">
            {SAMPLE_CHAT_TURNS.map((turn, i) => (
              <ChatMessage
                key={i}
                role={turn.role}
                text={turn.text}
                blockedReason={turn.blockedReason}
                viewUpdated={turn.viewUpdated}
              />
            ))}
          </div>
        </State>
        <State name="barre-de-saisie" note="Champ de composition (aucun envoi réseau)" width={640}>
          <ChatComposer
            onSubmit={noop}
            placeholder="Pose ta question à M.I.A…"
            ariaLabel="Message pour M.I.A"
            sendAria="Envoyer"
            privacyNote="La dictée vocale utilise la transcription du navigateur ; elle peut transiter par ses serveurs."
          />
        </State>
      </Surface>

      <Surface id="landing" title="Blocs de la page d’accueil (déjà statiques)">
        <State name="section-mia">
          <MiaSection />
        </State>
        <State name="carrousel-lectures">
          <ReadingCarousel />
        </State>
      </Surface>
    </div>
  );
}
