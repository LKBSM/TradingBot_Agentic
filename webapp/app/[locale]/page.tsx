import { ConversationReplaySection } from '@/components/landing/ConversationReplaySection';
import { HeroLive } from '@/components/landing/HeroLive';
import { MultiMarketSection } from '@/components/landing/MultiMarketSection';
import { PricingSection } from '@/components/landing/PricingSection';

/**
 * Landing commerciale ultime — composition par sections.
 *
 *   L2  · HeroLive                  (live MarketReadingCard + ChatPreview, no perf)
 *   L3  · MultiMarketSection        (S2 multi-actifs XAU + EUR + Bientôt)
 *   L3  · ConversationReplaySection (S3 conversations rejouables)
 *   L4  · BeforeAfter               (à venir)
 *   L4  · HonestConfidence          (à venir, full-width)
 *   L5  · PricingMinimal            (3 tiers post-pivot 2026-05-27)
 *   L5  · FAQ                       (à venir)
 *   L5  · Footer enrichi            (à venir)
 */
export default function LandingPage() {
  return (
    <>
      <HeroLive />
      <MultiMarketSection />
      <ConversationReplaySection />
      <PricingSection />
    </>
  );
}
