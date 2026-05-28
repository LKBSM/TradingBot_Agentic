import { BeforeAfterSection } from '@/components/landing/BeforeAfterSection';
import { ConversationReplaySection } from '@/components/landing/ConversationReplaySection';
import { HeroLive } from '@/components/landing/HeroLive';
import { HonestConfidenceSection } from '@/components/landing/HonestConfidenceSection';
import { MultiMarketSection } from '@/components/landing/MultiMarketSection';
import { PricingSection } from '@/components/landing/PricingSection';

/**
 * Landing commerciale ultime — composition par sections.
 *
 *   L2  · HeroLive                   (live MarketReadingCard + ChatPreview, no perf)
 *   L3  · MultiMarketSection         (S2 multi-actifs XAU + EUR + Bientôt)
 *   L3  · ConversationReplaySection  (S3 conversations rejouables)
 *   L4  · BeforeAfterSection         (S4 avant/après — chaos vs lecture MIA)
 *   L4  · HonestConfidenceSection    (S5 vrais chiffres + citation imposée, full-width)
 *   L5  · PricingMinimal             (3 tiers post-pivot 2026-05-27)
 *   L5  · FAQ                        (à venir)
 *   L5  · Footer enrichi             (à venir)
 */
export default function LandingPage() {
  return (
    <>
      <HeroLive />
      <MultiMarketSection />
      <ConversationReplaySection />
      <BeforeAfterSection />
      <HonestConfidenceSection />
      <PricingSection />
    </>
  );
}
