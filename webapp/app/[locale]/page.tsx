import { BeforeAfterSection } from '@/components/landing/BeforeAfterSection';
import { ConversationReplaySection } from '@/components/landing/ConversationReplaySection';
import { FaqSection } from '@/components/landing/FaqSection';
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
 *   L5  · PricingSection             (FREE / 9€ / 19€ post-pivot 2026-05-27 + Calendly B2B)
 *   L5  · FaqSection                 (6 questions clés)
 *   (Footer enrichi dans <Footer /> du layout — 9 pays Phase 1 + Early Access)
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
      <FaqSection />
    </>
  );
}
