import { DemoSection } from '@/components/landing/DemoSection';
import { HeroLive } from '@/components/landing/HeroLive';
import { HowItWorksSection } from '@/components/landing/HowItWorksSection';
import { PricingSection } from '@/components/landing/PricingSection';

/**
 * Landing commerciale ultime — composition par sections.
 *
 *   L2  · HeroLive            (live MarketReadingCard + ChatPreview, no perf)
 *   L3  · DemoSection         (3 cards multi-actifs)  → à remplacer par MultiMarket + ConversationReplay
 *   L3+ · HowItWorks          (legacy) → à supprimer en L4
 *   L4  · BeforeAfter         (à venir)
 *   L4  · HonestConfidence    (à venir, full-width)
 *   L5  · PricingMinimal      (3 tiers)
 *   L5  · FAQ                 (à venir)
 */
export default function LandingPage() {
  return (
    <>
      <HeroLive />
      <DemoSection />
      <HowItWorksSection />
      <PricingSection />
    </>
  );
}
