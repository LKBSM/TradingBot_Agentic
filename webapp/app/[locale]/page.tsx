import { DemoSection } from '@/components/landing/DemoSection';
import { HeroSection } from '@/components/landing/HeroSection';
import { HowItWorksSection } from '@/components/landing/HowItWorksSection';
import { PricingSection } from '@/components/landing/PricingSection';

/**
 * Public landing page — composes hero + live demo + how-it-works + pricing
 * (placeholders, LEGAL-PENDING). All sub-sections live under
 * components/landing/ so they can be reordered or reused independently.
 */
export default function LandingPage() {
  return (
    <>
      <HeroSection />
      <DemoSection />
      <HowItWorksSection />
      <PricingSection />
    </>
  );
}
