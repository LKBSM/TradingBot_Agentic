import { render, screen } from '@/components/test-utils';
import { describe, expect, it } from 'vitest';
import fr from '@/messages/fr.json';

import { PricingSection } from '../PricingSection';

/**
 * LEG-2 — the pricing block states how to leave, right next to the button that
 * makes you pay. A cancellation policy nobody can find before paying is not a
 * policy; and the footer is not "next to the price".
 */
describe('LEG-2 — the pricing section links to the refund clause', () => {
  it('shows the link next to the subscribe button, pointing at the clause', () => {
    const { container } = render(<PricingSection />);

    const link = screen.getByRole('link', { name: fr.landing.pricing.refundLink });
    expect(link).toHaveAttribute('href', '/conditions#remboursement');

    // « Next to the button » is the claim, so it is the claim that is tested:
    // the link is the button's immediate sibling inside the plan card.
    const button = screen.getByRole('button', { name: fr.landing.pricing.subscribeButton });
    expect(button.nextElementSibling).toBe(link);
    expect(container.querySelector('#tarifs')).not.toBeNull();
  });
});
