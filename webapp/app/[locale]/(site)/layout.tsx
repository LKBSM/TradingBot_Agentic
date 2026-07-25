import { Nav } from '@/components/Nav';
import { Footer } from '@/components/Footer';
import { SkipLink } from '@/components/a11y/SkipLink';
import { ChatPanel } from '@/components/chat/ChatPanel';
import { CookieBanner } from '@/components/compliance/CookieBanner';
import { JsonLd, softwareApplicationLd } from '@/components/seo/JsonLd';

/**
 * Marketing / auth / legal chrome (the "site" surface). Sticky Nav on top, a
 * flex-1 main between it and the Footer, plus the floating ChatPanel and the
 * cookie banner. The providers live one level up in the locale layout, so this
 * group only assembles the presentation. The product routes live in the sibling
 * (product) group and get the terminal shell instead of this chrome.
 */
export default function SiteLayout({ children }: { children: React.ReactNode }) {
  return (
    <>
      <SkipLink />
      <JsonLd data={softwareApplicationLd} />
      <Nav />
      {/* flex-1 fills the space between the sticky header and the footer without
          a hard-coded height guess. */}
      <main id="main" className="flex-1">
        {children}
      </main>
      <Footer />
      <ChatPanel />
      <CookieBanner />
    </>
  );
}
