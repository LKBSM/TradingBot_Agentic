/**
 * Skip-to-content link required for WCAG 2.1 keyboard navigation. Visually
 * hidden until focused (Tab key from page load), then sticks to the top-
 * left so a keyboard user can jump past the navbar. Targets `#main`.
 */
export function SkipLink() {
  return (
    <a
      href="#main"
      className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-[60] focus:rounded-md focus:bg-primary focus:px-4 focus:py-2 focus:text-sm focus:font-medium focus:text-primary-foreground focus:shadow-lg focus:outline-none focus:ring-2 focus:ring-ring"
    >
      Aller au contenu principal
    </a>
  );
}
