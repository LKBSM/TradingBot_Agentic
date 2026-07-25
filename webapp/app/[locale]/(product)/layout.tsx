import { ProductShell } from '@/components/shell/ProductShell';

/**
 * Product route group (/app · /scanner · /zones · /compte). These surfaces get
 * the terminal shell (rail · center · docked chat) instead of the marketing
 * Nav/Footer chrome, which lives in the sibling (site) group. Providers are
 * shared from the locale layout one level up.
 */
export default function ProductLayout({ children }: { children: React.ReactNode }) {
  return <ProductShell>{children}</ProductShell>;
}
