import { MiaLogo } from '@/components/brand/MiaLogo';

/**
 * Stacked M.I.A Markets lockup shown above an auth / access form (sign-up,
 * password reset, email verification, plan choice). The marketing Nav already
 * carries the horizontal lockup as the home link; this is the in-content brand
 * that anchors the form, so it is a labelled image (not a second home link) and
 * stays a single, centred instance — never a watermark, never repeated.
 */
export function AuthBrandHeader({ className }: { className?: string }) {
  return (
    <div className={className ?? 'mb-8 flex justify-center'}>
      <MiaLogo variant="stacked" height={92} />
    </div>
  );
}
