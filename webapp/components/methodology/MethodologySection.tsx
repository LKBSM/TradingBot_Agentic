/**
 * Wrapper de section pour la page /methodology (Chantier 5.D).
 *
 * Pose une ancre (scroll depuis les tooltips ⓘ et le sommaire) + un titre + une
 * intro optionnelle, puis rend ses enfants. Purement présentationnel.
 */
export function MethodologySection({
  id,
  title,
  intro,
  children,
}: {
  id: string;
  title: string;
  intro?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section
      id={id}
      aria-labelledby={`${id}-title`}
      className="scroll-mt-24 border-t border-border/60 py-12 first:border-t-0"
    >
      <h2
        id={`${id}-title`}
        className="text-balance text-xl font-semibold tracking-tight sm:text-2xl"
      >
        {title}
      </h2>
      {intro && (
        <p className="mt-3 max-w-2xl text-pretty text-muted-foreground">
          {intro}
        </p>
      )}
      <div className="mt-6">{children}</div>
    </section>
  );
}
