import { useTranslations } from 'next-intl';
import { Hourglass } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent } from '@/components/ui/card';

interface ComingSoonCardProps {
  /** Instrument ticker (proper noun, not translated). */
  label: string;
}

/**
 * Placeholder grisé "Bientôt" — montre que MIA reconnaît qu'il y a
 * d'autres marchés sans bluffer sur leur disponibilité. Aligné D4
 * instruments (XAU + EUR seuls en GA, le reste post-S16).
 */
export function ComingSoonCard({ label }: ComingSoonCardProps) {
  const t = useTranslations('landing.comingSoon');
  const subtitle = t('subtitle');
  return (
    <Card
      aria-label={`${label} — ${subtitle}`}
      className="flex h-full flex-col border-dashed bg-muted/30 opacity-70"
    >
      <CardContent className="flex flex-1 flex-col items-center justify-center gap-3 p-6 text-center">
        <div className="flex h-10 w-10 items-center justify-center rounded-full bg-muted text-muted-foreground">
          <Hourglass className="h-5 w-5" aria-hidden />
        </div>
        <p className="text-lg font-semibold text-foreground/80">{label}</p>
        <Badge variant="outline" className="text-[10px] uppercase tracking-wider">
          {subtitle}
        </Badge>
        <p className="text-xs text-muted-foreground">{t('note')}</p>
      </CardContent>
    </Card>
  );
}
