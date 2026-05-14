import { useTranslations } from 'next-intl';
import { GLOSSARY_TERMS } from '@/lib/glossary';

export default function GlossaryPage() {
  const t = useTranslations('glossary');
  return (
    <div className="container-prose py-12">
      <h1 className="text-3xl font-bold">{t('title')}</h1>
      <p className="mt-2 text-slate-600">{t('subtitle')}</p>

      <dl className="mt-8 divide-y divide-slate-200">
        {GLOSSARY_TERMS.map((term) => (
          <div key={term.term} className="py-4">
            <dt className="font-semibold">{term.term}</dt>
            <dd className="mt-1 text-sm text-slate-700">{term.definition_fr}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
