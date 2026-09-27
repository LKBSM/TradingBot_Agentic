'use client';

import { useTranslations } from 'next-intl';
import { useCallback } from 'react';
import { formatInstrument } from './formatters';

/**
 * Le nom d'un marché, dans la langue de la page.
 *
 * POURQUOI CE CROCHET EXISTE
 * `formatInstrument()` renvoie le libellé « FR de base » du registre
 * (`config/markets.json`) : « Livre / Dollar (GBP/USD) ». C'est du français, et
 * il s'affichait tel quel sur /en. Avec 2 marchés c'était deux mots ; avec 80,
 * c'est toute la barre latérale d'un client anglophone qui est en français.
 * Le code promettait depuis longtemps une clé i18n « pour les composants
 * multilingues » — elle n'existait pas. La voici.
 *
 * CE QUE RENVOIE LA CLÉ
 *   change et cryptos → le code de paire : « GBP/USD », « BTC/USD ».
 *     Neutre dans toutes les langues, et c'est ainsi que ces instruments
 *     s'écrivent partout dans le métier. Rien à traduire, rien à faire dériver.
 *   métaux → le nom traduit : « Or » / « Gold », « Argent » / « Silver »,
 *     avec la paire en suffixe quand la cotation n'est pas en dollars
 *     (« Or (XAU/EUR) »).
 *
 * Repli : si la clé manque, on retombe sur le libellé du registre plutôt que
 * d'afficher un identifiant brut ou une clé i18n crue.
 *
 * NB sur l'espace de noms : les libellés vivent sous `calendar.market.*` pour
 * des raisons historiques (ils y sont nés). Ils sont en réalité valables dans
 * tout le produit. Les déplacer sous un espace `markets` serait plus juste,
 * mais toucherait dix appels du calendrier sans rien changer à l'écran — dette
 * cosmétique assumée, pas une urgence.
 */
export function useInstrumentLabel(): (instrument: string) => string {
  const t = useTranslations('calendar.market');
  return useCallback(
    (instrument: string): string => {
      if (!instrument) return '';
      try {
        const label = t(instrument as 'XAUUSD');
        // next-intl renvoie la clé elle-même quand elle est absente.
        if (label && label !== instrument) return label;
      } catch {
        // Clé absente : on préfère le libellé du registre à une clé crue.
      }
      return formatInstrument(instrument);
    },
    [t],
  );
}
