/**
 * NW-5 — "Aller à la source" : jusqu'à quatre liens NOMMÉS par publication vers
 * des documents précis de l'organisme émetteur, et lui seul.
 *
 * Chaque lien pointe UNIQUEMENT vers le domaine de l'organisme qui émet la
 * publication (bls.gov, bea.gov, census.gov, federalreserve.gov, ec.europa.eu,
 * ecb.europa.eu). Une URL précise inconnue n'est pas remplacée par un lien
 * générique : le lien est simplement absent (voir la règle E de la mission).
 *
 * L'intitulé et la description de chaque lien viennent de l'i18n
 * (`calendar.pub.source.docs.<kind>`), pas d'ici — ce fichier ne fige que les
 * URLs, auditables ligne par ligne. Le test de conformité vérifie que chaque
 * hôte appartient bien au domaine de l'organisme.
 */

/** The four kinds of official document a publication can link to. */
export type SourceDocKind = 'methodology' | 'release' | 'series' | 'schedule';

/** Order in which links are rendered when present. */
export const SOURCE_DOC_ORDER: readonly SourceDocKind[] = [
  'methodology',
  'release',
  'series',
  'schedule',
] as const;

/** Allowed host suffix per organism source — the conformity test enforces it. */
export const SOURCE_DOMAINS: Record<string, string> = {
  bls: 'bls.gov',
  bea: 'bea.gov',
  census: 'census.gov',
  federal_reserve: 'federalreserve.gov',
  eurostat: 'ec.europa.eu',
  ecb: 'ecb.europa.eu',
};

/**
 * Named links per recurring event key. Only URLs verified to live on the issuing
 * organism's domain are listed; a missing kind simply renders one fewer link.
 * Start with the publications whose official document pages are known (mission:
 * « commence par le périmètre actuel »).
 */
export const SOURCE_LINKS: Record<string, Partial<Record<SourceDocKind, string>>> = {
  us_cpi: {
    methodology: 'https://www.bls.gov/opub/hom/cpi/',
    release: 'https://www.bls.gov/news.release/cpi.htm',
    series: 'https://www.bls.gov/cpi/data.htm',
    schedule: 'https://www.bls.gov/schedule/news_release/cpi.htm',
  },
  us_employment_situation: {
    methodology: 'https://www.bls.gov/opub/hom/ces/',
    release: 'https://www.bls.gov/news.release/empsit.htm',
    series: 'https://www.bls.gov/ces/data/',
    schedule: 'https://www.bls.gov/schedule/news_release/empsit.htm',
  },
  us_ppi: {
    methodology: 'https://www.bls.gov/opub/hom/ppi/',
    release: 'https://www.bls.gov/news.release/ppi.htm',
    series: 'https://www.bls.gov/ppi/data.htm',
    schedule: 'https://www.bls.gov/schedule/news_release/ppi.htm',
  },
  us_gdp: {
    methodology: 'https://www.bea.gov/resources/methodologies/nipa-handbook',
    release: 'https://www.bea.gov/data/gdp/gross-domestic-product',
    schedule: 'https://www.bea.gov/news/schedule',
  },
  us_pce: {
    methodology: 'https://www.bea.gov/resources/methodologies/nipa-handbook',
    release: 'https://www.bea.gov/data/income-saving/personal-income',
    schedule: 'https://www.bea.gov/news/schedule',
  },
  us_retail_sales: {
    methodology: 'https://www.census.gov/retail/marts/how_surveys_are_collected.html',
    release: 'https://www.census.gov/retail/index.html',
    schedule: 'https://www.census.gov/economic-indicators/calendar-listview.html',
  },
  us_housing_starts: {
    methodology: 'https://www.census.gov/construction/nrc/methodology.html',
    release: 'https://www.census.gov/construction/nrc/index.html',
    schedule: 'https://www.census.gov/economic-indicators/calendar-listview.html',
  },
  us_durable_goods: {
    methodology: 'https://www.census.gov/manufacturing/m3/technical_documentation/methodology/index.html',
    release: 'https://www.census.gov/manufacturing/m3/index.html',
    schedule: 'https://www.census.gov/economic-indicators/calendar-listview.html',
  },
  us_fomc_rate: {
    methodology: 'https://www.federalreserve.gov/monetarypolicy/fomc.htm',
    release: 'https://www.federalreserve.gov/newsevents/pressreleases.htm',
    schedule: 'https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm',
  },
  ea_hicp_flash: {
    methodology: 'https://ec.europa.eu/eurostat/cache/metadata/en/prc_hicp_esms.htm',
    series: 'https://ec.europa.eu/eurostat/databrowser/view/prc_hicp_manr/default/table',
    schedule: 'https://ec.europa.eu/eurostat/web/main/news/release-calendar',
  },
  ea_gdp_flash: {
    methodology: 'https://ec.europa.eu/eurostat/cache/metadata/en/namq_10_esms.htm',
    series: 'https://ec.europa.eu/eurostat/databrowser/view/namq_10_gdp/default/table',
    schedule: 'https://ec.europa.eu/eurostat/web/main/news/release-calendar',
  },
  ea_unemployment: {
    methodology: 'https://ec.europa.eu/eurostat/cache/metadata/en/une_esms.htm',
    series: 'https://ec.europa.eu/eurostat/databrowser/view/une_rt_m/default/table',
    schedule: 'https://ec.europa.eu/eurostat/web/main/news/release-calendar',
  },
  ea_ecb_rate: {
    methodology: 'https://www.ecb.europa.eu/mopo/implement/html/index.en.html',
    release: 'https://www.ecb.europa.eu/press/pr/date/html/index.en.html',
    schedule: 'https://www.ecb.europa.eu/press/calendars/mgcgc/html/index.en.html',
  },
};

/** Ordered, present-only links for one event key (empty ⇒ render no link list). */
export function sourceLinksFor(
  eventKey: string | null,
): Array<{ kind: SourceDocKind; url: string }> {
  if (!eventKey) return [];
  const entry = SOURCE_LINKS[eventKey];
  if (!entry) return [];
  return SOURCE_DOC_ORDER.filter((k) => entry[k]).map((k) => ({
    kind: k,
    url: entry[k] as string,
  }));
}
