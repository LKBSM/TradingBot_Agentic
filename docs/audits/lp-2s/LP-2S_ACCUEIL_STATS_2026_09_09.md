# LP-2S — Accueil : retrait du bloc de statistiques secondaires

**Date** : 2026-09-09
**Branche** : `feat/lp2s-accueil-stats` (worktree dédié `C:\MyPythonProjects\wt-lp-2s`)
**Base** : `origin/main` @ `54265ef` (PR #200 MIA-3), à jour au `git fetch`
**Écart au départ** : le HEAD du dépôt principal (`docs/preserve-data-1-audit` @ `e0dc69c`)
était **57 commits derrière** `origin/main` — le worktree part de `origin/main`.

---

## 1. Ce qui a été retiré

La bannière de quatre tuiles du hero — **2 marchés suivis · 6 unités de temps ·
22 conditions de recherche · 7 structures détectées** — et la ligne
« Accès anticipé · 2 marchés aujourd'hui, 50 à 80 prévus au lancement ».

Motif : aucune de ces quatre valeurs ne dit à un visiteur qui découvre le produit
s'il doit s'y intéresser. « 22 conditions de recherche » ne se compare à rien.

## 2. Ce qui la remplace

Une seule ligne, à la même position, dans la typographie du texte environnant
(`.roadmap` : mono 12 px `var(--faint)` — le style du `.micro` juste au-dessus,
donc pas de nouveau bloc visuel) :

> Jusqu'à 80 marchés au programme — **XAUUSD et EURUSD disponibles dès aujourd'hui.**

La **partie réelle** est celle mise en gras (`.roadmap b` → `var(--dim)`, plus
contrasté). C'est délibéré : le fait vérifiable ressort, l'ambition reste en
retrait. Le contraire aurait vendu l'objectif.

## 3. La ligne de conduite, tenue par un test

**80 est une ambition, pas un état.** La règle : ce nombre ne doit jamais
apparaître sans son qualificatif de portée, dans aucune des 9 langues.

Un garde-fou existait déjà — `home.test.tsx` interdisait purement et simplement
`/80\s*\+?\s*march/i`, hérité de LP-2 où la maquette annonçait « 80 marchés »
comme périmètre réel. **Ce garde bloquait la phrase demandée** : il est lexical
et ne sait pas distinguer un nombre nu d'un nombre qualifié. Il a été **resserré,
pas supprimé** — l'interdiction de `480` reste intacte.

Le nouveau garde (`bareMarketCounts`, 9 locales) autorise un décompte de marchés
dans exactement deux cas :

1. il **égale le périmètre réellement en production** (`ALL_MARKET_IDS.length`,
   lu depuis le registre MKT-1) — c'est alors un fait vérifiable, il n'a besoin
   d'aucune précaution (« Les 2 marchés », dans le bloc tarifs) ;
2. c'est un autre nombre **et** un mot de portée figure dans **le même segment de
   phrase** — il se lit alors comme l'ambition qu'il est.

Tout le reste est un nombre futur porté au présent. Le mot de portée doit être
dans le même segment : un qualificatif situé trois propositions plus loin ne
blanchit pas un décompte nu.

Deux raffinements imposés par le réel, chacun un vrai cas rencontré :

- **« Les 2 marchés »** (bloc tarifs) déclenchait le garde brut. C'est le
  périmètre réel énoncé comme fait — d'où la règle 1, comparée **au registre et
  non à un `2` en dur** : le jour où un 3ᵉ marché entre en production, cette
  phrase-là devient fausse et le test rougit.
- **« 12 market-and-timeframe combinations »** (en) déclenchait aussi : « market »
  y est un adjectif composé et compte des *combinaisons*. Le garde exige désormais
  que le mot soit un nom (`(?![\w-])`).

Enfin, un test verrouille **le garde lui-même** : les 9 phrases telles qu'un
traducteur les produit en « simplifiant » (« 80 marchés — XAUUSD et EURUSD… »,
qualificatif tombé) doivent toutes être attrapées, chacune dans sa langue. Un
garde-fou qui ne peut pas échouer ne protège rien.

## 4. Les 9 locales

| locale | ligne | qualificatif |
|---|---|---|
| fr | Jusqu'à 80 marchés **au programme** — XAUUSD et EURUSD disponibles dès aujourd'hui. | au programme |
| en | Up to 80 markets **planned** — XAUUSD and EURUSD available today. | planned |
| de | Bis zu 80 Märkte **geplant** — XAUUSD und EURUSD ab heute verfügbar. | geplant |
| es | Hasta 80 mercados **previstos** — XAUUSD y EURUSD disponibles desde hoy. | previstos |
| it | Fino a 80 mercati **previsti** — XAUUSD ed EURUSD disponibili da oggi. | previsti |
| pt | Até 80 mercados **previstos** — XAUUSD e EURUSD disponíveis desde hoje. | previstos |
| nl | Tot 80 markten **gepland** — XAUUSD en EURUSD vandaag al beschikbaar. | gepland |
| pl | **Docelowo** do 80 rynków **w planie** — XAUUSD i EURUSD dostępne już dziś. | docelowo / w planie |
| ar | ما يصل إلى 80 سوقًا **مخطَّطة** — XAUUSD وEURUSD متاحان اليوم. | مخطَّطة |

Le garde arabe neutralise les harakat avant comparaison, pour matcher le mot et
non une vocalisation particulière.

## 5. Fichiers touchés

| fichier | action |
|---|---|
| `components/landing/lp1/HomeLanding.tsx` | bloc `.stats` + import retirés ; `.roadmap` conservé |
| `components/landing/lp1/lp1.module.css` | `.stats/.stat/.statN/.statL` + media query retirés ; `.roadmap` `margin-top` 20 → 28 px (le bloc à 48 px qui l'espaçait a disparu) |
| `lib/landing/stats.ts` | **supprimé** — plus aucun consommateur après le retrait |
| `app/[locale]/(site)/page.tsx` | commentaire de tête mis à jour (il pointait le module supprimé) |
| `messages/*.json` × 9 | `home.stats` (5 clés) retiré ; `home.hero.roadmap` réécrit |
| `components/landing/lp1/__tests__/home.test.tsx` | garde-fou « jamais un nombre nu » + auto-test du garde |
| `tests/e2e/lp1-accueil.spec.ts` | `statStructures` → `marketsLine` + `marketsTickers` (fr, en) ; `tryFree` → `subscribeCta` (§7) |
| `tests/e2e/lp2s-shots.spec.ts` | **nouveau** — captures de preuve aux 2 viewports |

`home.stats.combinations` était déjà orpheline au rendu avant cette mission
(hors des 4 tuiles) ; elle part avec le reste. Aucune autre page n'utilisait
`home.stats.*` ni `home.hero.roadmap` — vérifié sur tout `webapp/`.

## 6. Vérifications

| contrôle | résultat |
|---|---|
| `tsc --noEmit` | **0 nouvelle erreur** (3 restantes = `dictation-copy-honesty`, pré-existantes) |
| `next build` (CI=1) | **vert** |
| vitest `home.test.tsx` | **18/18** |
| vitest `claims-cleanup` + `ui2b-i18n-keys` + `ui2-copy-honesty` | **25/25** |
| Playwright `lp1-accueil.spec.ts` (1280×800 + 390×844, fr + en) | **56/56** (après le correctif du §7) |
| Playwright `lp2s-shots.spec.ts` | **4/4**, captures dans `shots/` |

## 7. Un échec pré-existant, corrigé sur demande

Les 4 échecs de la première passe portaient tous sur `nav bar: a visitor gets no
App/Zones/Scanner, sees the free-trial CTA` — un test du **header**, sans lien
avec le hero, rouge avant cette mission.

Cause : le spec attendait `/Essayer gratuitement/i` (fr) et `/Try for free/i`
(en), alors que `nav.tryFree` vaut aujourd'hui **« S'abonner » / « Subscribe »**.
Le CTA a été renommé au passage à l'abonnement ; le test n'a pas suivi.

Corrigé (le fondateur l'a demandé après la première livraison) :

- `tryFree` → **`subscribeCta`**, nommé pour ce que le bouton fait. Le produit ne
  propose pas d'essai gratuit — garder « tryFree » dans le spec aurait reconduit
  la description fausse que le renommage du CTA venait justement de corriger.
- regex **ancrées** `/^S'abonner$/` et `/^Subscribe$/`, sur le libellé exact.
  Apostrophe **droite** (U+0027) : c'est ce que porte le fichier de messages.
- titre du test : « sees the subscribe CTA ».

**Laissé en l'état** : la clé i18n s'appelle toujours `nav.tryFree` alors qu'elle
rend « S'abonner ». La renommer touche les 9 locales et tous ses appelants —
c'est un changement à part entière, pas un à-côté de LP-2S. Un commentaire dans
le spec le signale à qui passera par là.

## 8. Captures — `shots/`

`hero-{fr,en}-{desktop-1280x800,mobile-390x844}.png` (le hero complet, bannière
de consentement écartée) et `line-{...}.png` (la ligne seule, en gros plan).

Piège rencontré et neutralisé dans le spec : à 390×844 la bannière cookies
recouvre le bas du hero, exactement là où la ligne se trouve. `toBeVisible()`
passait quand même — Playwright ne teste pas l'occlusion. **C'est la capture, pas
l'assertion, qui a révélé le problème.** Le helper partagé `dismissCookieBanner`
ne connaît que le libellé français ; `/en` est traité dans le spec plutôt que
d'élargir un helper dont dépendent les autres specs.

## 9. Points laissés au fondateur

**Redondance avec le badge du hero.** Trois lignes plus haut, `home.hero.pill`
affiche déjà « Accès anticipé · l'or et l'euro aujourd'hui ». La nouvelle ligne
redit le même périmètre en tickers. Ce n'est pas une faute (badge court vs ligne
d'ambition, l'un en langage courant, l'autre en symboles) et la mission ne le
visait pas — non touché.

**Le hero est plus aéré qu'avant.** Le bloc de quatre tuiles occupait un volume
réel ; une ligne de texte ne le remplace pas visuellement, et c'était l'intention.
À confirmer à l'œil sur les captures.
