# AUDIT — DETTE-1 · Trois (quatre) dettes de suivi

Branche : `fix/dette-1-suivis` · worktree `wt-dette-1` · base `origin/main` (f2643bd).

---

## DETTE 1 — « AppWorkspace : skeleton pendant le fetch » — DIAGNOSTIC (aucun code)

**Test :** `webapp/components/app/__tests__/AppWorkspace.test.tsx:96` — *« shows a skeleton while the initial fetch is in flight »*.

### Conclusion en une phrase
Ce test **n'est pas l'alarme du symptôme de production**. C'est un test de rendu qui vérifie
que le squelette *apparaît* au premier chargement — il échoue pour une **pollution entre tests**
introduite par PERF-1 (cache module non réinitialisé), pas parce que l'app reste bloquée.

### a) Que teste-t-il, et sur quelle condition échoue-t-il ?
- **Ce qu'il teste :** on remplace `fetchMarketReading` par une promesse qui ne se résout jamais
  (`new Promise(() => {})`), on monte `AppWorkspace` sur la combinaison XAU/M15, et on attend que
  `data-testid="reading-skeleton"` (le composant `ReadingSkeleton.tsx`) soit à l'écran. C'est une
  assertion **positive** : « pendant un premier chargement en cours, on montre un squelette ».
- **Pourquoi il échoue (dans le fichier complet) :** le test précédent du même fichier,
  *« fetches and renders the reading for the active combo »* (ligne 82), résout `FIXTURE_XAU_M15`.
  PERF-1 a ajouté un **cache mémoire au niveau module** dans `hooks.ts`
  (`const readingCache = new Map()`), clé `live:XAUUSD:M15`, qui **survit d'un test à l'autre**.
  Au montage du test squelette, `hooks.ts` trouve la lecture en cache et fait
  `setData(cached); setIsLoading(false); setIsRefreshing(true)` (comportement SWR : afficher le
  cache + revalider en fond) → **aucun squelette**, seulement la lecture ré-affichée →
  `getByTestId('reading-skeleton')` introuvable → `waitFor` expire à 5 s → échec.
- **Preuve :** en **isolation** (`-t "shows a skeleton"`) le cache est vide → le squelette
  s'affiche → **le test passe**. Enchaîné après *« fetches and renders »* → **il échoue** (5044 ms
  de timeout). Ordre-dépendance démontrée.

### b) Reproduit-il le symptôme réel (squelettes persistants puis « données indisponibles ») ?
**Non.** Il affirme que le squelette **doit apparaître** — c'est le bon comportement d'honnêteté
de chargement, pas le défaut. Le réseau est *mocké* : le test n'emprunte jamais le chemin qui
causait le blocage de 20 s. Le symptôme de production (≈20 s de squelettes → « données
indisponibles ») était un problème **backend** (100 % des lectures cache invalidées par le bump
`READING_LOGIC_VERSION` 4→5 + `expected_close` toujours en avance → un appel fournisseur à 20 s de
timeout à chaque ouverture), traité **côté serveur** par PERF-1 (read-through `candles.db`, 404
typé). Ce test unitaire frontend n'a jamais couvert ce chemin.

### c) Si PERF-1 corrige le chargement, ce test passera-t-il au vert SANS modification ?
**Non — et c'est le point important.** PERF-1 est **déjà mergé** (#111) et il est la **cause** de
l'échec (son cache module). Aucune correction supplémentaire du *chargement* ne le fera passer :
la panne est de l'**isolation de test**, pas du comportement de chargement.

Pour le rendre vert **sans affaiblir l'assertion**, il faut ajouter au `beforeEach` du fichier
l'appel `__resetReadingRetention()` — l'helper que **PERF-1 a lui-même livré pour exactement ça**
(`hooks.ts:59`, docstring : *« Test-only: clear the retention caches… Call it in beforeEach »*),
déjà câblé dans `hooks.test.ts:19` et `useCandles.test.ts:29`. C'est un correctif de **harnais**,
pas d'assertion : l'assertion « squelette au premier chargement » reste **intégralement** en place.

> **Donc ce test ne peut PAS servir de critère de validation de PERF-1.** Il garde le *contrat de
> rendu du squelette au premier chargement* (frontend). Les vrais garde-fous de PERF-1 contre le
> symptôme de production sont : le backend `tests/test_market_reading_readthrough_perf1.py`
> (mesuré 20 s → ~0,9 s, 404 typé) et l'e2e `webapp/tests/e2e/perf1-load-honesty.spec.ts`
> (chargement complet / lent / serveur injoignable / combo sans données). **Ce sont eux qu'il faut
> exiger verts pour valider PERF-1**, pas ce test unitaire de squelette.

### d) Le lien avec la PR #111 est-il confirmé ?
**Oui, formellement.** #111 (`ad5003d`, *perf-1-chargement*) a introduit `readingCache`/`candlesCache`
au niveau module **et** l'helper `__resetReadingRetention()` (docstring « Call it in beforeEach »),
et l'a câblé dans `hooks.test.ts` / `useCandles.test.ts` — **mais n'a pas touché
`AppWorkspace.test.tsx`** (absent de son diff), dont le `beforeEach` ne fait que
`fetchMock.mockReset()`. Le test était **vert avant #111** (pas de cache). #111 l'a rendu
ordre-dépendant.

### Recommandation (hors périmètre de cette mission)
Le test est **légèrement mal écrit** : il partage un état module sans le réinitialiser. Le correctif
est **une ligne** — `__resetReadingRetention()` dans le `beforeEach` d'`AppWorkspace.test.tsx` —
et il est **indépendant de PERF-1** (hygiène de test, aucun rapport avec le comportement de
production). À traiter dans un ticket séparé, **pas ici** (consigne : ne pas corriger la dette 1).

### ⚠️ Note pour le fondateur
Si le symptôme des **20 s puis « indisponible » persiste en production**, ce n'est **pas** ce test
qui l'attrapera, et ce n'est pas non plus la preuve que PERF-1 a échoué : vérifier plutôt que #111
est **déployé** sur le backend qui sert `/app` (et le `DATA_SOURCE` réel, cf. les 2 points d'entrée).
Sujet distinct de cette dette.

---

## DETTE 2 — « market-reading : Marché en range » — CORRIGÉ

### a) Cause : vocabulaire périmé (pas un défaut de comportement)
Le test `market-reading-components.test.tsx` « reflects a ranging / low-vol regime » attendait un
badge de **tendance** « Marché en range ». Or TR-1 a retiré `ranging` de l'énum de tendance :
`TrendValue = 'bullish' | 'bearish' | 'indeterminate'` (`webapp/types/market-reading.ts:37`). Le
`MarketPhasePanel` rend `fmt.trend(regime.trend)` où `regime.trend` ne peut plus valoir `ranging`.
`FIXTURE_EUR_H1` émet aujourd'hui `trend:'indeterminate'` + `market_phase:'ranging'` +
`volatility_observed:'low'` : le « range » vit désormais dans le badge **Phase**, pas Tendance.
→ **vocabulaire périmé**, pas de vrai défaut.

### b) Correctif
Test mis à jour vers l'état réellement émis (assertion **renforcée**, jamais affaiblie) :
`Tendance indéterminée` + `Volatilité basse` + `Phase de range`. Fichier passe 23/23.

### d) Le vocabulaire de tendance « range » avait survécu ailleurs — nettoyé
Vérification demandée : 3 clés i18n de **tendance** « range » subsistaient (mortes, car
`fmt.trend`/`fmt.trendAdj` ne reçoivent que bullish/bearish/indeterminate) :
`reading.labels.trend_ranging`, `reading.labels.trendAdj_ranging`, `reading.tags.trend_ranging`.
**Supprimées des 9 locales** (27 lignes, JSON re-validé). Conservé : `reading.tags.ranging` (tag de
narration = PHASE range, réellement émis par les fixtures/mocks — pas un état de tendance). Aucun
autre test ni composant ne référence « range/ranging » comme tendance. C'était bien « deux
vocabulaires pour le même concept » que TR-1 visait ; il est maintenant unique.

---

## DETTE 4 — specs e2e chatbot / vz-1-focus — REPOINTÉES

### a) Pourquoi ignorées
LP-1 (refonte de l'accueil) a retiré la **galerie multi-marché** de la page d'accueil. Cette galerie
servait de **fixture sans backend** : elle montait `StructureSection` (VZ-1) et le bouton « Ouvrir le
chatbot… » avec une lecture d'exemple. La route n'a pas changé ; c'est le composant-fixture qui a
disparu. Les specs (skippées en LP-1) pointaient dessus.

### b) Repoint + vérification
- **vz-1-focus.spec.ts → /app** : /app rend `StructureCard` (desktop, `.zrow`) et, en < 1280,
  `ReadingColumn → StructureSection` (mobile, accordéon « Structure de marché »), **tous deux sous le
  `ChartViewProvider` partagé** (VZ-1 identique). Endpoints `**/api/market-reading**` + `**/api/candles**`
  mockés (patron PERF-1) → surface sans backend. Sélecteurs adaptés par layout. **4/4 vert** (desktop
  ×3 + mobile ×1).
- **chatbot-backend-integration.spec.ts → /app** : le chat live de /app est `AppChatSidebar`, qui
  poste sur `/api/chatbot/message` (mocké) ; les libellés `chat.inputAria`/`chat.sendAria` sont
  partagés, donc le flux est identique. `apiAvailable` démarre à `'unknown'` (pas `false`) → l'entrée
  est active en e2e. **3/3 vert** (réponse, `blocked_reason`→« Question recadrée », 503→repli).

### c) Ce qui casse — chatbot.spec.ts (scripté) : SUPPRIMÉ, pas re-skippé
`chatbot.spec.ts` testait le chat **scripté contextuel** (dialog « Ouvrir le chatbot… », réponses
canned « Cette lecture décrit un marché plutôt haussier », refus canned) — une **fonctionnalité
marketing** portée par la galerie/`ConversationReplaySection`, **retirée par LP-1** et sans équivalent
sur /app (le chat de /app parle au backend, pas de réponses scriptées). Son seul test « backend »
(texte libre) est désormais **couvert** par `chatbot-backend-integration` repointé sur /app.
→ Un test d'une fonctionnalité supprimée se **supprime** (ni skip, ni faux repoint vers un autre
composant). Le chat LIVE reste surveillé. **À confirmer** : si tu veux ré-héberger le chat scripté
marketing quelque part, on le re-testera là.

---

## DETTE 3 — repli de langue — LE PÉRIMÈTRE RÉEL EST ~100× PLUS GRAND QUE « 7 CLÉS »

### a) La prémisse « sept clés » est fausse — mesuré
La page d'accueil (`home`) compte **262 clés** ; dans les **7 locales** de/es/it/pt/nl/pl/ar, **249 sont
en anglais** (les 13 restantes sont neutres : nombres, « M.I.A Agent », « 0 $ »). Ce n'est pas 7
chaînes : c'est **249 × 7 ≈ 1 743 chaînes** rien que pour l'accueil.

### d) Et le reste du produit est logé à la même enseigne (audit des 6 autres surfaces)
Deux problèmes distincts, tous **pré-existants** (pas introduits ici) :

**(1) Valeurs en anglais** (la clé existe, le texte n'est pas traduit) — par locale, ~identique sur les 7 :
| namespace | clés | anglais/locale |
|---|---|---|
| home | 262 | **249** |
| regimePanel | 181 | **163** |
| app | 239 | ~74–80 |
| reading | 204 | ~19–24 |
| scanner | 244 | 3–13 |
| zones/calendar/landing/footer/nav/billing/auth/cookies | — | 0–4 chacun |

Total « valeurs anglaises » ≈ **3 800 chaînes** sur les 7 locales.

**(2) Clés carrément ABSENTES** (pire : le client voit la clé brute / une erreur) — `fr` a **1 879**
clés ; chaque locale non-en en **manque 196** : **scanner 98 + calendar 91** + regimePanel 5 +
reading 1 + app 1. Et **`en` a 36 clés orphelines** que `fr` n'a pas (scanner) — donc `fr` est aussi
en retard sur `en` côté scanner. Origine : refontes récentes (scanner, calendrier) propagées à
fr(+en) mais pas aux 7 autres.

### b) Traduction — NON livrée telle quelle : décision produit requise
Traduire ~3 800 valeurs + combler ~196×7 clés absentes, à la machine, sur 7 langues (dont l'arabe
RTL), sur **la page qui vend le sérieux**, produirait exactement le « mélange qui fait douter » que la
mission veut éviter. Ce n'est pas le correctif court supposé : c'est un **chantier de traduction
produit** qui mérite une vraie relecture humaine. **Je m'arrête avant de traduire en masse et je te
demande la direction** (options en fin de rapport).

### e) Test de garde — LIVRÉ (ratchet)
`webapp/lib/i18n/__tests__/locale-parity.test.ts` : parité structurelle de clés, exécuté par `npm test`
(donc en CI, pas devant un client). Comme la dette de **clés absentes** est massive et pré-existante,
le garde **ratchet** : il fixe la dette actuelle comme plafond (en 0/36 ; les 7 à 196/2) et **échoue
si l'écart CROÎT** — ajouter une clé à `fr` sans la propager casse le build. Chaque nombre du baseline
est une dette à ramener à 0. (Une détection stricte « valeur en anglais » a été différée : l'activer
avant d'avoir payé la dette rougirait la suite sur ~3 800 chaînes.)

---

## ORDRE DE LA SUITE (vitest) — moins d'échecs qu'au départ
- **Au départ** : 2 échecs (`AppWorkspace` skeleton [dette 1] + `market-reading` « Marché en range » [dette 2]).
- **Après** : **1 échec** — `AppWorkspace` skeleton, **laissé volontairement** (dette 1 = diagnostic,
  ne pas corriger ; le vrai correctif d'isolation est une ligne, hors périmètre). market-reading est
  vert. Le garde de parité (ratchet) et le test market-reading passent.
- e2e : vz-1-focus 4/4 + chatbot-backend-integration 3/3 sur /app (verts) ; chatbot.spec supprimé.

## Options pour la dette 3 (à trancher)
1. **Traduction humaine/pro** de `home` (1 743) d'abord (page de vente), puis `regimePanel`/`app`, avant facturation.
2. **Passe machine relue** : je génère les traductions des 7 locales (au moins `home`), tu fais relire.
3. **Restreindre** temporairement les locales : ne servir que fr/en tant que le reste n'est pas traduit
   (évite le mélange), rouvrir langue par langue une fois traduite + clés absentes comblées.
Dans tous les cas : **combler les 196 clés absentes** (scanner/calendar) est prioritaire — elles
montrent des clés brutes aux clients non-fr/en dès aujourd'hui.

---

## Divers — correctif tsc pré-existant (pour tenir le gate « tsc verts »)
`tsc` était **rouge sur main** avant cette mission : `tests/e2e/calendar.spec.ts:549`
(`stale_sources` inféré `never[]`, du chantier #114). Corrigé au minimum (`[] as string[]` +
`last_success as Record<string,string>`) — annotation de fixture de test, aucun changement de
comportement. Fichier non lié aux dettes, corrigé uniquement pour rendre `tsc` vert.
