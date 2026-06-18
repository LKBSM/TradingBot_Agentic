# Correctif rendu graphique — reset auto + extent des zones (2026-06-18)

**Branche :** `feat/proto-live-tick-zones` · **Portée :** front-only (`webapp/`).
**Inchangé :** détection / structure / sécurité / prompts / comportement par défaut (sans tick live).

Deux bugs de rendu purement front dans le composant graphique (`ReadingChart`),
corrigés sans toucher au moteur ni au contrat de données.

---

## Problème 1 — RESET AUTO (vue écrasée + pic vertical à gauche)

### Cause — deux mécanismes qui se composent

**(a) Cause primaire — `fitContent()` à chaque rafraîchissement de données.**
`ReadingChart.tsx`, effet `useEffect([candles, structure])`. À chaque clôture de
bougie, `candleCloseTs` change (`lib/market-reading/hooks.ts`) → `useCandles`
renvoie un **nouveau tableau** → l'effet rejoue :

1. `series.setData(...)` — remplacement complet de la série ;
2. `chart.timeScale().fitContent()` — **réinitialise zoom + pan + échelle**.

Résultat : une bougie clôture en arrière-plan, le feed se re-tire, la vue se
rabat. « Se réinitialise sans interaction » — aucune action utilisateur.

**(b) Cause secondaire — le « pic vertical aberrant ».** Deux points d'entrée
n'avaient **aucun garde-fou de plausibilité** :

- **Tick → bougie en formation** : `series.update()` était appelé avec `livePrice`
  validé seulement par `Number.isFinite` (`lib/market-reading/live-price.ts`). Un
  `0`, un négatif ou un glitch de feed (décalage de décimale, fat-finger)
  faisait exploser le high/low de la bougie en formation.
- **Barres backend** : `setData` mappait l'OHLC tel quel, sans contrôle ; une
  seule barre corrompue (0 / négatif / `high < low`) dans le cache devenait un pic.

Une fois une valeur poubelle dans la série, le `fitContent()` suivant **recalait
l'axe pour l'inclure** → toutes les vraies bougies écrasées en une fine bande +
la barre fautive en pic. Les deux bugs s'amplifiaient.

### Correctifs

1. **Fit unique au chargement initial.** Ref `didInitialFitRef` : `fitContent()`
   uniquement au premier rendu (ou recréation du chart sur changement de thème).
   Ensuite, le `visibleLogicalRange` est **capturé avant `setData` et restauré
   après**, donc un rafraîchissement de fond ne touche plus au zoom/pan. Le bouton
   « Ajuster » reste le seul refit explicite.
2. **Garde-fou barres** — `isValidBar` (`lib/chart/sanitize.ts`) : OHLC fini & > 0,
   `low ≤ high`, open/close dans `[low, high]`. Les barres corrompues sont filtrées
   avant `setData` et le nombre de rejets est logué (`console.warn`).
3. **Garde-fou tick** — `isPlausibleTick(price, ref, maxDevPct = 0.5)` : rejette
   `0` / négatif / non-fini / écart > 50 % vs dernier close (glitch de feed) AVANT
   de grossir la bougie en formation. Un vrai gros mouvement reste très en-deçà de
   50 % sur un seul pas de temps et **passe** — on rejette l'erreur de données,
   jamais la volatilité réelle. Rejet logué.
4. **Mise à jour incrémentale** de la bougie courante au tick (`series.update`,
   déjà en place) — pas de `setData` remplaçant toute la série à chaque tick.

---

## Problème 2 — EXTENT DES ZONES (active s'arrête trop tôt à droite)

### Cause

L'anchrage de droite valait `mitigatedSec` (zone testée) **sinon `lastTime`** =
dernière bougie **clôturée**. Quand un tick live coule, la bougie courante / en
formation est à `slot > lastTime` (barre la plus à droite). Une zone **ACTIVE**
s'arrêtait donc une barre trop tôt.

`mitigated_at` distingue bien active vs mitigée (`lib/chart/zoneLayout.ts`,
`buildZoneModels`) : `mitigatedSec: null` + `tested:false` quand absent, sinon
`tested:true`. Logique correcte, seul l'anchrage de droite était en cause.

### Correctif

Fonction pure `zoneEndSec(zone, { lastBarSec, formingSec })` (`zoneLayout.ts`) :

- zone **mitigée** (testée + point de mitigation connu) → `mitigatedSec`
  (**bornée, honnête, jamais sur-étendue** au-delà d'un résultat résolu) ;
- zone **active** (et testée sans point de mitigation exploitable) → `formingSec`
  (bougie live en formation) **sinon** `lastBarSec` (dernière clôturée).

`computeRects` lit la bougie en formation (`formingRef.current?.time`) chaque
frame, l'ajoute aux candidats de snap, et l'utilise comme bord droit des zones
actives — elles atteignent la bougie courante / le prix actuel, **sans jamais
projeter dans le futur vide au-delà**. Sans tick (mode par défaut), `formingSec`
est `null` → bord droit = dernière bougie clôturée : **vue par défaut préservée
à l'identique**.

---

## Tests & vérifications

- **`lib/chart/__tests__/sanitize.test.ts`** (nouveau) : `isValidBar` rejette
  0/négatif/NaN/`high<low`/open-close hors bande ; `isPlausibleTick` rejette
  garbage mais **accepte un vrai gros mouvement** (+40 %).
- **`lib/chart/__tests__/zoneLayout.test.ts`** : `zoneEndSec` — zone active
  s'étend à la bougie live (et à la dernière clôturée sans tick) ; zone mitigée
  **reste bornée** à son point de mitigation.
- `npx vitest run` → **27 fichiers / 205 tests verts**.
- `npx tsc --noEmit` → **0 erreur**.
- `npx next build` → **vert**.

## Fichiers touchés

- `webapp/lib/chart/sanitize.ts` (nouveau) — `isValidBar`, `isPlausibleTick`.
- `webapp/lib/chart/zoneLayout.ts` — `zoneEndSec`.
- `webapp/components/app/ReadingChart.tsx` — fit initial-only + restauration de
  vue, filtre barres, garde-fou tick, bord droit zones actives = bougie live.
- `webapp/lib/chart/__tests__/sanitize.test.ts` (nouveau).
- `webapp/lib/chart/__tests__/zoneLayout.test.ts` — bloc `zoneEndSec`.
