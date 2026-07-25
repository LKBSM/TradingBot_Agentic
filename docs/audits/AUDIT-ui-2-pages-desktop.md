# AUDIT — UI-2 · Pages desktop (App · Scanner · Zones · Réglages · Connexion)

**Branche** `feat/ui-2-pages-desktop` (worktree dédié, depuis `origin/main` 978dc95, UI-1 incluse).
**Périmètre** Frontend uniquement. Moteur/API **intouchés**. Cible : `docs/design/reference-desktop.html`.
**Statut vérif** `tsc` 0 · **520 tests verts** (516 pré-existants + 4 honnêteté copy) · build Next vert · 3 smokes e2e verts.
**Ligne rouge** merge sur `main` **seulement après confirmation live** (backend/session requis pour les surfaces data).

---

## 0. Socle commun (réutilisé UI-1 + ajouts UI-2)

- Tokens `--bg…--fvg-l`, 4 thèmes `[data-design]`, JetBrains Mono, shell 3 colonnes, primitives `.card/.btn/.layer/.tf/.tagx/.livebadge/.ea` : **déjà livrés par UI-1** (`globals.css`, `components/shell/shell.css`). Réutilisés tels quels.
- **Créé** `components/shell/pages.css` — port verbatim des classes de **contenu** de la référence (que UI-1 avait laissées) : `.panels/.narr/.reggrid/.strow/.liqrow/.newsrow/.condpal/.combo/.zone/.tl/.fillbar/.setrow/.themetile/.pagewrap/.pghead` + classes du chat docké. Scopées sous `.app-shell`, tokens uniquement → 4 thèmes gratuits. Importé dans `ProductShell`.
- **Créé** `components/auth/login-hero.css` — port des classes `login-*` scopées sous `.login-hero` (la Connexion est sur le chrome site, pas `.app-shell`).
- **i18n** 72 clés nouvelles ajoutées aux **9 locales** (fr source + traductions en/de/es/it/pt/nl/pl/ar), fusion CRLF-préservée, **0 clé existante perdue ou modifiée** (vérifié par diff des ensembles de clés). Pas de test de parité CI, mais parité respectée.

---

## A. Page APP — `components/app/DesktopReading.tsx` (créé, desktop ≥1280px)

`ReadingColumn` étant partagé avec le mobile, la refonte est **isolée** dans un nouveau composant desktop branché par `DesktopWorkspace`. Le mobile/tablette (<1280) garde `ReadingColumn` + accordéon (forme adaptée aux petits écrans). Toutes les valeurs proviennent du `reading` typé via `useReadingFormatters` / `regime-facts` / `mtf-trend` — **aucune dérivation dupliquée du moteur, aucune donnée inventée**.

| Bloc réf | Traitement | Source données |
|---|---|---|
| Ligne de pills OB·FVG·Liquidité·BOS/CHOCH·**Mitigées** | **Créé**, câblé à `ChartViewState`. OB/FVG/Liquidité/BOS-CHOCH → `layers.*` (`set_layer_visibility`). **Mitigées** → vrai toggle sur `filter.activeOnly` inversé (`filter_zones`) — la couche existe réellement. | réel |
| Chart tokens | `ReadingChart` (lightweight-charts 5.2.0) lisait déjà `getComputedStyle` — réutilisé tel quel dans `.chartbox`. Logique data/détection non touchée. | réel |
| Lecture narrée + badge « Ancrée au moteur » + pied factuel | **Créé** `.card.wide`. Paragraphes = `conditions.description`. | réel |
| Régime de marché (6 mini-stats) | **Créé** `.reggrid`. Tendance (`regime.trend`), Volatilité, Maturité (bougies depuis rupture via `deriveTrendMaturity`), Alignement (glyphes MTF via `useMtfTrends`), Dernier évén., Densité. Champ absent → « non disponible ». | réel |
| Structure détectée | **Créé** `.strow` mono : CHOCH/BOS (tag bull/bear, direction·validation·prix), OB/FVG (band·statut). | réel |
| Poches de liquidité (Intacte/Balayée/Cassée) | **Créé** `.liqrow` : 3 états visuels + descriptions factuelles. Vide → état honnête « Aucune poche… ». | réel |
| Actus — volatilité (badge « Sans direction ») | **Créé** `.newsrow`, `news_upcoming` avec Quand + marché + Élevé/Moyen. **Aucune direction.** Vide → « Aucun événement à venir… ». | réel |

**Bloc sans donnée / traitement** : Actus et Liquidité affichent un **état vide honnête** quand le flux est vide (jamais inventé).

**Changement de comportement notable** : le panneau Régime étant toujours visible (vs accordéon fermé), `useMtfTrends` se déclenche **au chargement** (3 lectures MTF légères) au lieu d'à l'ouverture. Test `AppWorkspace` mis à jour en conséquence (mock persistant).

**Écart visuel restant — Chat M.I.A (DÉFÉRÉ, documenté)** : le chat docké reste sur le style UI-1 (cohérent tokens/thèmes) et **n'a pas été reterminalisé** aux bulles `.bub/.refuse/.chead`. Raison : la logique scroll/streaming (`useChatAnchorScroll`) est **hors périmètre** (mission séparée) et un restylage complet toucherait `ChatMessage/ChatInput/ChatWelcome` partagés avec le chat landing (risque de régression). Les classes chat de la référence sont **déjà portées dans `pages.css`**, prêtes pour une passe dédiée. Fonctionnellement intact (header, avatar, contexte mono, disclaimer 1 ligne, discussions, nouvelle discussion).

---

## B. Page SCANNER — `ScanResults.tsx` · `ComboCard.tsx` (restylés)

- **Réutilisé/restylé** : résultats en `.resgrid` de `.combo`, conditions `✓/○` (`.cl.yes/.no` + `.mk2`), bloc « Contexte » (`.ctx2` — montre aussi ce qui va à l'encontre), bouton **« Analyser »** (`.btn.acc`, deep-link App). **Aucun score, aucun classement** (affiché : « aucun classement »).
- **Créé/ajouté** : LiveBadge « Aligné sur les clôtures », note `.scannote` « Palette fermée, au présent… », carte estompée `.combo.na` « Non disponible — pas de lecture fraîche… » câblée au tableau `unavailable[]` réel + section « lecture plus ancienne » préservée.
- **Préservé** : builder de conditions fonctionnel (contrôles direction/tendance/phase/volatilité/proximité), stratégies sauvegardées, auto-refresh.
- **Écart visuel restant** : le mode **onboarding** (« Compose tes conditions ») garde ses contrôles fonctionnels — les chips `.condpal` de la référence n'existent que dans la vue résultats (les convertir supprimerait des fonctionnalités). Choix honnête assumé.

---

## C. Page ZONES — `ZoneLifecycleCard.tsx` · `ZoneTimeline.tsx` · `ZonesWorkspace.tsx` (restylés)

- **Réutilisé/restylé** : carte `.zone` (`.tagx` OB/FVG ↑↓, `.rng` mono, `.zstate` Actif/Comblé N %/Mitigée — carte `.mitig` à .6), timeline `.tl/.step`, barre `.fillbar` (FVG, `fill_level` réel), narration `.znarr`, `.zfoot` « Analyser la zone » (deep-link `?focus=`) + « Masquer du graphique » (masquage-par-id existant).
- **Bloc sans donnée / honnêteté DURE** : le moteur n'expose que `tested:boolean` + `mitigatedAt` (**un** horodatage = premier contact), aucun tableau/compte de tests. → Timeline **« Formé → Testé → Maintenant/Mitigé »** avec **une seule** étape « Testé » (**jamais « ×N »**, jamais d'horodatages multiples). Vérifié par test.

---

## D. Page RÉGLAGES — `/compte` · `AccountPanel.tsx` (restylé)

- **Réel, câblé** : courriel de session, « Modifier » email (`PATCH /api/auth/profile`), « Se déconnecter » (logout existant), « Consulter » (liens `/conditions` `/confidentialite` — pages réelles), 4 tuiles Apparence (`.themetile` via `useDesign()`/next-themes — application immédiate + persistée `data-design`), badge « Accès anticipé ».
- **Bloc sans backend → désactivé « à venir »** (aucun faux bouton) : **changement mot de passe**, **Exporter mes données**, **Supprimer le compte**. `.btn[disabled]` + note « à venir ».
- **Point légal signalé** : **Loi 25** exigera l'export + la suppression de compte **avant toute facturation** — actuellement placeholders.
- Route conservée `/compte` ; titre relabelé « Réglages ».

---

## E. Page CONNEXION — `connexion/page.tsx` · `LoginForm.tsx` · `CandleDriftCanvas.tsx` (créé)

- La page était **déjà** une carte centrée **sans panneau marketing** → rien à supprimer.
- **Réutilisé/restylé** : carte `.login-card` ~384px (brandline `.mark` + `MIA Markets` + badge `.ea`, titre, sous-titre, champs `.input`, « Mot de passe oublié ? », bouton primaire pleine largeur, « Créer un compte », encart `.trust`). **Logique d'auth (`onSubmit`) intouchée.**
- **Décision STOP appliquée** : case **« Rester connecté » omise** (aucun remember-me côté API → contrôle inerte interdit).
- **Obligatoire** : liens légaux `.under-links` sous la carte (Conditions · Confidentialité).
- **Créé** `CandleDriftCanvas.tsx` — fond animé bougies dérivantes, **sans dépendance**, DPR plafonné à 2, couleurs via tokens `--bull/--bear/--liq`, `prefers-reduced-motion` → **1 frame statique, pas de boucle** (vérifié e2e).
- **Écart assumé** : le chrome site (Nav/Footer/bannière cookies) reste autour de la carte (hors périmètre — la réf isole la carte ; la mission ne demandait que le retrait du **panneau marketing gauche**, fait).

---

## Vérifications & preuves

- `npx tsc --noEmit` → **0 erreur** (projet entier).
- `npx vitest run` → **520/520** (58 fichiers). Ajout `ui2-copy-honesty.test.ts` (4 tests) : échoue si une chaîne UI-2 promet « va rebondir / setup gagnant / cible / biais / signal d'achat / signal de vente » (sauf `scanner.note` qui les **refuse**), interdit un bouton « Trader », vérifie « Analyser ».
- `npm run build` → **compilé, toutes routes** (App 16 kB, Scanner, Zones, Compte, Connexion). Seul avertissement : symlink `.next/standalone` (spécifique worktree Windows, post-compilation, non bloquant).
- e2e `tests/e2e/ui2-pages.spec.ts` → **3/3** (Connexion : carte + badge + canvas + liens légaux ; reduced-motion statique ; App : rail + disclaimer chat).
- **Vérification visuelle** (screenshots dev, source données basculée en mock puis rétablie) : App (pills + legalbar + panneaux), Zones (timeline honnête, badges d'état), Connexion (carte + canvas) **conformes à la référence**. Réglages/Scanner-résultats derrière session/data → couverts par tests unitaires.

## Reste avant prod (non bloquant)

1. **Validation live** (backend + session) des surfaces data : App panneaux, Scanner résultats/`na`, Réglages actions réelles.
2. **Chat terminal restyle** (bulles/refus) — passe dédiée, classes déjà portées, ne pas toucher au scroll (mission séparée).
3. **Loi 25** : implémenter export + suppression de compte avant facturation (placeholders aujourd'hui).
4. Relecture native (ar RTL, pl pluriels) des 72 nouvelles clés avant prod.
