# AUDIT — LP-1 · Refonte de la page d'accueil avec démonstrations interactives

Branche : `feat/lp-1-page-accueil` · worktree `wt-lp-1` · base `origin/main` (d90cbd3)
Maquette de référence : `docs/design/reference-accueil.html` (mia-page-accueil-v2, poussée sur main 548f6bb)

---

## 1. La découverte centrale : le bandeau de la maquette était de la fiction

La maquette vendait **« 80 marchés · 480 combinaisons · 21 conditions · 6 unités »**. Trois de ces
quatre chiffres étaient faux. Vérifié dans le code :

| Maquette | Réalité | Source |
|---|---|---|
| 80+ marchés | **2 marchés** (Or XAU/USD, Euro EUR/USD) | `webapp/lib/market-reading/perimeter.ts:12`, `config/lookback_depths.json` |
| 480 combinaisons | **12** (2 × 6 unités) | `enabled_combos()` — `src/intelligence/lookback_config.py:193` |
| 21 conditions | **22** | `webapp/lib/conditions/palette.ts` (22 `type:`) |
| 6 unités | ✅ 6 (M1–D1, `perimeter:true`) | `config/timeframes.json` |

Toute la page a été recadrée sur ces chiffres. Le discours passe de « nombre de marchés » à
« profondeur par marché ». **Source unique** des chiffres : `webapp/lib/landing/stats.ts`
(`LANDING_STATS`), consommée par le bandeau et vérifiée par un test.

---

## 2. Inventaire des capacités réelles (résumé)

Diagnostic exhaustif produit en phase 1 (message de STOP). Surfaces couvertes :

- **/app** — 5 couches masquables (OB, FVG, Liquidité, BOS/CHOCH, Mitigées) ; **panneau Régime à
  10 tuiles** avec onglets Donnée/Concept + « ce que ça ne dit pas » ; lecture narrée « ancrée au
  moteur » ; cadrage caméra unifié VZ-1 ; badges d'état (EN DIRECT / Marché fermé / Pause / en retard).
- **/scanner** — 22 conditions en 4 familles (paramétrées) ; carte à 4 blocs dont « Ce qui va à
  l'encontre » (jamais masquable) et « Non évaluable » ; états 0-condition / 0-résultat honnêtes,
  « pas d'assouplissement » ; lectures enregistrées avec export/import ; compteur live ; auto-refresh.
- **/zones** — fiche cycle de vie (formée→testée→comblée, horodatée), comblement FVG à un prix,
  recoupements inter-unités, masquer/afficher par zone, 3 tris (Proximité/Fraîcheur/État).
- **/actualites** — calendrier mensuel (heure locale + organisme, filtres factuels) ; page de
  publication : courbe 12 derniers chiffres, 3 états d'absence, révisions, **3 mesures moteur**
  (calme avant / état structure / retour au calme) en tranches+extrêmes+dénominateur.
- **M.I.A** — présente sur /app, landing, page de publication ; `apply_chart_view` = **12 actions**
  (masquer/isoler/afficher par id OU catégorie, filtrer, focus, changer d'unité…) ; `get_ob_diagnostic`
  factuel ; 3 couches de refus (adverse / sortie / whitelist), tout id ré-vérifié contre le moteur.

---

## 3. Démonstrations interactives livrées (le cœur de la mission)

Cinq démonstrations, **100 % locales** (données figées, aucun appel réseau, aucune session).
Fichiers : `webapp/components/landing/lp1/DemoTabs.tsx` + `data.ts` + `chart.ts` + `CandleSvg.tsx`.
Mention « Données d'illustration » visible sur chaque démonstration.

1. **Lire une structure** — 4 couches masquables (chips) ; la narration se **recompose** à partir des
   couches restantes (fragments moteur réordonnés) ; état vide honnête (« n'invente rien pour remplir
   le vide ») ; boutons « ne garder que… ».
2. **Chercher un marché** — 5 conditions réelles à cocher ; liste qui se resserre ; **deux états
   conservés** : 0 condition → « pas tous les marchés », 0 résultat → « ce n'est pas une erreur »
   (sans jamais proposer d'assouplir) ; bloc **« Ce qui va à l'encontre »**.
3. **Suivre une zone** — 3 zones (vierge / entamée / comblée) ; frise de vie ; **taux de remplissage
   animé** (0→60 %) ; action **« Masquer du graphique »** avec note.
4. **Poser une question** — 4 questions : concept, description ancrée, **action qui modifie réellement
   les couches de la démo 1** (« montre-moi seulement les OB non testés » → aperçu du graphique
   recomposé + lien vers l'onglet Structure), et un **refus** de prédiction.
5. **Ouvrir le calcul** (5ᵉ démo ajoutée) — une tuile de Régime bascule du verdict (« Volatilité :
   Normale ») vers les **nombres bruts** (parcours récent vs référence, rapport, seuils) + « ce que
   ça ne dit pas ». C'est la démonstration la plus fidèle de ce qui distingue le produit :
   *chaque affirmation est vérifiable*.

Accessibilité : rôles `tab`/`tabpanel`, `aria-pressed`, cibles ≥ 44 px, focus visibles, animations
douces respectant `prefers-reduced-motion`.

---

## 4. Capacités réelles que la maquette ignorait (corrigées vers le produit)

- Régime **10 tuiles** (la maquette en citait 4) → démo 5 + puce « app.b4 ».
- M.I.A pilote le graphique de **12 façons** (pas seulement « OB non testés ») → puce + démo 4.
- **`get_ob_diagnostic`** (pourquoi une bougie est/n'est pas un OB) → évoqué section M.I.A.
- Scanner : bloc **« Ce qui va à l'encontre »**, **lectures enregistrées export/import** → puces + démo 2.
- Zones : **recoupements inter-unités**, **masquer/afficher par zone** → puces + démo 3.
- **Badges d'honnêteté d'état** (Marché fermé, en retard) et **vérifiabilité** (Donnée/Concept) → démo 5.

## 5. Corrections factuelles portées à la maquette (réalité > joli)

- Bandeau : 2 / 12 / 22 / 6 (voir §1).
- **Zones : compteur de touches retiré.** Le produit ne suit qu'un booléen `tested` (le compteur de
  touches est une mission différée). La maquette montrait « testée 2 fois » / « combien de fois
  touché » → **remplacé** par des libellés booléens (formée → pénétrée → comblée).
- **M.I.A « le scanner » retiré.** La maquette disait « disponible sur le graphique, le scanner et les
  publications ». M.I.A n'est **pas** sur /scanner ni /zones → corrigé en « le graphique et les publications ».
- « à chaque clôture, sans cliquer » → l'auto-refresh est opt-in ; formulation nuancée.

---

## 6. FAQ — réponses vérifiées contre le produit

| # | Question | Correction |
|---|---|---|
| Prix | 39 $/mois, 348 $/an (29 $/mois) | Conforme à la mission (voir §8 conflit) |
| Annulation « en un clic » | **Corrigé** en « depuis ton compte, sans écrire à personne » — Stripe est en cours d'intégration, l'affirmation « un clic » n'est pas encore garantie |
| Mobile | **Vrai** (shell responsive) — conservé |
| « 80+ marchés » | **Corrigé** → 2 marchés, « d'autres suivront » |
| Sources officielles | Vrai (BLS/BEA/Census/Fed/Eurostat/BCE) — conservé ; fetchers BEA/Census non écrits mais l'UI le dit honnêtement |
| Signaux d'entrée | Reformulé sans le mot interdit « signal » (voir §7) : « Est-ce que MIA dit quand acheter ou vendre ? » |

---

## 7. Vocabulaire interdit — décision sur « signal »

La mission §E interdit le mot « signal » ; la mission §F demande une mention légale « sans conseil ni
recommandation ni signal ». **Conflit littéral.** Résolution : la prohibition §E l'emporte ; l'intention
de §F (aucun conseil d'intervention) est satisfaite par un synonyme — la mention légale de l'accueil dit
« ni conseil en investissement, ni recommandation, **ni indication d'intervention** sur les marchés ».
Le mot « signal » n'apparaît nulle part dans le namespace `home` (fr et en) ; un test le vérifie dans
les deux langues, avec les autres mots interdits (setup, opportunité, gagnant, probabilité, meilleur
moment, ne rate plus, gain, rendement, réussite / EN : signal, setup, opportunity, winner, winning,
probability, best time, don't miss, guaranteed, profit).
NB : le pied de page global (`Footer`, hors périmètre `home`) conserve sa mention légale existante.

---

## 8. Écarts assumés & points à trancher

- **Prix vs page /abonnement.** L'accueil livre 39/348/29 $ (mission). La page `/abonnement` et l'i18n
  `landing.pricing` existants affichent encore **49,99/39,99 $**. À réconcilier — non touché ici pour
  rester dans le périmètre de l'accueil.
- **i18n 7 locales.** fr + en sont natifs et complets (exigence mission). de/es/it/pt/nl/pl/ar reçoivent
  le bloc **en** en repli (documenté, sans clé brute ni plantage) — traduction native à faire en suivi.
- **Blast radius e2e.** La refonte retire du homepage la galerie multi-marché qui servait de fixture
  sans backend à d'autres suites. Traitement :
  - `landing.spec.ts` et `sections.spec.ts` (tests de l'ancien accueil) → **supprimés**, remplacés par
    `lp1-accueil.spec.ts`.
  - `chatbot.spec.ts`, `chatbot-backend-integration.spec.ts`, `vz-1-focus.spec.ts` ouvraient le chat /
    la structure via la galerie → **skippés** avec raison explicite (à **repointer vers /app**, suivi).
- **Anciennes sections landing.** Les composants `components/landing/*` (HeroLive, gallery, etc.) ne sont
  plus référencés par l'accueil ; laissés en place (non supprimés) pour ne rien casser d'autre.

---

## 9. Vérifications

- `npm run typecheck` — ✅ vert.
- `npm run build` — ✅ compilé (l'`EPERM symlink` final = artefact de la jonction `node_modules` du
  worktree sur l'étape `output: standalone`, exit 0, non bloquant).
- Vitest LP-1 (`components/landing/lp1/__tests__/home.test.tsx`) — ✅ **12/12** : vocabulaire interdit
  (fr+en), chiffres du bandeau depuis la source unique, absence de « 480 »/« 80 marchés », mention
  « données d'illustration », devise sur chaque prix, mentions légales (fr+en), démos hors-ligne
  (fetch espionné → jamais appelé), action M.I.A qui change les couches, révélation du calcul.
- Vitest suite complète — 728 passés. **1 échec pré-existant** hors périmètre
  (`market-reading-components.test.tsx` « Marché en range » : dérive libellé/test présente sur
  `origin/main`, aucun fichier market-reading modifié par LP-1).
- Playwright `lp1-accueil.spec.ts` — ✅ **40/40** (fr + en × 1280×800 + 390×844 × {page complète,
  démos 1/2/4/5 en deux états, tarif}). Servi via `next dev` (le proxy `/api/auth/me`→:8000 échoue sans
  backend, sans incidence — la page est statique).

---

## 10. Fichiers

Nouveaux : `webapp/lib/landing/stats.ts`, `webapp/components/landing/lp1/{HomeLanding,DemoTabs,CandleSvg}.tsx`,
`{data,chart}.ts`, `lp1.module.css`, `__tests__/home.test.tsx`, `webapp/tests/e2e/lp1-accueil.spec.ts`,
`docs/design/reference-accueil.html`, ce rapport.
Modifiés : `webapp/app/[locale]/(site)/page.tsx`, `webapp/components/Footer.tsx` (ancres),
`webapp/messages/*.json` (namespace `home`), 3 specs e2e skippées.
