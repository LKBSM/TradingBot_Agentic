# Articles — état : DORMANTS

**Statut au Chantier 5.C (2026-06-07) : non routés, invisibles utilisateur.**

Les 10 fichiers Markdown de ce dossier (`01-…` à `10-…`) constituent un fonds
éditorial pédagogique rédigé lors des premières vagues. **Aucun n'est
actuellement routé** : il n'existe pas de page `/articles` ni d'import de
`content/articles/` dans `app/`, `components/` ou `lib/`. Ils ne sont donc
**pas visibles** par les utilisateurs et n'apparaissent pas au build.

## Pourquoi on les garde (Option B)

Plutôt que de les supprimer, on les conserve dormants : ils représentent un
travail rédactionnel réutilisable (SMC, COT, régimes, HAR-RV, FOMC, sessions,
CPCV/DSR…). Ils seront **retravaillés ou supprimés en pré-lancement**, une fois
décidée la stratégie de contenu (blog SEO ? base de connaissances in-app ?).

## ⚠️ Dette à traiter avant toute publication

Plusieurs articles contiennent des résidus à corriger **avant** d'être routés :

- **Branding pré-pivot** — 5 fichiers mentionnent encore « Smart Sentinel »
  (ancien nom) au lieu de **MIA Markets** :
  `01-comprendre-smart-money-concepts.md`, `02-comprendre-rapport-cot-or.md`,
  `07-fomc-impact-or-trading-calendrier.md`, `09-cpcv-dsr-backtest-rigoureux.md`,
  `10-pourquoi-aucun-systeme-bat-marche.md`.
- **Positionnement pré-pivot** — à auditer contre la règle niveau 1.5 stricte
  (`edge_claim=false`, pas de promesse de gain) avant publication, comme le
  reste du front en Chantier 5.C.

> Tant que ces articles ne sont pas routés, ils ne créent pas d'exposition
> client. Ne pas créer de route `/articles` sans avoir d'abord purgé le
> branding et le positionnement ci-dessus.

## Revue Chantier 5.D (2026-06-07) — maintien dormant confirmé

Revérifié : toujours **aucune route**, aucun import de `content/articles/` dans
`app/` / `components/` / `lib/` → **zéro exposition client**. Le 5.D n'a donc
**pas** supprimé ces fichiers : ils représentent un travail rédactionnel
réutilisable et leur dette (branding pré-pivot + positionnement) ne crée aucun
risque tant qu'ils restent non routés. Décision cohérente avec l'Option B du 5.C.

La page `/methodology` (créée en 5.D) couvre désormais le besoin pédagogique
**in-app** essentiel (concepts SMC, calcul des éléments affichés, source de
données) en niveau 1.5 strict. Ces articles restent une réserve éditoriale pour
une future stratégie de contenu (blog SEO ou base de connaissances). Leur
purge/réécriture reste à faire **avant** tout routage, comme indiqué ci-dessus.
