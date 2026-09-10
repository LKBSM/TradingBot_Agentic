# AUDIT — MKT-1 : Frontend à l'échelle de 100 marchés (test UX)

**Branche** : `feat/mkt-1-frontend-100` · **Base** : `origin/main` = `be27380` (PR #207)
**Worktree** : `C:\MyPythonProjects\wt-mkt-1-front`
**Date** : 2026-09-10

> ## ⚠️ LE DRAPEAU RESTE DÉSACTIVÉ APRÈS LE MERGE
>
> Cette fonctionnalité est **entièrement inerte** une fois mergée. Aucun visiteur,
> aucun early-access user, aucun déploiement de production ne verra le catalogue
> de test tant que personne ne positionne explicitement la variable
> d'environnement sur son propre environnement.
>
> **Pour l'activer sur votre machine** (et nulle part ailleurs) :
> ```bash
> # webapp/.env.local
> NEXT_PUBLIC_SHOW_MARKET_CATALOG_UX_TEST=1
> ```
> puis redémarrer le serveur (`npm run dev`, ou `npm run build && npm start`) —
> la valeur est lue à la compilation, il faut donc reconstruire.
>
> **Pour la désactiver** : retirer la ligne (ou la mettre à `0`) et rebuilder.
> Tant qu'elle est active, un bandeau **« Mode test UX »** permanent et non
> masquable s'affiche dans la colonne des marchés, pour que ça ne s'oublie pas.

---

## 1. Ce que la mission demandait, et ce qui a été livré

| Demande | État |
|---|---|
| Catalogue de 100 marchés, séparé de la config réelle | ✅ `config/market_catalog_ux_test.json` |
| Drapeau OFF par défaut, production comprise | ✅ `NEXT_PUBLIC_SHOW_MARKET_CATALOG_UX_TEST` |
| Recherche / liste latérale à l'échelle de 100 | ✅ groupement par catégorie, recherche élargie, contrôles collants |
| État vide honnête, aucune donnée inventée | ✅ 7ᵉ cas de l'état d'absence existant + refus **avant** toute requête |
| Aucune modification de la config réelle | ✅ `config/markets.json` et ses 8 consommateurs backend intacts |

**Le périmètre produit n'a pas bougé** : le moteur suit toujours XAUUSD et EURUSD, et rien d'autre.

---

## 2. Le point de départ : ce qui existait déjà

Le diagnostic (lecture seule, avant tout code) a établi trois choses qui ont façonné l'implémentation :

**La fondation MKT-1 précédente est propre et il ne fallait pas la polluer.** `config/markets.json` est la source unique : le backend la lit via `market_registry.py` → `supported_instruments()` → 8 routes ; le frontend consomme un module **généré** (`gen_markets.mjs` → `markets.generated.ts`). Ajouter 100 marchés à ce fichier les aurait rendus « supportés » partout automatiquement — précisément ce que la mission interdit. D'où une **chaîne jumelle et strictement séparée**.

**L'état vide existait déjà, et il était meilleur que ce que j'aurais construit.** `ReadingPlaceholders.tsx` distingue **six** modes d'échec, avec un principe écrit dans le code :

> *« Distinct honest copy per failure mode (PERF-1): the user must be able to tell "trop lent", "serveur injoignable", "aucune donnée" and "combo non supporté" apart — never one vague "données indisponibles" for all of them. »*

Il a donc été **étendu**, pas remplacé. Le cas manquant était « marché que le moteur ne suit pas » : sans ajout, un marché du catalogue serait tombé sur le 400 « cette combinaison n'est pas prise en charge », qui décrit un fait différent (une combinaison invalide, qu'un réessai pourrait résoudre).

**Le sélecteur ne tenait pas à 100 entrées — trois ruptures**, dont une grave : `.rail { overflow-y: auto }` fait scroller le rail entier, donc à 100 lignes le champ de recherche sortait par le haut et les unités de temps se retrouvaient 100 lignes plus bas. Les deux contrôles qui rendent une longue liste utilisable étaient les premiers à quitter l'écran.

---

## 3. Le catalogue de test

`config/market_catalog_ux_test.json` — **100 marchés**, générés en module TS par `scripts/gen_market_catalog.mjs` (jumeau de `gen_markets.mjs`, avec `--check`).

| Catégorie | Nombre |
|---|---|
| Devises majeures | 7 |
| Devises mineures | 21 |
| Devises exotiques | 20 |
| Métaux | 8 |
| Indices | 20 |
| Crypto | 24 |
| **Total** | **100** |

Trois décisions de conception :

- **Champ `group`, pas `type`.** Le registre réel type les marchés sur `metal | fx | crypto | index` (validé côté Python). Les six catégories d'affichage vivent dans le catalogue de test seul — le type du moteur n'a pas été élargi pour un besoin d'affichage.
- **Ni `priceDecimals`, ni `timeframes`.** Un marché que le moteur ne suit pas n'a pas de prix à formater ni d'unité à servir. Leur absence rend structurellement impossible d'afficher un prix pour lui. Un test le verrouille.
- **XAUUSD et EURUSD figurent dans le catalogue** (ce sont des marchés populaires), mais sont **dédoublonnés à l'affichage** : `CATALOG_ONLY_ENTRIES` = catalogue − registre réel = **98**. Un marché promu dans `markets.json` quitte donc le catalogue de test tout seul, sans seconde édition à ne pas oublier.

### ⚠️ Provenance de la liste — à lire

Le fichier `100-marches-populaires.md` annoncé dans la mission **était introuvable** : absent de la racine, de `docs/`, du `git ls-tree` complet de `origin/main`, et de Downloads/Desktop/Documents. Sur votre GO, **j'ai constitué les 100 lignes moi-même** à partir des conventions usuelles du marché (paires FX majeures/mineures/exotiques, métaux précieux, indices actions de référence, principales cryptomonnaies).

Conséquence à connaître : **ce sont mes noms, mes catégories et mes symboles présumés, pas les vôtres, et aucun n'est validé par un fournisseur de données.** Ni la couverture ni les symboles ne sont garantis. C'est sans risque produit — aucune donnée n'est demandée avec ces symboles — mais si votre liste de référence réapparaît, elle doit remplacer celle-ci. Cette réserve est aussi inscrite dans le champ `_provenance` du JSON.

---

## 4. Le drapeau

`NEXT_PUBLIC_SHOW_MARKET_CATALOG_UX_TEST`, lu à **un seul endroit** (`webapp/lib/market-catalog.ts`) :

```ts
export const CATALOG_UX_TEST_ENABLED =
  process.env.NEXT_PUBLIC_SHOW_MARKET_CATALOG_UX_TEST === '1' ||
  process.env.NEXT_PUBLIC_SHOW_MARKET_CATALOG_UX_TEST === 'true';
```

Le préfixe `NEXT_PUBLIC_` n'est pas un choix de style : Next.js n'expose aucune autre variable au navigateur, et la liste de marchés est un composant client. La règle de véracité (`'1'` ou `'true'`) suit la convention déjà en place dans `middleware.ts`. Un test vérifie que `'0'`, `'false'`, `'yes'`, `'on'`, `'TRUE'`, `' 1'` et `''` laissent la porte **fermée** — pas de véracité accidentelle.

Le motif `NODE_ENV === 'production' → notFound()` (utilisé par la galerie DS-1) a été **écarté délibérément** : il aurait rendu la fonctionnalité invisible sur un preview de production, c'est-à-dire exactement là où vous voulez la voir.

### ⚠️ Ce que le drapeau ne fait PAS — mesuré sur un vrai build

J'avais d'abord écrit dans ce rapport que le catalogue était **absent** du bundle
d'un build standard. **C'était faux, et la vérification l'a montré.** Next.js
n'inline une variable `NEXT_PUBLIC_*` que si elle **existe** au moment du build ;
absente, l'expression reste dynamique, il n'y a donc aucune élimination de code
mort, et la table des 100 marchés est bien expédiée au navigateur.

Mesure : le chunk qui la contient pèse **12,3 Ko** (dont ~8 Ko pour le catalogue,
soit ~2-3 Ko une fois compressé). Ce sont des **noms publics de marchés** — aucune
donnée sensible, aucun secret, aucune information sur le moteur.

**Ce que le drapeau garantit est donc un comportement, pas un poids de bundle :**
avec le drapeau off, rien ne s'affiche, aucune structure dérivée n'est peuplée,
aucune requête n'est émise. C'est ce que la mission demandait, et c'est vérifié
par 40 tests. Le script qui prétendait prouver l'absence du bundle a été
**supprimé** plutôt que conservé en promettant une garantie intenable.

Si vous voulez réellement l'éliminer du bundle, la voie est de définir
explicitement `NEXT_PUBLIC_SHOW_MARKET_CATALOG_UX_TEST=0` au build (une variable
définie *est* inlinée, ce qui rend l'élimination possible) — non fait ici, car
cela ferait dépendre une propriété du produit d'une configuration de déploiement.

**L'indicateur.** Tant que le mode est actif, un bandeau permanent, non masquable, s'affiche en tête de la colonne des marchés (teinte `--sentinel-warn`, définie sur les 4 thèmes) :

> **Mode test UX** — 98 marchés listés pour l'affichage seulement. Le moteur n'en suit que 2.

Les deux comptes sont **calculés**, jamais écrits en dur : ils resteront justes si le catalogue change.

---

## 5. L'interface à 100 entrées

**Séparation visible test / réel.** Quand le mode est actif, la section des marchés réels s'intitule **« Suivis par le moteur »**, suivie des six catégories du catalogue. La distinction n'est pas seulement dans le JSON, elle est à l'écran.

> *Note — question laissée ouverte au GO.* Vous n'aviez pas tranché explicitement ce point (question 4 du diagnostic). J'ai appliqué ma recommandation : séparation visuelle. Si vous préférez une liste unique où la distinction n'apparaît qu'au clic, c'est un changement d'une dizaine de lignes.

**Catégories repliables, fermées par défaut.** 98 marchés dépliés d'un coup forment un mur. Le dépôt avait déjà tranché cette même question dans le même sens pour les familles de conditions du scanner (commit `c6266e5`). Chaque en-tête indique son nombre d'entrées. **Une recherche ouvre automatiquement** les catégories qui ont un résultat — sinon la recherche semblerait ne rien trouver.

**Recherche élargie** à `id + libellé + symbole + nom de catégorie` : à 100 entrées, « crypto » ou « indices » est la façon dont on cherche réellement. Filtres mémoïsés (`useMemo`).

**Contrôles collants** (`position: sticky`) : le champ de recherche en haut, les unités de temps en bas. C'est le correctif de la rupture n°1. **Aucun effet visible à 2 marchés** — rien ne déborde.

**Pas de virtualisation** — et c'est mesuré, pas affirmé. Les catégories étant fermées par défaut, le mur de 98 lignes n'est jamais rendu : un test vérifie que le DOM est proportionnel à ce qui est **ouvert**, pas à la taille du catalogue. Virtualiser aurait cassé le Ctrl+F du navigateur et la navigation clavier pour un gain nul.

**Aucune épingle sur un marché du catalogue.** Épingler un marché sans donnée serait une promesse que le produit ne peut pas tenir. La ligne reste sélectionnable — c'est tout l'objet du test — elle n'offre simplement rien qui suggère une couverture.

---

## 6. « Aucune donnée inventée » — comment c'est garanti

La garantie n'est pas « on n'affiche pas la réponse », c'est **« il n'y a pas de réponse »**.

Les quatre hooks de données (`useMarketReading`, `useCandles`, `useLatestPrice`, `useMtfTrends`) refusent un marché du catalogue **avant tout appel réseau** :

```ts
if (isCatalogOnly(instrument)) {
  setData(null);
  setError(new MarketNotCoveredError(instrument));
  return;                     // ← aucune requête n'est émise
}
```

Aucune requête ne part, donc rien ne peut être rendu à partir d'une réponse, et la garantie devient **structurelle** plutôt que déclarative. Bénéfice secondaire : plus de 4 appels 400 en cascade à chaque clic. Les tests vérifient que `fetch` n'est **jamais** appelé — et qu'un marché **réel** l'est toujours normalement, pour que la garde reste étroite.

### L'état vide — texte exact (Option A, validée)

> ### Pas encore disponible sur ce marché
> Le moteur ne suit pas encore **Bitcoin (BTC/USD)**. Aucune lecture, aucun graphique et aucun prix n'existe pour lui — rien n'est simulé en attendant.

Variante graphique : *« Aucune bougie n'existe pour {marché} : le moteur ne suit pas ce marché. Rien n'est dessiné à la place. »*

- **Aucun bouton « Réessayer »** : réessayer ne changerait pas la couverture ; le proposer serait malhonnête.
- Le marché est nommé par son **libellé humain**, pas un ticker brut.
- C'est un **7ᵉ cas du composant existant**, pas un nouveau composant — il ne contredit donc rien de ce qui est déjà en place.
- Traduit dans les **9 locales**.
- Un test vérifie que le rendu ne contient **aucun chiffre** (hors le nom du marché lui-même) : ni prix, ni niveau, ni score.

---

## 7. Vérifications

| Vérification | Résultat |
|---|---|
| `tsc --noEmit` | ✅ **0 erreur** |
| `npm run build` (production, sans le drapeau) | ✅ **exit 0** |
| `node scripts/gen_market_catalog.mjs --check` | ✅ à jour |
| Tests MKT-1 (4 fichiers) | ✅ **40/40** |
| `MarketSelector.test.tsx` (existant) | ✅ 11/11 |
| `markets-guard.test.ts` (garde MKT-1) | ✅ 3/3 |
| `reading-load-honesty.test.tsx` (états d'absence) | ✅ 5/5 |

**Les 40 tests MKT-1 :**

- `market-catalog-flag.test.ts` (10) — drapeau off par défaut ; seuls `1`/`true` l'ouvrent ; aucun marché du catalogue dans `SUPPORTED_INSTRUMENTS` ni `ALL_MARKET_IDS` ; registre réel toujours exactement `[EURUSD, XAUUSD]` ; dédoublonnage des marchés présents dans les deux fichiers ; absence de `priceDecimals`/`timeframes` ; module généré en phase avec le JSON ; 100 entrées, ids uniques, 6 groupes non vides.
- `mkt1-catalog-ux.test.tsx` (12) — bandeau et ratio réel ; séparation « Suivis par le moteur » ; 6 catégories fermées au chargement ; compteur par catégorie ; recherche par id, par libellé et **par catégorie** ; message d'absence explicite ; **performance** (montage + frappe bornés) ; DOM proportionnel à ce qui est ouvert ; pas d'épingle sur un marché sans donnée ; sélection émettant le bon combo.
- `mkt1-no-invented-data.test.tsx` (11) — les 4 hooks n'émettent **aucune requête** pour un marché du catalogue ; un marché **réel** est toujours requêté normalement ; l'état vide nomme le marché, n'offre aucun « Réessayer », n'est pas confondu avec la copie « combinaison non prise en charge », et **ne contient aucun chiffre**.
- `mkt1-catalog-off.test.tsx` (7) — drapeau off : aucun bandeau, aucune catégorie, en-tête « Marchés » d'origine, aucun marché du catalogue listé ni trouvable ; un lien profond vers un marché du catalogue suit le chemin ordinaire (400 backend), sans qu'aucun état MKT-1 ne fuite.

### Ce qui n'a PAS été vérifié

- **La suite de tests complète n'a pas pu être exécutée** : la machine a saturé sa mémoire (l'exécution a été tuée par le système, deux fois). Les tests directement impactés ont été lancés et passent, mais je ne peux pas affirmer « 0 régression » sur l'ensemble du dépôt.
- **Playwright n'a pas été exécuté** et **aucune capture n'a été produite**, pour la même raison. La spec `mkt1-catalog.spec.ts` est livrée et prête (2 viewports × 4 scénarios), avec le mode d'emploi en tête de fichier.

---

## 8. Fichiers

**Ajoutés**
```
config/market_catalog_ux_test.json          100 marchés, catalogue d'affichage
scripts/gen_market_catalog.mjs              générateur jumeau (+ --check)
webapp/lib/market-catalog.generated.ts      module généré (ne pas éditer)
webapp/lib/market-catalog.ts                drapeau + dérivation catalogue/réel
webapp/lib/__tests__/market-catalog-flag.test.ts
webapp/components/market/__tests__/mkt1-catalog-ux.test.tsx
webapp/components/market/__tests__/mkt1-catalog-off.test.tsx
webapp/components/app/__tests__/mkt1-no-invented-data.test.tsx
webapp/tests/e2e/mkt1-catalog.spec.ts
```

**Modifiés**
```
webapp/components/market/MarketSelector.tsx   groupement, recherche élargie, bandeau
webapp/components/shell/shell.css             sticky + styles catalogue
webapp/components/app/ReadingPlaceholders.tsx 7ᵉ cas d'absence
webapp/components/app/DesktopReading.tsx      câblage du placeholder graphique
webapp/components/app/ReadingColumn.tsx       idem
webapp/lib/market-reading/api-client.ts       MarketNotCoveredError
webapp/lib/market-reading/hooks.ts            court-circuit dans 4 hooks
webapp/lib/__tests__/markets-guard.test.ts    liste blanche du garde MKT-1
webapp/messages/*.json  (9)                   12 clés × 9 locales
webapp/.env.example                           documentation du drapeau
```

**Intouchés, par construction** — la configuration réelle des marchés :
```
config/markets.json                 src/intelligence/market_registry.py
src/intelligence/lookback_config.py webapp/lib/markets.generated.ts
webapp/lib/markets.ts               webapp/lib/market-reading/perimeter.ts
+ les 8 consommateurs backend de supported_instruments()
```

---

## 9. Ce qui reste ouvert

1. **La liste des 100 marchés est la mienne** (§3). Si votre fichier de référence réapparaît, il doit la remplacer — c'est une édition du JSON + `node scripts/gen_market_catalog.mjs`, rien d'autre.
2. **Séparation visuelle test/réel** (§5) : appliquée sur ma recommandation, pas sur votre arbitrage explicite.
3. **Ce test ne dit rien du backend.** Il mesure la tenue de l'**interface** à 100 entrées. Servir réellement 100 marchés (quotas fournisseur, stockage des bougies, charge de détection, calendrier macro) est une question entièrement distincte, non abordée ici.
