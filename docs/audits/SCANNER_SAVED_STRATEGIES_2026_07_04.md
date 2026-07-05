# Scanner — Stratégies nommées, sauvegardées et rechargeables (client-only V1)

**Date** : 2026-07-04
**Branche** : `feat/scanner-saved-strategies` (worktree dédié, depuis `origin/main` = `1526049`)
**Statut** : livré — tsc 0, build vert, 416 tests verts (45 fichiers). Merge sur main **après confirmation live uniquement**.

---

## 1. Contexte et décision de démarrage

La mission demandait d'attendre le merge de `feat/scanner-per-tf-conditions`. Constat au
2026-07-04 : cette branche **n'existe ni en local ni sur origin**, aucun commit de main n'y
fait référence, et le schéma actuel (`webapp/lib/conditions/types.ts` → `ScanCondition`)
ne porte **aucun champ timeframe**. Sur instruction explicite du fondateur (« continue,
fais en sorte que ça fonctionne »), la fonctionnalité a été construite sur le **schéma
actuel**, avec un versionnage conçu pour absorber honnêtement l'arrivée du schéma per-TF :
toute stratégie sauvegardée aujourd'hui qui deviendrait hors schéma demain sera **marquée
invalide avec le détail**, jamais réinterprétée (cf. §4).

## 2. Diagnostic (état des lieux)

- **État de la combinaison active** : `useConditionsConfig` dans
  `webapp/lib/conditions/config-store.ts` (clé `mia.conditionsConfig.v1`, localStorage,
  SSR-safe, sync inter-onglets). C'est LE store client existant du scanner — les
  stratégies nommées s'y adossent (même pattern, même famille de clés), pas de second
  mécanisme.
- **Persistance du chat** : `feat/chat-persistence-client` est PRÊTE mais **non mergée**
  (30176a9). Son store (`lib/chat/thread-store.ts`) n'est donc pas importable ; le pattern
  (sanitisation défensive, caps, dégradation sur quota) a été repris à l'identique.
- **Chemin du scan** : les conditions ne partent au backend qu'au scan, via
  `fetchConditionsScan()` → `POST /api/conditions-scan` (422 = Literals invalides côté
  serveur). Ce chemin est **inchangé**.

## 3. Ce qui a été construit

| Fichier | Rôle |
|---|---|
| `webapp/lib/conditions/strategy-store.ts` | **Nouveau.** Store client-only : `SavedStrategy` (id, nom, `schema_version`, config, createdAt, lastUsedAt), clé `mia.scannerStrategies.v1`, hook `useSavedStrategies` (save/rename/duplicate/delete/markUsed), `validateStrategy()` |
| `webapp/components/scanner/StrategyPanel.tsx` | **Nouveau.** Liste « Mes stratégies » (dernière utilisée en tête) : Charger / Renommer (inline) / Dupliquer / Supprimer (confirmation en 2 temps). Stratégie invalide = badge + raisons détaillées + Charger désactivé |
| `webapp/components/scanner/ConditionsBuilder.tsx` | Formulaire « Sauvegarder la stratégie » (nom libre ≤ 60 car.) sous la logique ET/OU ; nom pré-rempli quand la palette a été repeuplée depuis une stratégie |
| `webapp/components/scanner/ScannerWorkspace.tsx` | Câblage : Charger = repeupler la palette du builder (via `key`) puis le « Enregistrer & relancer » **existant** relance le scan ; panneau visible en mode builder et en mode résultats |

**Sérialisation** : tableau JSON de `SavedStrategy` sous `mia.scannerStrategies.v1`. La
`config` stockée est **exactement** la forme wire (`{logic, conditions}`) — c'est elle,
et elle seule, qui est POSTée au scan.

**UX de rechargement** (conforme mission) : « Charger » ne lance PAS le scan directement.
Il repeuple la palette avec les conditions de la stratégie ; l'utilisateur passe ensuite
par le bouton « Enregistrer & relancer » existant. Un seul chemin de scan, aucun doublon.

**Upsert par nom** : re-sauvegarder sous un nom existant (insensible à la casse) met à
jour cette stratégie en place — charger « London sweep M15 », ajuster, re-sauvegarder =
modification, pas doublon. « Dupliquer » crée « … (copie) » avec un id neuf et une config
copiée en profondeur.

## 4. Versionnage et honnêteté au rechargement

- Chaque stratégie porte `schema_version` (actuel : **1**).
- Au rechargement, `validateStrategy()` revalide TOUT contre les Literals actuels
  (types de conditions de la palette, enums direction/tendance/phase/volatilité,
  `max_bars` entier 1–50, logique AND/OR, version). Messages précis en français :
  « Condition non reconnue : “per_tf_trend_is” », « Champ non reconnu sur “trend_is” :
  “timeframe” », « Version de stratégie 2 non prise en charge », etc.
- Une stratégie invalide **reste visible** (jamais supprimée silencieusement), affiche
  ses raisons, et ne peut pas être chargée → **aucune exécution partielle silencieuse**.
  La validation serveur 422 reste le garde-fou final, inchangé.
- La sanitisation de lecture est volontairement **laxiste sur la config** (elle conserve
  les conditions hors schéma pour que la validation puisse les montrer) et stricte sur la
  structure (entrée sans nom exploitable = ignorée : rien d'affichable).

## 5. Plafond localStorage et politique de purge

- **Caps** : 20 stratégies max, nom ≤ 60 caractères, payload sérialisé ≤ 120 000 car.
  (20 stratégies complètes ≈ 12 Ko réels — marge ×10 sous le quota navigateur de ~5 Mo).
- **Purge : AUCUNE purge silencieuse.** Contrairement aux fils de chat (artefacts
  roulants), une stratégie nommée est un artefact utilisateur : la 21ᵉ sauvegarde est
  **refusée avec un message honnête** (« Limite atteinte — supprime une stratégie
  existante »), et un échec d'écriture (quota, navigation privée) est signalé
  (« Rien n'a été sauvegardé »), jamais présenté comme un succès.

## 6. Frontière légale (Loi 25) — vérifiée

- **Tout côté client** : store React + localStorage uniquement. Diff complet =
  4 fichiers webapp nouveaux + 2 composants webapp modifiés. **Zéro fichier sous `src/`**,
  aucune table, aucun endpoint, aucune écriture DB, aucun cookie.
- Test dédié : espion sur `fetch` pendant toutes les opérations du store → **0 appel réseau**.
- **Nom libre jamais évalué comme condition** : le nom vit hors de `config` ; test dédié
  avec un nom = `price_in_ob` → absent du payload sérialisé du scan, conditions intactes.

## 7. Tests (nouveaux : 21)

- `lib/conditions/__tests__/strategy-store.test.ts` (15) : fidélité sauvegarde/rechargement
  (round-trip byte-for-byte), upsert par nom, tri dernier-utilisé + `markUsed`, rename/
  duplicate (copie indépendante)/delete, cap 21ᵉ refusée, quota → `storage_failed` honnête,
  validation (type inconnu, champ inconnu type per-TF futur, enums inconnus, `max_bars`
  hors bornes, version non supportée, logique inconnue, liste vide), JSON corrompu → `[]`
  sans throw, stratégie hors schéma conservée pour affichage honnête, zéro fetch,
  nom-qui-ressemble-à-une-condition jamais interprété.
- `components/scanner/__tests__/StrategyPanel.test.tsx` (6) : rendu vide, chargement d'une
  stratégie valide, stratégie invalide = badge + raisons + Charger désactivé, suppression
  en 2 temps, renommage inline, erreur honnête sur cap atteint.

**Résultats** : `tsc --noEmit` = 0 erreur · `vitest run` = **416/416 verts (45 fichiers)** ·
`next build` = vert (`/[locale]/scanner` 15.6 kB).

## 8. Notes d'installation (worktree)

`npm ci` échoue sur un conflit de peer deps **préexistant** (vite 8.0.14 exige
`@types/node ≥ 22.12`, le lockfile fige 22.9.0) → installation via
`npm ci --legacy-peer-deps`, lockfile non modifié. À reconcilier un jour dans une mission
outillage (bump `@types/node`), hors périmètre ici.

## 9. Reste à faire (hors périmètre V1)

- Quand `feat/scanner-per-tf-conditions` arrivera : bump `CURRENT_STRATEGY_SCHEMA_VERSION`
  → 2 + migration explicite OU invalidation honnête des stratégies v1 (le mécanisme est
  déjà en place, c'est un choix produit à ce moment-là).
- Persistance serveur des stratégies : attend la politique de confidentialité (prompt #4).
