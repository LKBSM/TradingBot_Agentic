# Lot D1-b — Persistance BOS (option 1a) + cadrage éditorial + header produit /app

- **Date** : 2026-06-10
- **Branche** : `fix/app-live-evaluation-feedback`
- **Décision founder** : Option **1a uniquement** + cadrage éditorial (élément de l'option 2). **1b NON fait** (persistance OB/FVG « tant qu'actives » — déférée Groupe B). D2 inchangé.
- **Discipline tenue** : assembler + frontend uniquement · **AUCUN seuil/heuristique de détection touché** · contrat descriptif (on ne surface QUE l'actif/non-invalidé) · aucun champ prédictif · pas de `git add -A` · pas de PR.

| Sous-lot | Commit |
|----------|--------|
| 1a — persistance BOS (mapper + glue) | `6507ec3` |
| Cadrage éditorial section Structure | `b15ea9a` |
| Header produit /app | `8f84ce3` |

Vérifs : **pytest** mappers/assembler/endpoint/retest **57 verts** (+ SMC/scanner 66) · **vitest 146 verts** (22 fichiers, 0 régression) · **tsc** OK · **next build** vert (`/app` 159 kB).

---

## 1a — Persistance BOS pilotée par l'invalidation que le moteur produit déjà

### Principe & mécanique

Le moteur (`strategy_features._calculate_bos_retest_*`) maintient déjà, sur **chaque bougie**, une machine d'état du retest exposée en colonne `BOS_RETEST_STATE` :

| Valeur | Sens | Sortie vers 0 |
|--------|------|----------------|
| `0` | aucune cassure active | — |
| `±1` | **awaiting** (cassure faite, retest pas encore touché) | invalidation (clôture au-delà du niveau) · timeout (`awaiting_timeout=20`) |
| `±2` | **armed** (prix revenu au niveau, retest en cours) | invalidation · expiration (`armed_window=5`) |

→ `BOS_RETEST_STATE != 0` = fenêtre **bornée et auto-nettoyante** pendant laquelle le moteur considère la cassure **active et non-invalidée**. C'est exactement le cycle de vie demandé.

### Implémentation (mapper, niveau assembler)

`src/intelligence/market_reading_mappers.py` — `confluence_signal_to_structure` surface désormais une cassure quand :

```
fresh_break    = BOS_EVENT != 0                                  (cassure fraîche au bar)
persisted_break = BOS_RETEST_STATE != 0  ET  BOS_SIGNAL non inversé  (cassure antérieure encore active)
```

- **« BOS_SIGNAL n'est pas inversé »** : on exige que la tendance propagée ne contredise pas la direction de l'état de retest (`bos_direction == state_direction`).
- **Disparition** : dès que `BOS_RETEST_STATE` retombe à `0` (invalidation / reprise / timeout), la cassure n'est plus surfacée.
- **`broken_at` honnête** : pour une cassure persistée, l'horodatage d'origine vient du champ glue `BOS_BREAK_TS` (sinon repli sur le bar courant), jamais inventé.

**Glue (pas de logique moteur)** : `realized_levels` expose `BOS_BREAK_TS` = horodatage de la dernière bougie `BOS_EVENT != 0`, forward-carried (gardé sur `DatetimeIndex` seulement, pour ne jamais produire de timestamp bidon sur les frames de test à index entier).

**Aucun seuil touché** : `retest_tol_atr`, `invalid_tol_atr`, `awaiting_timeout`, `armed_window` restent intégralement dans `strategy_features.py`. Le mapper ne fait que **lire** l'état produit.

### Contraste avec le bug F6

F6 émettait un BOS sur **chaque bar à `BOS_SIGNAL` propagé**, sans porte d'invalidation → BOS (souvent périmé) sur ~100 % des lectures. Ici la porte est `BOS_RETEST_STATE`, **bornée** (≤ ~25 bougies) et **auto-nettoyée** par invalidation.

### Mesure sur données réelles (XAU M15, cache, 202 bougies)

| Métrique | Valeur |
|----------|--------|
| Cassures fraîches (`BOS_EVENT != 0`) | 8 bars (4,5 %) |
| Fenêtre active (`BOS_RETEST_STATE != 0`) | 106 bars (**59,9 %**) |
| Surfacé par 1a (fresh OU persisted-active) | 106 bars (59,9 %) |
| Dernière bougie | `state=-2` (retest baissier armé) → cassure baissière **désormais surfacée** + retest |

> **Transparence** : ~60 % des bougies XAU M15 sont en fenêtre de retest active. Ce n'est **pas** du « stale » au sens F6 — c'est la notion d'« active/non-invalidé » du moteur lui-même (le marché passe réellement beaucoup de temps en post-cassure/retest sur cette TF). La dernière lecture réelle (id=12, `bos=null` + retest armé baissier) montre désormais la cassure + le retest, **résolvant le « aucune cassure » à la source**. Si le founder juge 60 % trop large, on pourra restreindre à l'état **armed seul** (`±2`) — c'est un réglage, pas une refonte ; non fait ici car la décision visait `BOS_RETEST_STATE != 0`.

### Tests (pytest)

- `test_persisted_bos_during_active_retest_state` — armé → bos persisté (level forward-fill, broken_at = BOS_BREAK_TS) **+** retest présent.
- `test_persisted_bos_awaiting_state_without_armed_retest` — awaiting → bos persisté, **pas** de retest.
- `test_persisted_bos_broken_at_falls_back_to_bar_ts_without_break_ts` — repli broken_at.
- `test_bos_disappears_on_invalidation` — `state=0` → bos disparaît.
- `test_bos_not_persisted_when_trend_inverted_against_break` — BOS_SIGNAL inversé → non persisté.
- `test_realized_levels_emits_break_ts_from_last_event` — glue BOS_BREAK_TS (DatetimeIndex only).
- Inchangés et toujours verts : `test_propagated_bos_without_event_is_not_shown`, `test_confluence_signal_to_structure_no_signal_but_bos_in_features` (propagé sans état de retest → toujours non surfacé).

---

## Cadrage éditorial (section Structure)

`StructureSection.tsx` — ligne descriptive niveau 1.5 ajoutée en tête de section :

> « Structures décrites au présent : affichées tant qu'elles restent vérifiables, retirées dès leur reprise ou invalidation. Les zones Order Block et Fair Value Gap sont indiquées à leur formation. »

Factuel, aucun mot directif, aucune excuse alambiquée. Précise honnêtement la différence de traitement : cassures persistées tant que vérifiables (1a) vs OB/FVG indiqués à leur formation (1b non fait). Test i18n du libellé.

---

## Header produit /app (retrait de la nav marketing)

`Nav.tsx` devient **route-aware** (`usePathname`, gestion défensive d'un préfixe locale) :

- **Landing** (`/`) : nav marketing inchangée (Démo · Honnêteté · Tarifs · FAQ).
- **/app** : bascule sur `AppHeader` — **marque à gauche** + **cluster utilitaire à droite uniquement** :
  - badge de plan (« Accès libre »), aide (→ `/methodology`), langue (FR/EN via `LocaleToggle`), thème, **menu compte** (avatar + chevron).
- **Menu compte** (`AccountMenu`, dropdown léger sans nouvelle dépendance — fermeture clic-extérieur + Échap) : **Abonnement** (`/#tarifs`), **Langue**, liens **Le site → Honnêteté** (`/#honnetete`) / **FAQ** (`/#faq`), **Se déconnecter** (`/`).
- La **navigation entre marchés** reste dans la colonne de gauche (`InstrumentSidebar`), inchangée.

> Note posture : pas d'auth réelle en mode test (TESTING_MODE). Le badge de plan, « Abonnement » et « Se déconnecter » sont la **coquille produit** ; « Se déconnecter » renvoie à la landing publique. À brancher sur l'auth réelle quand elle existera.

Tests `Nav.test.tsx` : landing montre la nav marketing ; /app la masque (aucun « Démo »/« Tarifs » dans le header) et expose marque + menu compte ; le menu compte porte abonnement/Honnêteté/FAQ/déconnexion ; résolution de `/app` même avec préfixe locale.

---

## Mise à jour — séparation niveau de cassure / flag retest (commit `bd7ef91`)

Décision founder : **ne pas** restreindre la cassure à armed ; séparer plutôt les deux flags, tous deux lus sur le **même** signal canonique `BOS_RETEST_STATE` :

| Flag | Gate | Fenêtre |
|------|------|---------|
| **Niveau de cassure (BOS)** | `BOS_RETEST_STATE != 0` | awaiting **+** armed (inchangé) |
| **Retest en cours** | `abs(BOS_RETEST_STATE) == 2` | **armed seul** (jamais awaiting) |

- Le gating du retest bascule de `BOS_RETEST_ARMED` vers `abs(BOS_RETEST_STATE) == 2` — équivalent côté moteur, mais **explicite** et aligné sur le même signal que la persistance BOS.
- **Cohérence** : le retest n'est surfacé que si la cassure l'est aussi (`fresh_break or persisted_break`) → jamais de retest d'une cassure lâchée (trend inversé).
- Toujours niveau assembler-mapper, **aucun seuil touché**.

**Split mesuré (XAU M15 réel, 177 bougies)** : BOS persiste **59,9 %** · dont awaiting **4,5 %** (retest masqué) · armed **55,4 %** (retest affiché). La séparation déplace les ~4,5 % de bougies « awaiting » de *retest affiché* à *retest masqué* ; le niveau de cassure, lui, reste affiché sur toute la fenêtre active.

Tests ajoutés/ajustés : `test_retest_flag_only_during_armed_state_not_awaiting`, `test_retest_not_shown_when_break_dropped_by_inverted_trend`, et mise en scénario armed réaliste de `test_confluence_signal_to_structure_with_long_signal` + `_structure_rich`. Backend : **164 tests verts** sur les suites concernées, 0 régression.

## Déféré (NON fait ici)

- **1b** — persistance OB/FVG « tant qu'actives » : nécessiterait une logique d'invalidation OB/FVG inexistante côté moteur (zone consommée / FVG comblée), dont les définitions doivent d'abord être validées par l'**annotation manuelle**. **Groupe B.**
- **D2** — fenêtre news inchangée (« aucun événement » correct).
- Réglage possible (non fait) : restreindre la persistance BOS à l'état **armed** (`±2`) si ~60 % est jugé trop large.

## Commits

```
6507ec3  feat(app/D1-b): persistance BOS au mapper (option 1a) — actif tant que non invalidé
b15ea9a  feat(app): cadrage éditorial section Structure (présent, tant que vérifiable)
8f84ce3  feat(app): header produit sur /app (retrait nav marketing)
```
