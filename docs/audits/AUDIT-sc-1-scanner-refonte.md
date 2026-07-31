# AUDIT — SC-1 Refonte du Scanner (palette + interface)

Branche `feat/sc-1-scanner-refonte` (worktree dédié, depuis `origin/main` f22d260).
Read-only sur la détection : conservé et prouvé par test.

## 1. Tri final des 20 conditions

Base de vérité : `origin/main`. **Découverte majeure** : les 3 avertissements ⚠ de
la mission (TR-1 / TF-1 / LQ-D1 « bloquants ») étaient **périmés** — ces chantiers
sont mergés sur main (`trend` = bullish/bearish/**indeterminate**,
`READING_LOGIC_VERSION` = 4, LQ-D1 corrigé). Vérifié dans le code.

| # | Condition | Statut diagnostic | Livré ? | Type exposé |
|---|-----------|-------------------|---------|-------------|
| 1 | Tendance structurelle (haussière/baissière/indéterminée) | (1) déjà | ✅ | `trend_is` |
| 2 | L'unité supérieure va (même sens / sens opposé) | (2) calculable | ✅ | `higher_tf_agrees` |
| 3 | Dernier événement (BOS↑/↓, CHOCH↑/↓) | (2) calculable | ✅ | `last_event_is` |
| 4 | CHOCH dans les N dernières bougies | (1) déjà | ✅ | `choch_recent_confirmed` |
| 5 | BOS dans les N dernières bougies | (1) déjà | ✅ | `bos_recent_confirmed` |
| 6 | Prix dans un Order Block (dir.) | (1) déjà | ✅ | `price_in_ob` |
| 7 | Prix dans un FVG (dir.) | (1) déjà | ✅ | `price_in_fvg` |
| 8 | Zone jamais testée depuis sa formation | (2) calculable | ✅ | `zone_untested` |
| 9 | Zone testée au plus N fois | (3) **bloquée** | ❌ **non exposée** | — (voir §3) |
| 10 | Zone formée dans les N dernières bougies | (2) calculable | ✅ | `zone_formed_recent` |
| 11 | Prix à moins de X % d'une zone, sans y être | (1) déjà (raffiné) | ✅ | `price_near_ob`, `price_near_fvg` |
| 12 | Poche prise (balayée) dans les N dernières bougies | (1) déjà (LQ-D1 OK) | ✅ | `liquidity_swept_recent` |
| 13 | Poche intacte à moins de X % du prix | (1) déjà (LQ-D1 OK) | ✅ | `price_near_liquidity` |
| 14 | Poche cassée dans les N dernières bougies | (2) calculable | ✅ | `liquidity_broken_recent` |
| 15 | Creux/sommets égaux présents | (2) calculable | ✅ | `equal_levels_present` |
| 16 | Phase de marché | (1) déjà | ✅ (sans `distribution`) | `market_phase_is` |
| 17 | Volatilité (étendue/normale/contractée) | (1) déjà | ✅ | `volatility_is` |
| 18 | Prix dans le tiers … du range | (2) calculable | ✅ | `price_in_range_third` |
| 19 | Session en cours | (2) calculable | ✅ | `session_is` |
| 20 | Dernier événement remonte à (<10/10-50/>50) | (2) calculable | ✅ | `last_event_age` (famille Contexte) |
| 21 | Prix dans une zone déjà testée au moins une fois | nouveau (booléen `tested`) | ✅ | `price_in_tested_zone` |

**Décompte** : (1) déjà = 10 · (2) calculable = 9 · (3) bloquée = 1, + #21 nouveau.
**Livrées : 19 des 20 + #21 = 20 conditions exposées.** Seule non livrée : **#9**
(bloquée, cf. §3). `mtf_aligned` **retiré**, remplacé par #2 `higher_tf_agrees`.

Conditions **retirées** (signalées, pas supprimées en silence) :
- `ob_fvg_confluence` — exactement « prix dans OB » ET « prix dans FVG » ; la règle
  produit interdit d'offrir deux fois la même chose sous un troisième nom.
- ancien `retest_in_progress` — c'était un flag de retest de **NIVEAU BOS** (état
  ARMED du state-machine), pas « prix dans une zone testée ». Décision fondateur :
  retirer et le remplacer par le fait `price_in_tested_zone`.
- `mtf_aligned` (3 unités fixes H4/H1/M15) — remplacé par #2 `higher_tf_agrees`
  (comparaison RELATIVE à l'unité immédiatement supérieure ; cf. §2).

**#21 — formulation figée (validée fondateur), fr + en :**
- Libellé : « Le prix est dans une zone déjà testée au moins une fois » /
  « Price is inside a zone tested at least once ».
- Concept 3-temps : *ce que ça compte* (prix à l'intérieur d'un OB/FVG actif
  touché ≥1× depuis sa formation) · *convention SMC* (« testée » = retouchée après
  formation ; opposé « jamais testée ») · *ce que ça ne dit pas* (ni rebond, ni
  rejet, ni poursuite — un fait de position au présent, pas une attente).
- Ne dépend que du booléen `tested` (existant). Aucun mot d'attente/continuation.

## 2. Condition #2 livrée — `higher_tf_agrees` (arbitrage C1)

Décision fondateur (arbitrage C1) : **#2 devient la seule condition d'alignement,
`mtf_aligned` est retiré.** Motif : une comparaison RELATIVE à l'unité
immédiatement supérieure fonctionne identiquement sur les six unités ; un ensemble
fixe casse au sommet (rien au-dessus du D1).

Garanties livrées :
- **Structurel** : TR-1 étant sur main, `regime.trend` est déjà structurel
  (dernier BOS/CHOCH). `derive_structural_trend` est la **seule** source (vérifié :
  `_derive_trend` supprimé). Le caveat « déplacement de clôtures » de C1-a est donc
  **sans objet** et n'a pas été ajouté.
- **Unité nommée** dans le résultat (C1-b) : « Le 1 h va dans le même sens » (via
  `alignment_timeframes`, registre TF-1).
- **Trois cas NON ÉVALUABLES distincts** (C1-b/c), messages différents à l'écran,
  jamais remplis par défaut ni comptés comme échec : (1) pas d'unité au-dessus
  (unité la plus haute suivie) ; (2) unité supérieure sans tendance établie ;
  (3) cette unité sans tendance établie. « indéterminée » retirée des cibles de
  relation (choix mort sous cette règle) → `relation` = même sens / sens opposé.
- **Chiffre C1-c (MESURÉ, données réelles)** : `derive_structural_trend` sur les
  **74 snapshots réels** (XAU/EUR × M15/H1/H4, 500–2880 bougies) + les 6 combos
  courants du `market_readings.db` → **0 % indéterminé**. Avec 500+ bougies il y a
  toujours un BOS/CHOCH récent, donc la tendance structurelle **se tranche presque
  toujours**. Conséquence : `higher_tf_agrees` est **évaluable quasi tout le
  temps** côté tendance ; en pratique le NON-ÉVALUABLE vient surtout de « **pas de
  lecture pour l'unité supérieure** » (ex. H4 quand le D1 n'est pas chaud), un
  sujet de **couverture**, pas de tendance. (Le « ranging » 34 % de l'ancien DB
  est la **phase**, pas la tendance — cohérent §4/§10.B.)

⚠️ À la prochaine itération : trancher `mtf_aligned` (fixe) → `higher_tf_agrees`
(relatif, « unité au-dessus », valeurs même/opposé/indéterminée) et retirer l'un.

## 3. Condition #9 bloquée — chantier

#9 « zone testée au plus N fois » exige un **compteur de touches par OB/FVG**. Le
moteur n'expose qu'un **booléen `tested`**, pas un décompte (seules les
`liquidity_pools` ont un champ `touches`). Préparée dans `BLOCKED_PALETTE`
(documentée, avec la raison) mais **non exposée** ; le Literal de requête l'exclut
→ 422 si demandée. Elle deviendra offrable quand le mapper comptera les
ré-entrées par zone. Effort estimé : moyen (mapper + réémission cache).

## 4. Phases de marché — joignabilité (constat, à corriger ailleurs)

`_derive_market_phase` (unique producteur) peut émettre :
- **expansion** (directionnel + volatilité `elevated`)
- **trend** (directionnel + non-elevated)
- **ranging** (indéterminé + clôtures en oscillation)
- **accumulation** (indéterminé + sinon)

**`distribution` est INATTEIGNABLE** : aucun chemin ne l'émet post-TR-1 (la
tendance ne vaut plus que bull/bear/indéterminée ; la dérivation ne branche que
sur directionnel+vol ou indéterminé+oscillation). Conformément à la décision
fondateur, `distribution` est **retiré de la palette** (offrir une condition qui
ne peut remonter que zéro est pire qu'une condition absente) et **n'est pas exposé
vide**. ⚠️ C'est un **défaut de la détection de phase**, pas une contrainte à
contourner : sujet d'une mission dédiée (rendre `distribution` atteignable, ou
acter à 4 phases). Rien corrigé ici.

## 5. Persistance des lectures enregistrées

- **localStorage** conservé (`mia.scannerStrategies.v1`), derrière l'**interface**
  du store (seam : un futur adaptateur serveur = un adaptateur, pas une refonte).
- **Réévaluation (C4, LIVRÉE)** : compte de combos par lecture **réévalué à
  l'ouverture** (jamais en arrière-plan). **Péremption PAR COMBO sur son pas de
  temps** — réutilise `bars_behind` / `_compute_freshness` : un combo est périmé
  dès qu'une bougie de SON unité a clôturé depuis la lecture (M15 → 15 min,
  D1 → 1 jour ; **pas de seuil global**, qui aurait traité un combo journalier
  comme un M15). Affichage : compte + « évalué il y a X » quand tout est frais ;
  dès qu'un combo est périmé (ou sans lecture) → badge **« Compte incomplet »** +
  nombre de combos en attente + **Relancer**. Un compte partiel n'est **jamais**
  présenté comme complet.
- **Mention obligatoire non masquable** : « le compte est un constat, pas un
  classement ; 7 combos n'est pas meilleur, c'est plus large » + « conservées sur
  cet appareil, non synchronisées ».
- **Export / import texte** ajoutés (l'utilisateur déplace ses lectures lui-même).
- **Condition disparue** : `validateStrategy` signale précisément la/les
  condition(s) hors schéma ; chargement désactivé, jamais modifié en silence.

## 6. Interdits (section 0) — application

- **Aucun score / classement / tri par nombre de conditions** : ordre fixe
  (`SCAN_COMBOS`, marché puis unité) ; test `test_scan_never_sorts_by_match_count`.
- **Aucune condition prédictive / valeur hors palette** : Literal par valeur +
  `extra=forbid` → **422 avant évaluation** (type, valeur, champ, type bloqué).
  Tests : type prédictif, `distribution`, champ inconnu, `zone_tested_at_most`.
- **Aucun assouplissement suggéré** : aucun bouton/lien ne le propose ; état
  « aucun combo » dit explicitement que ce n'est pas une erreur.
- **Zéro condition ≠ tous les marchés** : `min_length=1` (422) + mention builder ;
  test `test_and_logic_not_matched_when_all_non_evaluable`.
- **Vocabulaire interdit** : testé fr + en (palette back + front), mots setup /
  signal / opportunité / meilleur / fort / idéal / qualité / score / rang / top…
- **Non évaluable** : 3ᵉ état, exclu du dénominateur, ni rempli ni non rempli ;
  tests back + front.
- **Bloc « à l'encontre » toujours rendu, non masquable** : test front
  (`against-block`).
- **Read-only** : `test_scan_is_read_only_touches_only_get_latest_reading`
  conservé (writes/détection lèvent).

## 7. Écarts restants avec la maquette (`reference-scanner.html`, la vraie)

La vraie maquette est sur `main`. Réconciliation live à faire avec le fondateur
(verrou de merge). Alignés depuis l'arbitrage : #2 `higher_tf_agrees`, bloc
« à l'encontre » enrichi, filtres d'affichage, compte live (construction) + par
lecture (C4), `last_event_age`→Contexte, phrase récap + Copier, états vides,
mention « ne pas assouplir ». Écarts **cosmétiques** restants (à trancher live) :
- En-têtes de carte résultat : badge symbole + noms amicaux (« Or · 15 min ») +
  valeur constatée alignée à droite (mono) — mon rendu inline le détail.
- Boutons segmentés : la maquette les pose inline à droite de la ligne ; mon
  rendu les met sous la condition avec un libellé de contrôle.
- État « aucune condition » : la maquette a une carte centrée à icône ; mon rendu
  replie la note dans la barre collante du builder.
- **C2 divergence assumée** : la maquette expose #9 « testée au plus N fois » —
  **non livrée** (donnée absente, cf. §3 et §10).
- **C3 divergence assumée** : la maquette montre la phase en binaire
  (expansion/consolidation) ; livré = **4 phases réelles** (décision GO-2).

## 8. Vérifications

- Backend : **62 tests scanner** + 42 adjacents verts.
- Front : `tsc` 0, **`next build` exit 0**, **78 vitest** (palette 4 familles,
  garde-fous résultats + bloc « à l'encontre » full-match, non-régression store,
  **vocabulaire interdit fr+en négation-aware**).
- **Playwright 8/8 verts** (5 états × 2 viewports, 1280×800 + 390×844) contre
  `next dev` (API mockée) : compte live, bloc « à l'encontre » enrichi, compte C4,
  « ne pas assouplir », aucun débordement, aucune clé i18n brute.
- **Validation données réelles (TestClient sur le vrai `market_readings.db`)** :
  scan **200**, `higher_tf_agrees` nomme l'unité (« Le 1 h va dans le même sens »),
  cas NON ÉVALUABLE réel sur H4 (D1 non chaud) avec dénominateur ajusté,
  `context_against` peuplé (1–3 signaux/combo), 4 combos `unavailable` non inventés.
- Reste **live** : Playwright contre le vrai backend `:8000` (uvicorn + provider)
  pour l'e2e navigateur bout-en-bout — la logique est déjà prouvée en TestClient.

## 9. Reste à faire avant merge

1. **Réconciliation live de la maquette** (écarts cosmétiques §7) — avec le
   fondateur, app + maquette côte à côte.
2. (Optionnel) Playwright bout-en-bout contre le vrai backend `:8000` — la logique
   est déjà prouvée en TestClient sur données réelles ; C1-c mesuré (0 %).
3. **Merge sur main SEULEMENT après confirmation live du fondateur.**

## 10. Mission de suivi (constats, RIEN implémenté ici)

### 10.A — Compteur de touches horodatées par zone (débloque #9 ET la page Zones)

- **Ce que le moteur enregistre aujourd'hui** : `_ob_lifecycle` / `_fvg_lifecycle`
  (`market_reading_mappers.py`) parcourent déjà les bougies APRÈS la formation de
  la zone et retiennent la **PREMIÈRE touche** (`first_tap` / `entry_idx` →
  `mitigated_at`) + un **booléen `tested`**. La boucle voit chaque bougie mais
  **ne compte pas** les ré-entrées suivantes ; après la 1re touche le statut passe
  à `mitigated` et le décompte s'arrête.
- **Ce qu'il faudrait ajouter** : dans cette même boucle, compter chaque **touche
  DISTINCTE** (le prix est ressorti de la zone puis y est revenu — pas chaque
  bougie consécutive à l'intérieur) avec son horodatage → champs
  `touch_count: int` + `touch_timestamps: list[datetime]` sur `OrderBlock`/
  `FairValueGap`. Définir « touche distincte » = transition dehors→dedans.
- **Où** : dans les **mappers** (lifecycle), **pas** dans `SmartMoneyEngine`. Le
  moteur détecte les zones ; le mapper calcule leur cycle de vie. La boucle existe
  déjà là. **Recommandation : mappers** (léger, local, testable en isolation).
- **Risque de régression** : **BOS / CHOCH = nul** (collecteurs séparés, non
  touchés). **OB / FVG = faible si confiné au compteur** — ne PAS toucher la règle
  de `mitigated` (mitigé dès la 1re touche) ni l'invalidation. Deux points de
  vigilance : (1) bump `READING_LOGIC_VERSION` → ré-émission de TOUT le cache →
  rejouer la garde LB-1 « incrémental == complet » sur les 6 unités ; (2) valider
  le décompte sur l'échantillon MT-D1 (sur/sous-comptage des touches).
- **Coût estimé** : **moyen** (~1–2 j) — compteur + horodatage + champs schéma +
  tests + bump cache + validation échantillon. Débloque `zone_tested_at_most` (#9)
  ET la chronologie « Formé → Testé ×N → Mitigé » de la page Zones (sans lui, cette
  page n'a rien à raconter).

### 10.B — Phase `distribution` inatteignable (défaut de détection)

- **Phases réellement émettables** (`_derive_market_phase`, unique producteur) :
  **expansion** (directionnel + vol. élevée), **trend** (directionnel + vol.
  normale/faible), **ranging** (indéterminé + clôtures oscillantes),
  **accumulation** (indéterminé sinon).
- **Inatteignable** : **`distribution`** — aucun chemin ne l'émet.
- **Hypothèse de cause** : post-TR-1, la tendance ne vaut plus que
  bull/bear/indéterminée ; la dérivation ne branche que sur (directionnel + vol.)
  ou (indéterminé + oscillation). `distribution` (phase directionnelle-mais-en-
  essoufflement, jadis appariée à un sommet/retournement) n'a plus de règle
  d'entrée. Deux issues possibles pour la mission : (a) réintroduire une règle
  d'émission fondée sur un essoufflement mesurable (volume/divergence/rejet au
  sommet), ou (b) acter officiellement 4 phases et retirer `distribution` du
  schéma `MarketPhase`. Retiré de la palette scanner en attendant.
