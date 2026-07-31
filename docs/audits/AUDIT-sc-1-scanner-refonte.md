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
| 2 | L'unité supérieure va (même/opposé/indéterminée) | (2) calculable | ⏸ **différé** | — (voir §2) |
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
| 20 | Dernier événement remonte à (<10/10-50/>50) | (2) calculable | ✅ | `last_event_age` |

**Décompte** : (1) déjà = 10 · (2) calculable = 9 · (3) bloquée = 1.
**Livrées : 18 des 20** + `price_in_tested_zone` (#21 nouveau). Non livrées : #9
(bloquée) et #2 (différée, cf. §2). Conservée transitoirement : `mtf_aligned`.

Conditions **retirées** (signalées, pas supprimées en silence) :
- `ob_fvg_confluence` — exactement « prix dans OB » ET « prix dans FVG » ; la règle
  produit interdit d'offrir deux fois la même chose sous un troisième nom.
- ancien `retest_in_progress` — c'était un flag de retest de **NIVEAU BOS** (état
  ARMED du state-machine), pas « prix dans une zone testée ». Décision fondateur :
  retirer et le remplacer par le fait `price_in_tested_zone`.

## 2. Condition #2 différée (et non bloquée)

#2 « l'unité supérieure va » est **calculable** aujourd'hui (TF-1 a tranché
l'alignement relatif ; `mtf_confluence` porte le biais structurel des unités
au-dessus). Elle est **différée par décision fondateur** (point 5) : `mtf_aligned`
est conservé transitoirement, relibellé pour dire exactement ce qu'il compare
(**3 unités FIXES H4/H1/M15**, pas « l'unité au-dessus » de l'unité scannée). Pour
**ne pas se retrouver avec les deux**, #2 arrivera quand `mtf_aligned` sera retiré.

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
- **Réévaluation** : à l'ouverture uniquement (recommandé). Le compte de combos
  par lecture enregistrée « à jour » (mission D) est **différé** (nécessite un scan
  par lecture ; l'arrière-plan n'apporte rien tant que le store n'est rafraîchi
  que par le scheduler 60 s). À câbler avec la réévaluation on-open.
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

## 7. Écarts avec la maquette

`docs/design/reference-scanner.html` était **absent** ; le fondateur a demandé de
concevoir la maquette moi-même et de réconcilier l'écran ensemble ensuite. La
maquette 5 états jointe reflète l'implémentation. **Réconciliation live à faire
avec le fondateur** (verrou de merge). Écarts connus à ce stade :
- Compte de combos par lecture enregistrée : non affiché (différé, §5).
- `higher_tf_agrees` (#2) absent (différé, §2).

## 8. Vérifications

- Backend : 56 tests scanner + 42 adjacents verts.
- Front : `tsc` 0, `next build` exit 0, vitest scanner verts (palette 4 familles,
  garde-fous résultats, non-régression store).
- Playwright 1280×800 + 390×844 sur les 5 états : specs fournies ; exécution
  complète nécessite le backend `:8000` (données réelles) — à lancer live.

## 9. Reste à faire avant merge

1. Réconciliation live de la maquette avec le fondateur.
2. Trancher #2 vs `mtf_aligned` (ne pas garder les deux).
3. Compte de combos par lecture enregistrée (réévaluation on-open).
4. Exécuter Playwright contre un backend réel.
5. **Merge sur main SEULEMENT après confirmation live du fondateur.**
