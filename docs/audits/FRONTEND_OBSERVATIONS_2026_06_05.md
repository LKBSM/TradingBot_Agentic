# Observations frontend du founder — landing page (visualisation locale)

**Date** : 2026-06-05
**Auteur** : Loukmane Bessam (founder)
**Contexte** : visualisation locale de la landing page après livraison des
Chantiers 5.A (décommissionnement chatbot) et 5.B (vue applicative `/app`).
**Traité dans** : Sous-Chantier 5.C — Nettoyage + suppression résidus
positionnement pré-pivot.
**Référence pivot** : `docs/governance/decisions/2026-05-27_pivot_positioning_audit.md`
(`edge_claim=false`, niveau 1.5 strict, aucune promesse de gain).

---

## Citation du founder

> « Pourquoi cette partie parle de profit si mon produit a même pas besoin de
> trade ? Faut qu'on évalue mon indicateur précisément sur ses fonctionnalités
> seulement. »

Le produit est un **indicateur descriptif de compréhension du marché**, pas un
système de trading. Toute la copie doit donc évaluer l'indicateur **sur ses
fonctionnalités** (structure SMC, régime, événements, synthèse), jamais sur une
performance de trading.

---

## 🔴 Glissements niveau 2 identifiés (résidus pré-pivot du 27 mai)

| # | Résidu observé | Emplacement | Pourquoi c'est un problème |
|---|---|---|---|
| 1 | **Score de conviction 62/100** sur la « Lecture MIA » | `BeforeAfterSection` + `ConvictionGauge` (carte) | Score implicitement directif — un indicateur descriptif ne note pas une décision |
| 2 | **Section entière « Backtest 7 ans »** : Profit Factor 0,786, Deflated Sharpe Ratio 0,000, Probability of Backtest Overfitting 0,50, sous-performance −318 pts, mention « LightGBM 2 niveaux » | `HonestConfidenceSection` (4 StatCards + note technique) | Parle de profit/performance de trading — hors périmètre d'un indicateur ; chiffres pré-pivot |
| 3 | **Section « Ce sur quoi nous travaillons »** : « Améliorer la valeur prédictive du scoring », « Atteindre un edge mesurable (DSR > 1, PF lo > 1,05) », « Tester en paper-trading » | `HonestConfidenceSection` (3e colonne) | Promesse implicite d'edge futur — interdit niveau 1.5 |
| 4 | **Section « Amplitude prévue »** : « Le modèle anticipe une expansion » | (composants `insight/` — `VolatilitySection`) | Prédiction explicite — l'indicateur décrit, ne prédit pas |
| 5 | **Section « Historique des setups similaires »** | (composants `insight/` — `HistorySection`) | Implique la répétition future des patterns passés (+ bug `formatProfitFactor(null)`) |

---

## Sections à GARDER (validées par le founder)

Hero · MultiMarket · ConversationReplay · Pricing · FAQ · **Engagement public**
(citation imposée lock 2) · **« Ce que nous ne ferons jamais »**.

---

## Traitement appliqué (Chantier 5.C)

- **#1 (62/100)** : retiré de `BeforeAfterSection` (texte + SVG) ; `ConvictionGauge`
  supprimé avec tout `components/insight/` (la carte landing consomme désormais
  `MarketReadingCard` natif → `MarketPhasePanel`, descriptif).
- **#2 (backtest 7 ans)** : `HonestConfidenceSection` refactorée — 4 StatCards +
  note technique edge_claim/CPCV/PBO supprimées ; citation « Engagement public »
  et colonnes « jamais » / « aujourd'hui » conservées.
- **#3 (ce sur quoi nous travaillons)** : colonne « edge mesurable » supprimée.
- **#4 (amplitude prévue) & #5 (historique setups)** : supprimés avec
  `components/insight/sections/*` (code mort après migration landing) ; le bug
  `formatProfitFactor(null)` disparaît mécaniquement.
- **FAQ q2 / PricingSection** : reformulés sans PF/DSR/PBO/edge/paper-trading.

> Hors-périmètre 5.C (renvoyé en 5.D) : reformulation copy niveau 1.5 fine —
> ATR, retest armé, « intervalle conformel », vulgarisation des termes
> techniques. Le label de section « Honnêteté conformelle » est conservé tel
> quel en attendant cette passe.

---

*Note : ce document a été créé au Chantier 5.C à partir des observations
consignées dans le brief de mission. Le founder peut y ajouter/compléter le
contenu canonique s'il le souhaite.*
