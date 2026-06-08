# HAIKU_COHERENCE_AUDIT_2026_06_06 — Évaluation de cohérence des descriptions Haiku

> **Audit de validation algorithmique — Phase 3**
> Date : 2026-06-07 · Branche : `audit/validation-algorithmique-detection-structures`
> Script : `scripts/evaluate_haiku_coherence.py` · Données : `data/validation/haiku_coherence_results.json`
> n = **60** readings évalués (55 Haiku live + 5 fallback template — superset des 50 demandés).

---

## 1. Méthodologie

Chaque description (champ `haiku_description`) est confrontée aux **données structurées**
du même MarketReading (tags + structure + régime + events) via 4 tests automatiques :

| Test | Question | Méthode |
|---|---|---|
| **A — Mention** | Les éléments présents dans les tags sont-ils mentionnés dans la prose ? | lexique de synonymes par élément (BOS/CHOCH/OB/FVG/news) |
| **B — Cohérence directionnelle** | La prose contredit-elle le `trend` du régime ? | mots haussier/baissier vs `regime.trend`, neutralisé si MTF mixte |
| **C — Forbidden tokens** | La prose contient-elle un token interdit niveau 1.5 ? | `OutputFilter` **chatbot canonique** (`ALL_FORBIDDEN_TOKENS`, 4 catégories) |
| **D — Pas d'invention** | La prose mentionne-t-elle un élément **absent** des données ? | mention d'un élément sans le tag correspondant |

> **Rappel architecture (F5)** : le moteur Haiku ne reçoit que `tags` + `regime`
> (`haiku_description_engine.py:97`). Il n'a jamais accès aux niveaux de prix ni à
> l'objet `structure` complet — il infère la présence BOS/OB/FVG **uniquement via les tags**.
> Les tests A et D sont donc évalués contre les **tags**, ce que le LLM voit réellement.

---

## 2. Métriques agrégées

| Test | Résultat | Seuil cible | Statut |
|---|---|---|---|
| **C — Sans forbidden token** | **100.0 %** | = 100 % (gate critique) | 🟢 **PASS** |
| **D — Sans invention** | **100.0 %** | élevé | 🟢 **PASS** |
| **B — Cohérence directionnelle** | **100.0 %** (41 applicables) | élevé | 🟢 **PASS** |
| **A — Mention (couverture complète)** | **0.0 %** (60 applicables) | — | 🔴 **À traiter** |

**Détail Test A par élément** (mentionné / applicable) :

| Élément | Mention | Applicable | Taux |
|---|---|---|---|
| BOS | 0 | 60 | **0 %** |
| CHOCH | 0 | 14 | **0 %** |
| Order Block | 0 | 8 | **0 %** |
| FVG | 3 | 12 | **25 %** |
| News | — | 0 | n/a (events vides) |

---

## 3. Interprétation — le Test A bas n'est PAS une hallucination

> **Conclusion centrale** : aucune contradiction, aucune invention, aucun token interdit
> (B/C/D = 100 %). Le Test A bas traduit une **omission descriptive**, pas une incohérence.
> Le seuil de STOP « > 30 % de descriptions contredisant les données » (cas critique #4)
> **n'est PAS franchi** : 0 % de contradiction (Test D), 0 % de token interdit.

Trois causes structurelles, toutes documentées :

### 3.1 Le tag `bos_recent_*` est présent sur **100 %** des readings (cause dominante — finding F6)
- 60/60 readings portent un tag `bos_recent_*`, mais seulement **28/60 sont des BOS frais**
  (`validation_status="confirmed"`, `BOS_EVENT≠0`). Les **32 autres** sont l'**état de tendance
  propagé** (`BOS_SIGNAL≠0`, `validation_status="pending"`) — pas un break récent.
- Un tag quasi-omniprésent (et à moitié « périmé ») n'apporte aucune information discriminante :
  un LLM correct l'ignore plutôt que de répéter « BOS » à chaque phrase. **L'omission est ici
  un comportement raisonnable du LLM, induit par un défaut de design du tag**, pas une faute Haiku.

### 3.2 Le moteur Haiku est **conçu pour décrire le RÉGIME en une phrase**
- System prompt (`haiku_description_engine.py:42-49`) : « Tu écris UNE phrase descriptive ».
- Le user-prompt met le régime (trend/vol/phase/mtf) au premier plan ; la structure n'arrive
  que via la liste de tags. Le LLM privilégie donc systématiquement le régime + l'alignement MTF.
- C'est cohérent avec la sobriété niveau 1.5 — **mais cela signifie que la prose ne résume PAS
  la structure**. La structure est portée par les **champs structurés + tags**, pas par la phrase.

### 3.3 FVG mentionné à 25 % seulement
- Quand `fvg_active` est présent (12 cas), Haiku le mentionne 3 fois (souvent via « déséquilibre »
  ou « zone »). Les Order Blocks (8 cas) ne sont jamais nommés. Cohérent avec 3.1/3.2.

---

## 4. Patterns problématiques identifiés

| # | Pattern | Occurrences | Sévérité |
|---|---|---|---|
| P1 | **BOS jamais mentionné** alors que le tag est sur 100 % des readings | 60/60 | 🟡 (artefact du tag F6) |
| P2 | **OB jamais mentionné**, CHOCH jamais mentionné | 8/8, 14/14 | 🟡 |
| P3 | **Phrasing vague/garbled** : ex. #3 « des signaux heureux récents » (tags BOS+CHOCH bullish) — paraphrase floue qui ne nomme aucune structure | ≥1 net | 🟡 |
| P4 | **5/60 fallback template** car le moteur a rejeté la sortie Haiku sur le token `entre` (« FVG entre X et Y »), homonyme faux-positif | 5/60 | 🟠 incohérence inter-systèmes |

### Zoom P4 — incohérence des deux ensembles de forbidden tokens
- Le moteur Haiku MarketReading filtre avec `market_reading_mappers.FORBIDDEN_TOKENS`
  (13 tokens) qui inclut le **bare `entre`**.
- Le système chatbot (`chatbot/constants.py:22-47`) **exclut délibérément** `entre`
  (homonyme « entre X et Y » = preposition), ne gardant que `entrez`/`entrer`/`entry`.
- Conséquence : des descriptions Haiku **légitimes** (« FVG entre 2376 et 2378 ») sont
  rejetées et retombent en template par le moteur MarketReading, alors que le filtre canonique
  du chatbot les laisserait passer. **Les deux gardes divergent.**

### Citations de descriptions (échantillon)
- #1 (tags `bos_recent_bearish`, `mtf_aligned`) : *« Le marché oscille dans une bande de consolidation avec une volatilité normale et un alignement haussier sur les timeframes supérieures. »* → régime OK, BOS non nommé.
- #3 (tags `bos_recent_bullish`, `choch_recent_bullish`, `fvg_active`) : *« …des signaux heureux récents… »* → vague/garbled, P3.
- #31 (tags `bos_recent_bullish`, `mtf_mixed`, trend bearish) : *« …tendance baissière… le jour montre une orientation haussière… »* → directionnellement cohérent (MTF mixte), B PASS.

---

## 5. Recommandations

### 🔴 Avant bêta
1. **Corriger le tag `bos_recent_*` (F6)** : ne l'émettre que sur `BOS_EVENT≠0`
   (break frais), pas sur l'état propagé `BOS_SIGNAL`. Sinon le produit affiche « BOS récent »
   en permanence (53 % des cas sont en réalité un simple état de tendance). Touche
   `market_reading_mappers.py` (mapping), **pas le moteur de détection** — à valider founder.
2. **Aligner les deux ensembles de forbidden tokens (P4)** : retirer le bare `entre` de
   `market_reading_mappers.FORBIDDEN_TOKENS` (garder `entrez`/`entrer`/`entry`), comme le fait
   déjà `chatbot/constants.py`. Réduit les fallbacks template injustifiés.

### 🟠 Important (1 mois)
3. **Décider du rôle de la prose** : soit (a) assumer que la description = **résumé de régime**
   (et le documenter côté produit / `/methodology`), soit (b) si on veut que la prose couvre
   la structure, **passer la `structure` au moteur Haiku** (et enrichir le prompt pour nommer
   BOS-frais / OB / FVG quand pertinents). Recommandation : **(a)** — plus sobre, plus sûr,
   moins de surface d'erreur. La structure reste exposée via les champs structurés + tags.
4. **Filtre anti-vague** : ajouter une vérification légère que la phrase nomme au moins le
   trend + la phase (éviter les « signaux heureux » de P3).

### 🟢 Nice-to-have
5. Étendre l'évaluation à un échantillon plus large + suivi de régression (CI) une fois le
   tag F6 corrigé, pour mesurer si le taux de mention BOS-frais remonte.

---

## 6. Verdict Phase 3

| Dimension | Verdict |
|---|---|
| **Sécurité niveau 1.5 (forbidden tokens)** | 🟢 **100 % clean** — aucun glissement directif. Gate critique tenu. |
| **Honnêteté (pas d'invention/hallucination)** | 🟢 **100 %** — Haiku n'invente jamais une structure absente des données. |
| **Cohérence directionnelle** | 🟢 **100 %** — aucune contradiction haussier/baissier. |
| **Complétude descriptive** | 🔴 faible mais **par omission, pas par erreur** — pilotée par le tag F6 + le design « 1 phrase régime ». Corrigeable sans risque. |

**Aucun cas STOP déclenché.** Les descriptions Haiku sont **sûres et honnêtes** ; elles sont
**centrées régime** et sous-décrivent la structure. Le correctif prioritaire (tag F6 + alignement
forbidden tokens) est côté **mapping**, pas côté moteur de détection ni LLM.
