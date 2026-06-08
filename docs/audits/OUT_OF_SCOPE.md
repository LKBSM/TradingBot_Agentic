# OUT_OF_SCOPE — Descriptive Quality Audit (2026-05-27)

Sujets rencontrés pendant l'audit qui sortent du périmètre descriptif, avec traçabilité.

## 1. Sentiment news (H.3 — `sentiment_score`, `sentiment_confidence`)

**Statut** : ❓ Non évaluable rigoureusement.
**Raison** : les feeds RSS sources (Reuters / Bloomberg / Investing.com) **ne sont pas archivés** côté projet. Sans corpus textuel historique, on ne peut ni reconstruire le sentiment "vrai" ni mesurer la justesse du sentiment produit par l'algo.
**Annotation manuelle écartée** : 50 events seraient statistiquement insuffisants (IC 95 % trop large) et introduiraient un biais d'annotateur unique. Mieux vaut ne rien dire que dire faux.
**Action recommandée pour V2** :
- Démarrer l'archivage daily des feeds RSS dès cette semaine (cron, ~50-100 MB/mois compressé)
- **12 mois minimum** d'historique requis avant d'avoir un échantillon évaluable
- Date de re-évaluation possible : **2027-05-27** au plus tôt
- Sources prioritaires : Reuters Markets, Bloomberg Top News, Investing.com Forex, ForexFactory Calendar

## 2. Intervalle conformel sur la conviction (D — UncertaintyContext.conformal_lower/upper)

**Statut** : ❓ Non évaluable dans ce périmètre descriptif.
**Raison** : la définition de l'outcome conformel (cf. doc `client_information_explained.txt:247`) est **« R-multiple du backtest »** — un outcome de trading, donc dans le périmètre de l'audit du 2026-05-27 (AUDIT_ALGO), pas de cet audit descriptif. Évaluer la couverture conformelle exigerait de définir un outcome `outcome_i` ; le seul disponible (R-multiple) est non descriptif.
**Risque commercial identifié** : ce claim **est aujourd'hui exposé au client** (`uncertainty.conformal_lower`, `conformal_upper`, `empirical_coverage`, `n_calibration`) **sans qu'on puisse le défendre empiriquement dans le cadre indicateur descriptif**.
**Action recommandée pour le produit visible** :
- **Retirer** ou **minimiser** l'exposition de cet intervalle dans la webapp B2C et le mockup `client_view_full.html` tant que le claim n'est pas validé (cf. statut `edge_claim=False`)
- **Conserver** la mécanique côté backend (utile pour l'audit interne et la phase de validation OOS), mais ne pas l'afficher au client
- Cohérent avec la décision pivot 2026-05-27 (`pivot_positioning_audit.md`) : retirer tous claims non validés des copies clients
- À ré-exposer **uniquement quand** `edge_claim=True` (PF rolling > 1.20 + DSR > 1.0 + PBO < 0.5 + WF ≥ 2 ans OOS)

## 3. Sources RAG (sources_cited, K.4)

**Statut** : ❓ Non évaluable.
**Raison** : marqué « Phase 2B en cours » dans le doc, dossier `src/intelligence/rag/` présent mais non wired en prod. Pas de claim à auditer aujourd'hui.
**À ré-évaluer** : quand la Phase 2B livre.

## 4. liquidity_zone_upper / lower (E.8)

**Statut** : ❓ Champ aspirationnel.
**Raison** : explicitement noté `null en prod` dans le doc. Pas implémenté côté algo.
**Action** : OK tel quel — pas exposé au client tant que non implémenté.

## 5. Hors-mission rencontrés au passage

- **Performance trading** (PF, hit rate, R-multiple, Sharpe) : traité par l'audit `AUDIT_ALGO_2026_05_27.md`. **Ne pas re-discuter ici.**
- **LLM narratif** : qualité éditoriale / hallucinations → hors mission descriptive. Pour l'audit RAG voir la Phase 2B quand elle livre.
- **Architecture / infra / latence** : hors mission.
- **Compliance / pricing / frontend** : hors mission.

Si un point ci-dessus est ré-ouvert pendant l'exécution, je l'ajoute ici sans interrompre l'audit.
