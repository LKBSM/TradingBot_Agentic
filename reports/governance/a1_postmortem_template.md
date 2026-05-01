# Post-mortem Test A1 — YYYY-MM-DD

> **Document critique de transition Phase 1 → Phase 2A ou 2B.**
> À remplir mécaniquement, **avant** toute discussion émotionnelle.
> Référence plan : `reports/roadmap_2026_2027/PLAN_12_MOIS.md` (Partie VI.4).
> Owner : Elena (calcul) + Sofia (validation).

---

## 1. Verdict mécanique

| Métrique | Cible | Observé | Status |
|---|---|---|---|
| **DSR** (Deflated Sharpe Ratio) | > 1.0 | __ | 🟢/🔴 |
| **PBO** (Probability of Backtest Overfitting) | < 0.3 | __ | 🟢/🔴 |
| **CPCV PF moyen** (28 paths) | > 1.20 | __ | 🟢/🔴 |
| **CPCV PF p25** | > 1.05 | __ | 🟢/🔴 |
| **Holm-significant features** (α=0.05) | ≥ 3 | __ | 🟢/🔴 |
| **DM test vs HAR baseline** p-value | < 0.05 | __ | 🟢/🔴 |
| **DM test vs constant baseline** p-value | < 0.01 | __ | 🟢/🔴 |
| **SHAP top-3** inclut ≥1 macro non-redondante | oui/non | __ | 🟢/🔴 |

**Score green** : __/8 critères

---

## 2. Décision automatique (ne pas négocier)

- Si **8/8** 🟢 → **GO 2A**
- Si **5-7/8** 🟢 dont DSR ET PBO → **GO 2B+** (Phase 2B avec emprunt sélectif 2A : QUANT-2A.6 calibration + REGIME-2A.2 Jump Model)
- Si **<5/8** 🟢 OU DSR<0 OU PBO>0.5 → **GO 2B**

**Décision formelle** : ☐ 2A   ☐ 2B+   ☐ 2B

---

## 3. Apprentissages techniques

### 3.1 Features
- **Feature qui a le plus surpris** : __ (positif/négatif, magnitude SHAP : __)
- **Feature qui n'a pas marché alors que je m'y attendais** : __ (raison hypothétique : __)
- **Feature top-1 SHAP** : __ (interprétation : __)
- **Feature top-2 SHAP** : __
- **Feature top-3 SHAP** : __

### 3.2 Hypothèses
- **Hypothèse réfutée** : "{Hypothèse de départ : ex. 'COT z52 ferait passer Holm'}" → résultat : {p-value : __, conclusion : __}
- **Hypothèse confirmée** : __

### 3.3 Modèle
- **Best fold CPCV PF** : __ | **Worst fold CPCV PF** : __ | **Spread** : __
- **Stack niveau 2 ajoute-t-il par rapport à niveau 1 le meilleur** : oui / non, magnitude : __
- **Sur-fit suspecté ?** : oui / non, indices : __

---

## 4. Apprentissages méthodologiques

1. **Erreurs de protocole détectées** (le cas échéant) : __
2. **Améliorations CPCV pour next iteration** : __
3. **Quality of dataset** :
   - Gap suspecté : oui / non — __
   - Bias suspecté : oui / non — __
   - Leak suspecté : oui / non — __ (résultat des tests vintage : __)
4. **Comparaison à `falsification_2026_04_30.md`** :
   - Cet audit avait estimé prob PF>1.20 à 30-40%. Verdict actuel : __
   - β-capture risk : corrélation A1 prédictions avec returns spot XAU = __
   - Si corr > 0.85 → β-capture confirmé, dégrader confiance même en cas DSR > 1

---

## 5. Implications produit

> 3-5 lignes sur ce que ce verdict implique pour le pivot Phase 2.

{texte}

---

## 6. Plan immédiat (jours suivants la décision)

### Si GO 2A :
- [ ] Notifier équipe (publique/communauté Discord) du verdict + plan
- [ ] Démarrer INFRA-2A.1 (ONNX serving) en S9
- [ ] Démarrer INFRA-2A.2 (forward-test paper harness, gate Stripe) en S9 — **PRIORITÉ ABSOLUE**
- [ ] Verrouiller code A1 v1 dans `models/a1_v1.0.0.pkl` (versioning semver)
- [ ] Karim : activer brief `positioning_2A_edge_confirmed.md`

### Si GO 2B (ou 2B+) :
- [ ] Notifier équipe (publique/communauté Discord) du verdict + pivot — **transparence renforce moat 2B**
- [ ] Démarrer INFRA-2B.1 (webapp infra) en S9
- [ ] Démarrer LLM-2B.1 (RAG architecture) en S9 en parallèle
- [ ] Karim : activer brief `positioning_2B_narrative_first.md`
- [ ] Mettre à jour landing/Discord avec messaging 2B

---

## 7. Engagement écrit (anti-rationalisation)

> Cet engagement est l'antidote au piège classique du "et si je rejouais le test A1 avec d'autres features".

> Je, **{nom}**, m'engage par écrit à exécuter Phase **{2A / 2B / 2B+}** telle que définie dans le plan 12 mois (`reports/roadmap_2026_2027/PLAN_12_MOIS.md`), sans rationaliser un retour vers la phase non-choisie pendant **≥ 90 jours**, sauf incident kill criteria explicite documenté.
>
> Je m'engage également à publier publiquement (blog interne, Discord, ou newsletter) le résultat de ce post-mortem dans les 7 jours, pour cristalliser la décision et réduire le biais de réécriture.

**Signature** : ___________________  
**Date** : YYYY-MM-DD  
**Validation Sofia** : ___________________

---

## 8. Annexes (liens vers artefacts)

- Verdict chiffré complet : `reports/a1_verdict_2026.md`
- Modèle versionné : `models/a1_stack_v1.pkl`
- CPCV result raw : `data/research/cpcv_a1_paths.parquet`
- SHAP values : `reports/a1_shap.html`
- Feature matrix : `data/research/a1_matrix_2019_2025.parquet`
- Post-mortem précédent (si relancé) : N/A

---

## 9. Changelog (append-only)

- YYYY-MM-DD HH:MM : post-mortem rédigé et signé
- YYYY-MM-DD HH:MM : {correction ou complément éventuel}
