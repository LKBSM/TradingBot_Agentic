# Templates légaux V0 — M.I.A. Markets

**Date** : 2026-05-26
**Statut** : V0 bootstrap — à migrer V1 avocat-signed à M3
**Référence** : `docs/governance/legal_bootstrap_strategy_2026_05_26.md`

---

## Contenu du dossier

| Fichier | Usage | Effort customisation |
|---|---|---|
| `disclaimer_compliance.md` | Wording multilingue FR/EN à coller sur toutes surfaces | 1h |
| `cgu_cgv_v0_template.md` | Clauses fintech-specific à ajouter au template Iubenda | 4-6h |
| `privacy_policy_v0_template.md` | Privacy Policy RGPD-compliant | 2-3h |
| `mentions_legales_auto_entrepreneur.md` | Mentions légales auto-entrepreneur | 30 min |
| `cookie_notice_minimal.md` | Notice cookies sans bandeau intrusif | 30 min |
| `incident_response_runbook.md` | Process réponse aux 6 types d'incidents | À lire, pas à customiser |

**Effort total customisation** : ~8-10h sur 1-2 jours.

---

## Ordre d'usage recommandé

1. **Lire la stratégie globale** : `../legal_bootstrap_strategy_2026_05_26.md`
2. **Créer auto-entreprise** (`autoentrepreneur.urssaf.fr`) → récupérer SIRET sous 7-14 jours
3. **Souscrire Iubenda Pro** ($30/mo) → générer CGU/CGV/Privacy/Cookie de base
4. **Customiser les 5 templates V0** dans l'ordre :
   - Mentions légales (besoin SIRET) → 30 min
   - Disclaimer compliance (audit wording site) → 1h
   - CGU/CGV (insertion clauses dans Iubenda) → 4-6h
   - Privacy Policy (renforcement Iubenda) → 2-3h
   - Cookie notice (si Plausible self-hosted déployé) → 30 min
5. **Lire le runbook incidents** pour anticiper les cas
6. **Publier sur le site** :
   - `/mentions-legales`
   - `/cgu`
   - `/privacy`
   - `/cookies`
   - Disclaimer footer permanent
7. **Souscrire RC Pro Freelance** (Hiscox / Wemind) : 300-500 €/an
8. **Tester** : checkout Stripe en mode test, vérifier que toutes les pages CGU/Privacy sont liées dans le funnel

---

## Coût total stack légal V0

| Ligne | Coût annuel |
|---|---|
| Auto-entreprise (création) | 0 € |
| Iubenda Pro | ~360 € |
| RC Pro Freelance basique | 300-500 € |
| Cyber-risque (optionnel) | 150-300 € |
| Domiciliation (optionnel, sinon adresse perso) | 0-360 € |
| Compte bancaire pro (au-delà 10k€/an CA) | ~120 € |
| **Total bootstrap** | **~750-1640 €/an** |

vs **3-5 k€ avocat fintech one-shot** non disponible.

→ Bootstrap viable. À migrer V1 dès que revenue le permet (cf. `../legal_migration_plan_to_lawyer.md`).

---

## Limitations de V0

Les templates V0 sont **bons mais pas parfaits**. Un avocat fintech ajoutera à M3 :

- Clauses MiFID 2024/2811 spécifiques avec exemptions financial influencer documentées
- Limitation de responsabilité optimisée selon jurisprudence fintech récente FR
- Médiation conso plateforme nommée + procédure intégrée
- DPA B2B complet pour tier INSTITUTIONAL
- Clauses internationales si extension géographique au-delà FR+BE+CH+LU
- Audit conformité finfluencer avec engagement de responsabilité de l'avocat

**Risque résiduel acceptable pendant M0-M3** si Piliers 3 (posture éducative), 5 (cap utilisateurs), 6 (refund 30j) sont strictement respectés.

---

## Maintenance

Les templates V0 doivent être réindicatés :
- **Tous les 3 mois** : audit changements réglementaires UE (RGPD, MiFID, finfluencer)
- **À chaque évolution produit majeure** : ajout fonctionnalité, nouveau pays, nouveau tier
- **À M3 lors migration V1** : remplacement complet par version avocat-signed

Tenir un changelog dans `docs/governance/legal_archive/changelog.md`.
