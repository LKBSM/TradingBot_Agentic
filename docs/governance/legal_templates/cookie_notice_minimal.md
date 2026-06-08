# Notice Cookies minimaliste — M.I.A. Markets

**Date** : 2026-05-26
**Justification** : avec un stack analytique Plausible self-hosted (sans cookies, sans fingerprinting), **aucun bandeau cookies intrusif n'est nécessaire** au sens de la recommandation CNIL 2024 sur les exemptions de consentement.

Une simple page d'information accessible suffit. Ce template est cette page.

---

## Politique relative aux cookies

**Dernière mise à jour** : 2026-05-XX

### En résumé

M.I.A. Markets **n'utilise pas de cookies publicitaires, de cookies de profilage ni de trackers tiers**.

Le seul "cookie" présent est un **cookie de session technique**, indispensable pour vous maintenir connecté à votre compte. Ce type de cookie est **exempté de consentement** par l'article 82 alinéa 2 de la loi Informatique et Libertés et par la recommandation CNIL de 2020.

---

### 1. Cookies utilisés

| Cookie | Type | Finalité | Durée | Consentement requis |
|---|---|---|---|---|
| `ss_session` | Technique strictement nécessaire | Maintien de l'authentification utilisateur | Session (suppression à la fermeture du navigateur) | Non — exempté CNIL |
| `ss_csrf` | Sécurité | Protection contre les attaques CSRF | Session | Non — exempté CNIL |
| `ss_lang` | Préférence | Mémorisation langue choisie | 12 mois | Non — exempté (préférence utilisateur) |

### 2. Pas de cookies tiers

M.I.A. Markets **n'utilise pas** :

- ❌ Google Analytics (remplacé par Plausible self-hosted)
- ❌ Meta Pixel / Facebook tracking
- ❌ TikTok Pixel
- ❌ LinkedIn Insight Tag
- ❌ Cookies publicitaires (Google Ads, Criteo, etc.)
- ❌ Cookies de profilage cross-site
- ❌ Fingerprinting navigateur

### 3. Analytique sans cookie : Plausible

Notre outil d'analytique produit est **Plausible Analytics, en mode self-hosted sur nos serveurs Fly.io en France**.

Caractéristiques :
- **Pas de cookies posés sur votre navigateur**
- **Pas d'identifiant unique persistant** ni de fingerprinting
- **Pas de transfert de données vers des tiers** (self-hosted)
- **Données agrégées uniquement** (pages vues, temps passé, source de trafic)
- **Conforme RGPD par construction** (pas de PII collectée)
- **Approuvé par la CNIL** comme alternative exemptée de consentement (recommandation 2024)

Plus d'informations : `plausible.io/data-policy`

### 4. Comment gérer les cookies

Bien que nos cookies techniques soient exemptés de consentement, vous gardez le contrôle :

- **Mozilla Firefox** : `Paramètres → Vie privée et sécurité → Cookies`
- **Google Chrome** : `Paramètres → Confidentialité et sécurité → Cookies`
- **Safari** : `Préférences → Confidentialité → Cookies`
- **Microsoft Edge** : `Paramètres → Cookies et autorisations de site`

⚠️ Si vous bloquez tous les cookies, l'authentification à votre compte ne fonctionnera plus.

### 5. Évolution

Si à l'avenir nous introduisons des cookies non exemptés (par exemple, retargeting publicitaire), un **bandeau de consentement explicite conforme CNIL** sera affiché, avec opt-in clair, opt-out facile, et historique de votre choix.

Tant que cette page reste en vigueur, aucun cookie nécessitant consentement n'est utilisé.

### 6. Contact

Question sur les cookies : `privacy@mia.markets`

---

## ⚙️ Notes pour l'implémentation

### Stack analytique recommandé bootstrap
- **Plausible Analytics self-hosted** sur même VPS Fly.io que l'app principale
- Image Docker officielle : `plausible/analytics:latest`
- Coût ~$0 supplémentaire (mutualisation infra)
- 5 minutes d'install via docker-compose

### Pourquoi PAS un bandeau cookies intrusif
- Plausible self-hosted = pas de cookies tiers, pas de tracker = pas d'obligation CNIL
- Cookies techniques + préférence langue = exemptés (CNIL art. 82.2)
- Bandeau intrusif = friction UX inutile + bounce rate +5-15 % observé en B2C SaaS
- **Page d'info accessible suffit légalement**

### Si tu dois ajouter un tracker (à éviter en bootstrap)
- Tracker pub (Meta, Google Ads) → bandeau cookies obligatoire (Tarteaucitron self-hosted recommandé, DG-048 activable à ce moment-là)
- Coût qualité UX vs bénéfice acquisition : généralement défavorable pour SaaS B2C niche fintech
- Recommandation : **rester sans tracker pendant bootstrap M0-M3**, évaluer post-PMF

---

## Checklist publication

- [ ] Plausible self-hosted déployé sur Fly.io
- [ ] Script analytics intégré au front (script `<script defer data-domain="mia.markets" src="..."></script>`)
- [ ] Cookie `ss_session` configuré avec attributs `HttpOnly`, `Secure`, `SameSite=Lax`
- [ ] Cookie `ss_csrf` configuré avec attributs `HttpOnly`, `Secure`, `SameSite=Strict`
- [ ] Cookie `ss_lang` configuré avec `SameSite=Lax`, durée 12 mois
- [ ] Publication de cette politique sur `/cookies`
- [ ] Lien dans footer
- [ ] Pas de bandeau cookies (cohérent avec recommandation CNIL exemptions 2024)
- [ ] Audit final : ouvrir DevTools → Application → Cookies → vérifier qu'aucun cookie tiers n'est posé
