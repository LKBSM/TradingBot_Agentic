# Acceptance criteria — 10 P0-strict-MVP

Format Given/When/Then. À utiliser pour générer les tests automatisés.

---

## DG-160 — Plausible self-hosted

- **Given** un browser propre sans cookie
  **When** je visite `https://mia.markets`
  **Then** la requête vers `https://analytics.mia.markets/js/script.js` est OK 200
  **And** aucun cookie tiers n'est posé dans le navigateur
  **And** un event `pageview` apparaît dans le dashboard Plausible sous 30s

- **Given** Plausible déployé sans `DISABLE_REGISTRATION=invite_only`
  **When** je visite `/register` sur l'instance Plausible
  **Then** la création de compte est bloquée

---

## DG-161 — Event tracking core

- **Given** un user FREE authentifié
  **When** il consulte une lecture XAU
  **Then** un event `signal_view` est capturé avec `{signal_id, instrument: "XAU", direction, tier_user: "FREE", surface: "webapp"}`

- **Given** un user qui pose "Dois-je acheter ?" au chatbot
  **When** la requête /chat/ask est traitée
  **Then** un event `chatbot_question` est émis avec `question_category: "prescriptive_refusal"`

- **Given** un user STARTER qui clique CTA "Upgrade Pro" depuis pricing
  **When** le clic est enregistré
  **Then** event `upgrade_clicked` émis avec `from_tier: "STARTER", to_tier: "PRO", cta_location: "pricing_page"`

- **Given** un user qui complete checkout Stripe pour STARTER
  **When** le webhook `invoice.paid` arrive
  **Then** event `paid_conversion` émis SERVER-SIDE avec `tier, price_id, amount_cents, currency, billing_cycle, trial_converted`

- **Given** un audit des events Plausible
  **Then** aucune propriété ne contient d'email, IP exacte, contenu de message, ou autre PII

---

## DG-101-MODIFIED — Renderer unique + sections collapsibles

- **Given** un user FREE qui visite `/lecture/abc123`
  **When** la page charge
  **Then** la hero card est visible immédiatement (non collapsée)
  **And** les 4 sections collapsibles sont collapsed par défaut
  **And** la section "Détail technique" affiche un badge `🔒 STRATEGIST`

- **Given** un user FREE qui clique sur la section "Détail technique" verrouillée
  **When** le clic est traité
  **Then** la section ne s'ouvre PAS
  **And** un overlay "Section réservée au tier STRATEGIST" est visible avec lien vers `/pricing`

- **Given** un user PRO qui clique "Tout déplier"
  **When** le clic est traité
  **Then** les 4 sections sont ouvertes simultanément
  **And** le bouton change pour "Tout replier"

- **Given** un user STARTER qui clique "Pourquoi cette conviction ?"
  **When** la section s'ouvre
  **Then** event `section_expanded` émis avec `{section_id: "conviction", tier_user: "STARTER", was_locked: false}`

- **Given** un user FREE qui tente d'accéder via API à `/api/v1/lectures/abc123?include=expert_detail`
  **When** la requête arrive au backend
  **Then** le backend refuse 403 ou retourne le payload SANS le détail technique (gating server-side)

---

## DG-103 — Mobile-first responsive

- **Given** un viewport 375×667 (iPhone SE)
  **When** je visite `/lecture/abc123`
  **Then** le hero card s'affiche en 1 colonne (track record + conviction empilés)
  **And** aucun scroll horizontal forcé
  **And** les boutons "Demander à Sentinel" + "Tout déplier" font 100 % largeur

- **Given** un viewport 375×667
  **When** je visite la page lecture
  **Then** le chatbot apparaît en FAB (icône bas-droite) et non en sidebar
  **And** le tap sur le FAB ouvre un overlay plein écran chatbot

- **Given** viewport 1440×900
  **When** je visite la page lecture
  **Then** le chatbot sidebar est visible à droite (largeur ~320px) et persistant

- **Given** un viewport mobile
  **When** je tape un bouton ou un chevron collapsible
  **Then** le touch target fait au moins 44×44px (Apple HIG)

- **Given** test Lighthouse mobile sur `/`
  **Then** score performance ≥ 90, LCP < 2.5s, CLS < 0.1

---

## DG-110 — Wire chatbot 8 composantes

- **Given** un signal avec conviction 72 et breakdown 8 composantes
  **When** un user STARTER demande "Pourquoi 72 ?"
  **Then** la réponse contient les 8 noms de composantes
  **And** la réponse contient les contributions chiffrées (ex "Smart Money +24.5")
  **And** la réponse ne contient aucun mot interdit (signal, achetez, garanti, etc.)
  **And** le model utilisé est `claude-haiku-4-5-20251001` (tier STARTER)

- **Given** un user PRO
  **When** il pose une question
  **Then** le model utilisé est `claude-sonnet-4-6`

- **Given** un user FREE qui a déjà posé 5 questions aujourd'hui
  **When** il pose une 6ème question
  **Then** HTTP 429 retourné avec message "Daily chatbot quota reached"

- **Given** un user pose la même question 2 fois en 30 minutes
  **When** la 2ème requête arrive
  **Then** la réponse vient du cache sémantique (`cache_hit=true` dans la réponse)
  **And** aucun appel Anthropic n'est fait (économie tokens)

- **Given** un user pose 3 questions consécutives dans la même session
  **When** la 3ème requête est traitée
  **Then** le prompt envoyé à Claude inclut les 2 échanges précédents dans la session memory

- **Given** une réponse Claude qui contient malencontreusement "achetez à 2390"
  **When** la réponse passe par le post-processing `contains_forbidden_token`
  **Then** la réponse est remplacée par fallback safe
  **And** un incident est loggé dans `docs/incidents/`
  **And** la métrique `chatbot_forbidden_blocked_total` est incrémentée

---

## DG-112 — Tests adversariaux refus pédagogique

- **Given** un user pose "Dois-je acheter XAU ?"
  **When** la question est analysée par `is_prescriptive`
  **Then** True est retourné
  **And** la réponse est un refus pédagogique scripté (pas un appel Claude)
  **And** la métrique `chatbot_refusals_total` est incrémentée
  **And** la réponse contient un tag visuel `REFUS PÉDAGOGIQUE · compliance UE 2024/2811`
  **And** la réponse inclut le contexte du signal (conviction, régime, event)

- **Given** un user pose "Pourquoi 72 ?" (question légitime)
  **When** la question est analysée
  **Then** False est retourné par `is_prescriptive`
  **And** la question est envoyée à Claude normalement

- **Given** 10 users distincts posent "Dois-je acheter ?"
  **When** les 10 réponses sont retournées
  **Then** au moins 3 templates différents sont utilisés (rotation random)

- **Voir** `adversarial_chatbot_tests.md` pour la liste complète des 30+ patterns

---

## DG-114-REDUCED — 3 questions suggérées contextuelles

- **Given** un signal avec conviction 72 et FOMC dans 2h30
  **When** je fetch `/api/v1/chat/suggestions/{signal_id}`
  **Then** 3 questions retournées
  **And** Q1 contient "72" (dynamique conviction)
  **And** Q3 contient "FOMC" et "2h" (event ≤ 4h priorité)

- **Given** un signal avec conviction 80 et aucun event ≤ 4h
  **When** je fetch suggestions
  **Then** Q3 contient "historiquement" (fallback historical si conviction ≥ 70)

- **Given** un user clique sur chip "Pourquoi 72 ?"
  **When** la question est envoyée au chatbot
  **Then** event `chatbot_question` émis avec `question_source: "suggested"` (pas "free")

- **Given** un user a envoyé 1+ message
  **When** la UI re-render
  **Then** les 3 chips suggested sont masquées (visibles uniquement avant 1er message)

---

## DG-120 — Landing hero card

- **Given** un browser propre visite `/`
  **When** la page charge
  **Then** le H1 contient "L'analyse de marché de niveau institutionnel, traduite"
  **And** 4 stats sont visibles : 329, 1.30, 7 ans, 95 %
  **And** CTA "Essayer gratuitement" → `/signup?tier=free`
  **And** CTA "Voir la méthodologie" → `/methodologie`
  **And** LiveLectureExample affiche un signal demo avec la section "Détail technique" verrouillée

- **Given** audit textuel du contenu de la page `/`
  **Then** aucune occurrence des mots interdits (signal, achetez, vendez, garanti, recommandation, profit X%)
  **And** disclaimer compliance visible en footer permanent

- **Given** un audit Lighthouse mobile
  **Then** score performance ≥ 90
  **And** LCP < 2.5s

- **Given** event tracking actif
  **When** je visite `/`
  **Then** un event `pageview` est émis dans Plausible

---

## DG-132 — Page pricing decoy + dual trial

- **Given** un user visite `/pricing`
  **When** la page charge
  **Then** 4 cards visibles : FREE / Starter / Pro / Institutional
  **And** Pro affiche le badge "RECOMMANDÉ"
  **And** Institutional $1990 visible (decoy permanent)
  **And** toggle mensuel/annuel fonctionne (prix mis à jour <100ms)
  **And** annuel affiche réduction -16,7 % visible

- **Given** un user FREE clique "14 jours d'essai sans CB" sur Starter
  **When** le clic est tracké
  **Then** event `upgrade_clicked` émis avec `from_tier: "FREE", to_tier: "STARTER", cta_location: "pricing_page"`
  **And** redirection vers `/signup?tier=starter&trial=nocard`

- **Given** un user clique CTA Institutional
  **When** le clic est traité
  **Then** redirection vers Calendly demo (pas un checkout direct)

- **Given** viewport mobile 375px
  **When** je visite `/pricing`
  **Then** les 4 cards sont stackées verticalement (pas de scroll horizontal)
  **And** toggle mensuel/annuel visible

- **Given** un user qui s'inscrit en trial STARTER sans CB
  **When** la 14ème journée arrive
  **Then** l'abonnement repasse automatiquement en FREE
  **And** AUCUN débit n'est effectué
  **And** un email D+13 est envoyé la veille (reminder)

---

## DG-142 — Tableau performance public

- **Given** un visiteur non authentifié
  **When** il accède à `/track-record`
  **Then** la page charge sans demander login
  **And** stats agrégées visibles : n trades, win rate, PF avec IC 95 %, DD max
  **And** disclaimer "Paper-trading uniquement" visible

- **Given** filtre période "3m"
  **When** je clique
  **Then** les stats sont recalculées pour les 90 derniers jours
  **And** les trades du tableau sont filtrés idem

- **Given** le cron nightly s'exécute J+1 23:59 UTC
  **When** il termine
  **Then** un nouveau snapshot est sauvegardé en base
  **And** la page `/track-record` reflète les nouvelles stats au prochain refresh

- **Given** un mois sans trade clôturé
  **When** je visite `/track-record?period=1m`
  **Then** message "Aucun trade clôturé sur cette période" affiché (pas crash)

- **Given** stats agrégées
  **When** je compare PF aux valeurs hardcodées du hero landing (1.30)
  **Then** les valeurs correspondent (ou le hero est mis à jour automatiquement depuis l'agrégateur)

---

## Gate de sortie Vague 1 — Synthèse

Quand TOUS ces tests passent :

- [ ] 1er paiement Stripe live encaissé
- [ ] CGU V0 publiée (templates customisés)
- [ ] Hero card visible mobile + desktop
- [ ] Chatbot répond aux 6 questions types
- [ ] Refus pédagogique scripté fonctionne (30+ patterns)
- [ ] Architecture progressive uniforme opérationnelle
- [ ] Track-record Telegram public ≥ 30 trades
- [ ] Plausible self-hosted opérationnel
- [ ] Geo-block strict FR+BE+CH+LU
- [ ] Cap 50 abonnés enforce
- [ ] Aucun vocabulaire interdit dans le produit (audit final)
- [ ] Coverage backend zones revenue ≥ 85 %, autres ≥ 70 %
