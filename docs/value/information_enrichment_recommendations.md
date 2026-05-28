# Smart Sentinel Co-Pilot — Recommandations d'enrichissement P0/P1/P2

**Date** : 2026-05-18
**Auteur** : trio quant senior + PM SaaS B2C + growth strategist
**Source** : `docs/value/client_information_explained.txt` (vérité terrain), `client_relevance_review.md` (Livrable 1), `best_product_concept.md` (Livrable 2), `mockups/v3/best_concept_demo.html` (Livrable 3).
**Contrainte de scope** : **zéro modification du pipeline algo**. Toutes les recos qui suivent sont des **recompositions**, **traductions**, **agrégations à partir de champs existants**, ou des **améliorations UX / prompt engineering**. Aucune nouvelle feature ML, aucun nouveau modèle, aucun nouveau provider de données.

---

## 0. TL;DR pour le décideur

- **24 recommandations** réparties en 3 priorités. Toutes faisables sans toucher au pipeline algorithmique.
  - **P0 — 8 bloqueurs commerciaux** (non-négociables avant tout go-to-market sérieux) : ~80–120h de travail produit/UX/prompt cumulées.
  - **P1 — 9 multiplicateurs de valeur** (haut ROI, 1-2 sprints) : ~140–200h cumulées.
  - **P2 — 7 renforcements moat long terme** (post-launch, defensibility) : ~180–260h cumulées.
- **Effort total estimé** : ~400–580h de travail produit / UX / contenu / prompt — **0h algo**.
- **Logique de séquencement** :
  - P0 livre **un produit présentable** au prospect (hook commercial frontal, jargon compréhensible, chatbot pilier, compliance assumée).
  - P1 livre **un produit qui retient** (pédagogie complète, historique vérifiable, personnalisation, sources).
  - P2 livre **un produit qui se défend** contre les copycats (replay, mode comparaison, app native, voice).
- **ROI commercial attendu** : P0 débloque la conversion >0.5% sur landing FR (vs friction actuelle). P1 augmente la rétention M3 de ~30%. P2 augmente l'ARPU et le LTV via stickiness conversationnelle.

---

## 1. P0 — Les 8 bloqueurs commerciaux non-négociables

Sans ces 8 enrichissements, le produit reste un **dashboard quant techniquement remarquable mais commercialement gaspillé**. Aucun montant de marketing payant ne compensera leur absence.

### P0.1 — Hero card "track record honnête" en façade permanente

**Champ source** : `historical_stats.profit_factor` + `historical_stats.profit_factor_ci95` + `historical_stats.n_observations` + `historical_stats.lookback_window` (block J.* de l'`InsightSignalV2`).

**Reco** : Le hero card de la webapp et de la page landing doit afficher en **premier plan permanent** : *« 329 setups similaires depuis 2019 · 1.30€ gagnés pour 1€ perdu · IC 95% : 1.12–1.49 · walk-forward 7 ans »*. Pas en footer, pas dans un onglet "Stats", pas après un clic. **En façade, en gros, avant tout autre élément.**

**Justification commerciale** : C'est la seule pépite vraiment différenciante (Livrable 1, classification HERO ABSOLU). Aucun concurrent retail ne publie PF+IC bootstrap. L'enterrer = saboter le moat.

**Effort estimé** : 8–12h (composant React/HTML hero + intégration data + responsive + variants pour Telegram/email/landing).

**Risque si non fait** : Le prospect compare Smart Sentinel à LuxAlgo/TradingView sur la base "quel indicateur est le plus joli" → on perd. La pépite n°1 invisible = produit indéfendable au prix premium.

---

### P0.2 — Toggle 3 modes (FOCUS / CO-PILOT / EXPERT) avec préférence persistante

**Champ source** : Aucun nouveau — la même `InsightSignalV2` est projetée en 3 vues.

**Reco** : Implémenter le toggle décrit dans le Livrable 2 (section 2.7) et démontré dans la démo HTML (`mockups/v3/best_concept_demo.html`). La préférence utilisateur est stockée côté serveur (table `user_preferences`) et restaurée à la connexion.

**Justification commerciale** : Résout la **tension fondamentale** entre richesse algo et surcharge cognitive (objection utilisateur du 2026-05-18). Sans le toggle, soit on simplifie et on perd le moat, soit on garde tout et on perd les prospects débutants. Avec le toggle, on garde tout sans rien sacrifier.

**Effort estimé** : 24–32h (3 layouts CSS + state management + persistance préférence + onboarding "découverte du toggle" + responsive mobile FOCUS-first).

**Risque si non fait** : Surcharge cognitive sur mobile et chez débutants → bounce rate landing élevé, churn M1 élevé.

---

### P0.3 — Reformulation systématique du jargon SMC en français grand public

**Champ source** : `narrative_short`, `narrative_long`, labels UI affichant `BOS`, `CHOCH`, `FVG`, `OB`, `retest_armed`, `displacement`, `equilibrium_zone`, etc.

**Reco** : Maintenir un **glossaire de mapping** centralisé (`src/intelligence/locale/glossary_fr.json`, `glossary_en.json`) — utilisé partout dans l'UI :

| Terme technique | Affichage grand public |
|---|---|
| `BOS` (Break of Structure) | « cassure de structure » / « breakout confirmé » |
| `CHOCH` (Change of Character) | « renversement de tendance » |
| `FVG` (Fair Value Gap) | « zone de déséquilibre » / « écart à combler » |
| `OB` (Order Block) | « zone d'absorption institutionnelle » |
| `retest_armed` | « niveau testé puis défendu » / « zone re-validée » |
| `displacement` | « impulsion forte » |
| `equilibrium_zone` | « zone d'équilibre » |
| `liquidity_sweep` | « ratissage de stops » |

L'affichage technique brut reste disponible **uniquement en mode EXPERT**, et chaque terme du grand public a un **tooltip d'explication courte au survol** (1 phrase + analogie + lien vers chatbot "expliquer plus").

**Justification commerciale** : Le wedge identifié (eval_25 : SMC FR-first) ne couvre que ~10% du marché retail FR. Les 90% restants sont **bloqués par le jargon**. Cette traduction multiplie le marché adressable par ~4–6x à effort marginal.

**Effort estimé** : 20–30h (glossaire FR + EN + DE + ES, intégration UI, tooltips, validation native speakers).

**Risque si non fait** : Smart Sentinel reste un produit pour SMC-natives — ARR plafonné, marché adressable trop étroit.

---

### P0.4 — Chatbot Sentinel comme pilier permanent (sidebar desktop / FAB mobile)

**Champ source** : Tout `InsightSignalV2` accessible en contexte LLM, plus prompt système enrichi par scénario (block N).

**Reco** : Le chatbot ne doit **jamais** être caché derrière un onglet ou un menu. Il est **visible en permanence** :
- Desktop : sidebar fixe à droite (largeur ~320px), affichée d'office en CO-PILOT et EXPERT, repliable mais re-déployable d'un clic.
- Mobile : FAB (Floating Action Button) en bas à droite, badge "Sentinel" + indicateur d'activité.
- Mode FOCUS : un seul CTA principal "💬 Demander à Sentinel" sous le hero card, ouvre un overlay chat.

**6 questions suggérées contextuelles** affichées par défaut, adaptées à la lecture en cours (cf. démo HTML section 3).

**Justification commerciale** : **C'est le moat (problème #4 du Livrable 1).** Si le chatbot est caché ou optionnel, il devient une feature ; visible et permanent, il devient l'expérience principale et le différenciateur défendable.

**Effort estimé** : 30–40h (composant sidebar + FAB + state management + 6 prompts contextuels par scénario + animations + responsive).

**Risque si non fait** : Le produit ressemble à "encore un dashboard SMC avec une icône de chat" → commoditisation, churn, le moat disparaît.

---

### P0.5 — Footer compliance permanent, lisible, assumé

**Champ source** : `compliance.disclaimer_short`, `compliance.is_paper_demo`, `compliance.jurisdiction_blocked` (block L.*).

**Reco** : Footer permanent une ligne, **lisible** (pas mosquito text), **assumé comme un argument** : *« Démonstration paper-trading · Lecture algorithmique éducative · Ne constitue ni un signal ni un conseil. »*

Variante page méthodologie : *« Nous ne vous disons pas quoi faire. Nous vous donnons les meilleurs outils pour décider. »*

**Justification commerciale** : Conforme UE 2024/2811 finfluencer + posture anti-finfluenceur retournée en avantage différenciation. La compliance bien designée **vend la confiance**, mal designée elle ressemble à du CYA juridique honteux.

**Effort estimé** : 4–6h (composant footer + variant pages + traduction multi-langue).

**Risque si non fait** : Soit risque légal (UE 2024/2811, MiFID II finfluencer), soit perception "site louche qui cache son disclaimer" → bounce.

---

### P0.6 — Bannière event imminent ≤4h avec chronomètre visuel

**Champ source** : `economic_calendar.upcoming_events` + `economic_calendar.next_high_impact_event` + `economic_calendar.minutes_until_blackout` (block H.*).

**Reco** : Si un event high-impact est dans ≤4h (FOMC, NFP, CPI, ECB, BoE…), afficher une **bannière jaune permanente** au-dessus du hero card avec :
- Nom de l'event ("FOMC Minutes")
- Chronomètre live ("dans 2h47", refresh chaque minute)
- Impact attendu sur la volatilité ("+18% vs normale pendant 4h post-publication")
- Statut blackout ("Blackout actif si ≤30 min")

**Justification commerciale** : Pépite de soutien identifiée Livrable 1 (5.0/5 sur les 3 axes). Aucun concurrent ne fait ce niveau d'intégration calendrier macro. Argument de valeur **immédiat et visuellement frappant**.

**Effort estimé** : 12–16h (composant bannière + chronomètre live + calcul impact + responsive + traduction events).

**Risque si non fait** : Le client se fait surprendre par un FOMC → perd de l'argent → blâme Sentinel → churn + bad review.

---

### P0.7 — Définitions au survol (tooltips) sur chaque terme jargon résiduel

**Champ source** : Glossaire P0.3 + chatbot context.

**Reco** : Pour chaque terme qui reste technique malgré la traduction (ex: "intervalle conformel", "walk-forward", "profit factor", "conviction calibrée"), tooltip au survol/clic :
- 1 ligne de définition simple
- 1 analogie concrète
- Lien "En savoir plus avec Sentinel" → ouvre le chatbot avec cette question pré-remplie

Exemple sur "Profit factor 1.30" : *« Profit factor = total des gains ÷ total des pertes. Si je perds 1€ à chaque fois que je perds, je gagne en moyenne 1.30€ à chaque fois que je gagne. → Demander à Sentinel : c'est quoi un bon profit factor ? »*

**Justification commerciale** : Réduit la friction cognitive (problème #1 et #2 du Livrable 1) sans amputer la richesse. Le power-user ignore les tooltips, le débutant les utilise.

**Effort estimé** : 10–14h (composant tooltip réutilisable + 25–30 définitions + analogies + intégration chatbot).

**Risque si non fait** : Le débutant rebondit sur le premier terme incompris → bounce.

---

### P0.8 — Refus pédagogique chatbot incarné pour "Donc je dois acheter ?"

**Champ source** : Prompt système chatbot (`src/intelligence/llm_narrative_engine.py`, lecture seule) + post-processing `contains_forbidden_token` existant.

**Reco** : Implémenter (au niveau prompt + post-processing, **pas algo**) une réponse pédagogique **explicite et soignée** au pattern "dois-je acheter / vendre / ouvrir une position" :

> « Je ne peux pas vous dire d'acheter, et je ne le ferai pas — par règle et par éthique. Ce que je peux faire, c'est résumer ce que vous savez : [pour / contre / à considérer]. La décision et le risque vous appartiennent. »

Cette réponse est **scriptée comme un atout commercial**, pas comme un cul-de-sac. Elle apparaît dans les conversations démo, sur les screenshots marketing, et c'est un argument de vente face à LuxAlgo / Telegram signals.

**Justification commerciale** : Compliance assumée incarnée = différenciation anti-finfluencer. C'est ce qui vous protège juridiquement ET ce qui rassure le client sérieux.

**Effort estimé** : 6–10h (prompt engineering + 10 patterns détecteur + 5 variantes réponse + tests adversariaux).

**Risque si non fait** : Risque légal UE 2024/2811 + le chatbot peut être manipulé par prompt injection → désastre PR.

---

## 2. P1 — Les 9 multiplicateurs de valeur (haut ROI, 1-2 sprints)

Ces enrichissements **scalent** la valeur perçue et la rétention. P0 livre un produit présentable ; P1 livre un produit qui retient.

### P1.1 — Waterfall pédagogique 8 composantes avec hover explicatif

**Champ source** : `signal_breakdown.components` (block I.*) — 8 contributions chiffrées avec poids.

**Reco** : Mode EXPERT : waterfall horizontal des 8 composantes avec :
- Contribution chiffrée (+28.0, +14.0, etc.)
- Poids relatif (25%, 15%, 12%…)
- Bar visuelle proportionnelle
- **Au survol** : 2 phrases expliquant ce que mesure cette composante en grand public ("Smart Money : détecte les cassures de structure et zones de déséquilibre. Ici, la cassure 2391.5 est confirmée et la zone 2378–2381 est armée.")

**Justification commerciale** : Résout le problème "blackbox" anti-finfluencer. Le client voit que 8 facteurs sont pesés, comprend lequel pèse, et peut interroger Sentinel pour le détail.

**Effort estimé** : 18–24h (composant waterfall + 8 explications grand public × 4 langues + intégration tooltip).

---

### P1.2 — Conformal interval visualisé (bande + point + ticks)

**Champ source** : `uncertainty.point_estimate`, `uncertainty.conformal_lower`, `uncertainty.conformal_upper`, `uncertainty.coverage_alpha`, `uncertainty.coverage_observed` (block D.*).

**Reco** : Mode EXPERT : visualisation graphique de l'intervalle conformel sur une échelle 0-100. Bande colorée pour l'intervalle, point gold pour le point estimé, ticks pour 0/25/50/lower/point/upper/100. Légende : « Couverture nominale 95% · couverture observée OOS 94.3% ».

**Justification commerciale** : Transforme un chiffre abstrait ("[54, 82]") en une vue intuitive. Différenciation technique majeure — aucun concurrent retail ne visualise sa marge d'erreur calibrée.

**Effort estimé** : 14–18h (composant viz + responsive + animation + tooltips éducatifs).

---

### P1.3 — Stats J.* enrichies pédagogiquement traduites

**Champ source** : `historical_stats.win_rate`, `historical_stats.drawdown_max`, `historical_stats.exposure_time_avg`, `historical_stats.pnl_distribution_skew`, `historical_stats.pnl_distribution_kurtosis`, etc.

**Reco** : Mode EXPERT : grille de 6–8 statistiques avec :
- Valeur chiffrée
- **Label grand public** ("Win rate" → « Taux de réussite », "Drawdown max" → « Pire perte historique », "Skew P&L" → « Asymétrie des résultats »)
- **Tooltip explicatif** ("Win rate 41.6% = sur 100 setups similaires, 42 ont touché le take profit avant le stop. Les 58 autres ont touché le stop. Mais le ratio gain moyen / perte moyenne fait que le profit factor reste positif à 1.30.")

**Justification commerciale** : Transforme des chiffres techniques en pédagogie statistique. Augmente la confiance utilisateur et la sophistication perçue.

**Effort estimé** : 16–22h (composant stats + 8–10 explications grand public + 4 langues).

---

### P1.4 — Tracker live "lecture en cours" avec chronomètre validité

**Champ source** : `identity.created_at` + `identity.expires_at` + `identity.validity_remaining_minutes` (block A.*).

**Reco** : Affichage permanent dans le header du signal : *« Lecture publiée à 14:32 · valable encore 2h47 »* avec chronomètre live (refresh 30s). À 30 min de l'expiration : couleur orange + notification chatbot ("La lecture expire dans 30 minutes — voulez-vous que je vous résume l'état actuel ?"). À expiration : lecture marquée "archivée", overlay grisé, lien vers la nouvelle lecture.

**Justification commerciale** : Crée un sens d'urgence honnête (pas FOMO artificiel). Le client comprend que la lecture a une durée de vie, pas une éternité. Argument de fraîcheur.

**Effort estimé** : 12–16h (composant + state + notifications + archives).

---

### P1.5 — Historique des 50 dernières lectures avec PnL paper

**Champ source** : `historical_stats` agrégé sur la table `signal_store` (toutes les lectures publiées depuis 6 mois).

**Reco** : Page "Mes lectures" accessible depuis le menu utilisateur. Tableau chronologique avec :
- Date + heure + actif
- Direction + label
- Conviction calibrée
- PnL paper si lecture clôturée (TP touché / SL touché / expirée)
- Statut compliance ("paper-trading")
- Lien vers le signal complet

**Cumul mensuel** affiché en haut : "Sur les 30 dernières lectures : 13 TP, 10 SL, 7 expirées. PnL paper cumulé : +12.4R."

**Justification commerciale** : Transparence vérifiable individuelle. Le client peut **lui-même** mesurer la performance, pas seulement croire les stats agrégées. Argument de confiance massif.

**Effort estimé** : 22–30h (page + agrégation backend + paginated UI + filtres + export + paper-trading audit trail).

---

### P1.6 — Personnalisation par utilisateur (mode défaut, langue, alertes, watchlist)

**Champ source** : Nouvelle table `user_preferences` (sans toucher algo).

**Reco** : Page paramètres avec :
- Mode par défaut (FOCUS / CO-PILOT / EXPERT)
- Langue (FR / EN / DE / ES)
- Watchlist actifs (selon tier)
- Préférences notifications (Telegram / email / webhook, fréquence, seuils)
- Heures de silence (pas d'alerte 22h-7h heure locale)
- Préférence post-event ("résume-moi par chat après chaque FOMC")

**Justification commerciale** : Personnalisation = rétention. Sans elle, le produit est "one-size-fits-all" et le power-user comme le débutant sont insatisfaits.

**Effort estimé** : 24–32h (page + persistance + integration partout dans l'UI + Telegram bot adaptation).

---

### P1.7 — Sources RAG cliquables (citations académiques inline)

**Champ source** : Phase 2B RAG `narrative_sources.references` (block K.4) — déjà prévu en pipeline.

**Reco** : Dans le narratif long (mode EXPERT et chatbot mode Pro), chaque affirmation méthodologique pointe vers sa source académique :
- López de Prado 2018 (CPCV walk-forward)
- Corsi 2009 (HAR-RV volatility)
- Gibbs & Candès 2021 (ACI conformal coverage)
- Barndorff-Nielsen & Shephard 2004 (bipower jump detection)
- Adams & MacKay 2007 (BOCPD regime change)

Chaque citation est cliquable → ouvre une mini-fiche avec :
- Référence complète
- DOI / arXiv ID
- 2 phrases vulgarisées ("Cette méthode permet de…")
- Pourquoi Sentinel l'utilise

**Justification commerciale** : Anti-blackbox absolu. Le client power-user vérifie. Le client débutant est rassuré par la sophistication. Aucun concurrent retail n'a ça.

**Effort estimé** : 20–28h (composant citation + 8–12 mini-fiches + intégration narratif + traduction).

---

### P1.8 — Onboarding 4-step contextuel à la première connexion

**Champ source** : N/A (UX pur).

**Reco** : 4 étapes interactives la première fois qu'un utilisateur arrive sur le dashboard :
1. **« Voici la lecture du marché »** → spotlight sur le hero card
2. **« 3 modes pour 3 profondeurs »** → spotlight sur le toggle + démo switch
3. **« Sentinel répond à vos questions »** → spotlight sur le chatbot + suggestion d'une 1ère question
4. **« Nous sommes honnêtes sur ce qu'on ne sait pas »** → spotlight sur l'IC + tooltip conformal

Skippable, mais offert une fois par défaut. Tracking analytics complétion vs skip.

**Justification commerciale** : Onboarding qui montre la valeur en <60s = activation rate trial → paid multiplié par 2-3x typiquement (benchmark SaaS).

**Effort estimé** : 16–22h (composant tour + 4 étapes scriptées + tracking + responsive).

---

### P1.9 — Email digest hebdomadaire récap performances + lectures clés

**Champ source** : Agrégation `signal_store` + `historical_stats` sur 7 jours.

**Reco** : Email envoyé chaque dimanche 18h (timezone user) avec :
- Top 3 lectures de la semaine (les plus convictées, ou les plus surprenantes)
- PnL paper agrégé semaine
- 1 événement macro à anticiper la semaine suivante (depuis economic_calendar)
- 1 paragraphe pédagogique Sentinel ("Cette semaine, le régime a basculé en stress 2x — voici ce que ça veut dire")
- Lien direct vers le dashboard

**Justification commerciale** : Touchpoint hebdomadaire = réduction churn massive (benchmark SaaS : +20-30% rétention M3). Sans coût LLM si templates + variables dynamiques.

**Effort estimé** : 18–26h (template email + agrégation + cron + opt-out + tracking ouvertures).

---

## 3. P2 — Les 7 renforcements moat long terme (post-launch)

Ces enrichissements **défendent** le produit contre les copycats et **augmentent l'ARPU** sur le long terme.

### P2.1 — Replay historique chart annoté (M15/H1/H4)

**Champ source** : OHLCV historique + `signal_store` (annotations) + Smart Money events.

**Reco** : Tier STRATEGIST : chart interactif (Lightweight Charts, sans dépendance lourde) avec :
- 3 timeframes synchronisés (M15 / H1 / H4)
- Annotations BOS, FVG, OB sur le chart aux niveaux/temps détectés
- Marqueurs event macro (lignes verticales avec label)
- Lecture passée affichée en overlay avec PnL final
- Rejouable bar par bar (scrubber)

**Effort estimé** : 60–90h (charts + annotations + sync TF + scrubber + tests).

---

### P2.2 — Mode "Comparer 2 actifs" sur les mêmes facteurs

**Champ source** : Multi-asset déjà supporté côté algo (XAU + EUR + indices + crypto).

**Reco** : Tier STRATEGIST : vue côte-à-côte de 2 actifs (ex: XAU vs EUR) avec :
- Hero cards en parallèle
- Comparaison des 8 facteurs côte-à-côte
- Corrélation cross-asset si disponible (block N.*)
- Chatbot mode comparaison ("Pourquoi XAU est plus haussier que EUR aujourd'hui ?")

**Effort estimé** : 30–45h (layout + state + chatbot prompts comparaison).

---

### P2.3 — Tear sheet PDF mensuelle automatique (Pitchbook style)

**Champ source** : Agrégation `signal_store` + `historical_stats` mensuelle.

**Reco** : Premier du mois, génération auto d'un PDF tear sheet pour chaque utilisateur Strategist+ :
- Couverture branded
- Stats mensuelles (PF, Sharpe, DD, exposure time)
- Top 5 lectures du mois
- Régime dominant du mois
- 1 paragraphe Sentinel ("Le mois en perspective")
- Sources méthodologiques (RAG citations)

Envoyé par email, downloadable depuis l'interface, partageable (URL signed).

**Effort estimé** : 40–55h (template Pandoc/LaTeX ou ReportLab + cron + email + URL signed).

---

### P2.4 — Chatbot voice mode (Sentinel lit la lecture)

**Champ source** : `narrative_short` et `narrative_long` + API TTS (Anthropic Voice si dispo, OpenAI TTS, ElevenLabs).

**Reco** : Bouton "écouter cette lecture" sur le hero card et dans le chat. Voix sobre, posée, professionnelle (pas crypto-bro). Idéal pour le client qui consulte en mobilité.

**Effort estimé** : 24–32h (intégration TTS + cache audio + lecteur + accessibilité).

---

### P2.5 — Notifications proactives chat ("Hey, le régime vient de basculer")

**Champ source** : Changements détectés sur `regime_readout.regime_label` ou `regime_readout.bocpd_change_point` (block F.*).

**Reco** : Push proactif côté chatbot (sans spammer) lorsqu'un événement notable arrive entre 2 sessions du client :
- Régime bascule (trend → range, ou trend → stress)
- Event macro de la watchlist à <30 min
- Conviction passe au-dessus du seuil "PREMIUM" de l'utilisateur
- Lecture en cours invalide (SL touché en paper)

Notification in-app + optionnel push mobile.

**Effort estimé** : 30–42h (engine de détection + dedup + delivery channels + opt-in / opt-out fin).

---

### P2.6 — Mode "Apprentissage" : Sentinel pose des questions

**Champ source** : Prompt engineering Sentinel.

**Reco** : Mode optionnel "Apprendre avec Sentinel" : à la place de répondre frontalement, Sentinel pose des questions au user pour vérifier sa compréhension :
- *« Avant de te répondre — peux-tu me dire ce que tu comprends de ce hero card ? »*
- *« Si je te dis 'IC 95%', tu réponds quoi spontanément ? »*
- Quiz hebdomadaire 5 questions (gamification subtile, pas badges puerils).

**Justification** : Engagement profond = rétention M6+. Différenciation pédagogique majeure.

**Effort estimé** : 40–55h (prompt scaffolding + 30 scénarios + dashboard progression + opt-in).

---

### P2.7 — Mobile app native (iOS + Android), FOCUS-first

**Champ source** : Aucun nouveau côté algo, juste API mobile-friendly.

**Reco** : App native React Native ou Flutter, présentant **uniquement le mode FOCUS** sur écran principal, avec :
- Notifications push system natives
- Widget iOS / Android montrant lecture XAU
- Lock screen "Lecture haussière XAU · conviction 72"
- Bouton chatbot rapide
- Lien vers webapp pour CO-PILOT / EXPERT

**Justification** : 60-70% du retail consulte sur mobile. Une vraie app native vs webapp mobile = ARPU +25-40% (benchmark Robinhood, eToro).

**Effort estimé** : 180–260h (app native + 2 plateformes + tests + soumission stores).

---

## 4. Tableau récapitulatif (priorité, effort, levier)

| # | Recommandation | Priorité | Effort | Levier commercial |
|---|---|---|---|---|
| P0.1 | Hero card track record honnête | P0 | 8–12h | Différenciation pépite n°1 |
| P0.2 | Toggle 3 modes | P0 | 24–32h | Résout surcharge cognitive |
| P0.3 | Reformulation jargon SMC | P0 | 20–30h | Marché adressable ×4-6 |
| P0.4 | Chatbot pilier permanent | P0 | 30–40h | Moat #1 |
| P0.5 | Footer compliance assumé | P0 | 4–6h | Conformité + différenciation anti-finfluencer |
| P0.6 | Bannière event ≤4h | P0 | 12–16h | Pépite calendar visible |
| P0.7 | Tooltips définitions | P0 | 10–14h | Friction cognitive |
| P0.8 | Refus pédagogique chatbot | P0 | 6–10h | Compliance incarnée + différenciation |
| **P0 cumul** | | | **114–160h** | **Bloque go-to-market** |
| P1.1 | Waterfall pédagogique | P1 | 18–24h | Anti-blackbox |
| P1.2 | Conformal interval viz | P1 | 14–18h | Sophistication perçue |
| P1.3 | Stats J.* traduites | P1 | 16–22h | Confiance + sophistication |
| P1.4 | Tracker validité live | P1 | 12–16h | Urgence honnête |
| P1.5 | Historique 50 lectures + PnL | P1 | 22–30h | Transparence vérifiable |
| P1.6 | Personnalisation user | P1 | 24–32h | Rétention |
| P1.7 | Sources RAG cliquables | P1 | 20–28h | Anti-blackbox + sophistication |
| P1.8 | Onboarding 4-step | P1 | 16–22h | Activation trial→paid |
| P1.9 | Email digest hebdo | P1 | 18–26h | Rétention M3 +20-30% |
| **P1 cumul** | | | **160–218h** | **Scale valeur perçue** |
| P2.1 | Replay chart annoté | P2 | 60–90h | Moat technique |
| P2.2 | Comparaison 2 actifs | P2 | 30–45h | Upsell Strategist |
| P2.3 | Tear sheet PDF mensuel | P2 | 40–55h | Stickiness + branding |
| P2.4 | Voice mode chatbot | P2 | 24–32h | Mobile / accessibilité |
| P2.5 | Notifications proactives | P2 | 30–42h | Engagement |
| P2.6 | Mode apprentissage | P2 | 40–55h | Rétention M6+ |
| P2.7 | App mobile native | P2 | 180–260h | ARPU +25-40% mobile |
| **P2 cumul** | | | **404–579h** | **Defensibility long terme** |
| **TOTAL** | | | **678–957h** | |

**Lecture du tableau** :
- **P0 livre un produit présentable** en ~3-4 semaines (1 designer + 1 dev front + 1 PM produit/contenu, mi-temps).
- **P1 livre un produit qui retient** en ~5-6 semaines additionnelles.
- **P2 livre un produit qui se défend** sur 3-4 mois supplémentaires.
- **Total ~12-16 semaines** pour un produit commercialement mature, **sans aucune modification du pipeline algo**.

---

## 5. Recommandation de séquencement

**Sprint 1 (semaine 1-2, ~70h)** :
- P0.1 (hero card honnête) + P0.5 (footer compliance) + P0.6 (bannière event) + P0.7 (tooltips) + P0.8 (refus pédagogique chatbot)
- → Produit présentable, pépite n°1 visible, compliance assumée.

**Sprint 2 (semaine 3-4, ~70h)** :
- P0.2 (toggle 3 modes) + P0.3 (reformulation jargon) + P0.4 (chatbot pilier permanent)
- → Produit ergonomique, jargon traduit, moat chatbot construit. **Ici on peut commencer à montrer le produit à des prospects.**

**Sprint 3 (semaine 5-6, ~80h)** :
- P1.1 + P1.2 + P1.3 (pédagogie EXPERT : waterfall + conformal + stats)
- → Mode EXPERT débloqué et défendable, sophistication visible.

**Sprint 4 (semaine 7-8, ~70h)** :
- P1.4 (tracker validité) + P1.5 (historique 50 lectures) + P1.6 (personnalisation)
- → Transparence individuelle + rétention activée.

**Sprint 5 (semaine 9-10, ~70h)** :
- P1.7 (sources RAG cliquables) + P1.8 (onboarding) + P1.9 (email digest)
- → Anti-blackbox + activation + rétention long terme.

**À ce stade (~10 semaines de produit, 0h algo) : produit commercialisable.** Les P2 deviennent des optimisations post-PMF.

**Sprints 6-12 (semaines 11-22)** : P2 selon arbitrages business (replay si Strategist demande, app mobile si web mobile sature, voice si accessibilité demandée).

---

## 6. Ce qui n'est PAS dans cette liste — et pourquoi

Les recos suivantes sont **volontairement écartées**, malgré leur attrait apparent :

- **Promesses de profit / "PF garanti"** → bloqué par `edge_claim = False` (verdict A1 2026-05-01) et par UE 2024/2811.
- **Système de score "easy buy/sell" simplifié** → tue le moat anti-finfluencer et la posture compliance. Concept C rejeté en Livrable 2.
- **Notifications push agressives "OPPORTUNITÉ MAINTENANT"** → finfluencer style, incompatible avec la posture éthique du produit.
- **Communauté / forum / leaderboard** → demande modération coûteuse et risque légal (utilisateurs qui se donnent des conseils). Reportée à phase 3+.
- **Trading automatique / copy-trading** → hors scope absolu, bloque licence dans la plupart des juridictions.
- **Crypto-actifs en P1** → données fiables coûteuses (`eval_08` : 5/6 presets sans CSV), licence zone grise, public crypto incompatible avec posture honnête.
- **Refonte du pipeline algorithmique** → hors scope mission par contrainte explicite. Toutes les améliorations algo sont déléguées à `MISSION_ACK.md` (institutional overhaul).

---

## 7. Conclusion

Le pipeline algorithmique de Smart Sentinel est **techniquement remarquable** (LightGBM-Isotonic-Conformal, ACI Gibbs-Candès, HMM 3-états, BOCPD, HAR-RV, walk-forward 7 ans CPCV). Son problème n'est **pas** la qualité de la donnée produite — c'est la **manière dont cette donnée est présentée, hiérarchisée, traduite et expliquée** au client retail.

**24 enrichissements, 0h d'algo, ~12-16 semaines de produit** suffisent à transformer un dashboard quant prometteur en un produit B2C commercialement défendable, avec un moat chatbot, une posture compliance assumée comme argument, et une architecture multi-vues qui résout la tension richesse-vs-clarté sans rien amputer.

Les 8 P0 sont **non-négociables**. Tout euro de marketing dépensé avant qu'ils ne soient livrés est un euro gaspillé. Les 9 P1 multiplient la valeur perçue et la rétention. Les 7 P2 défendent la position long terme contre les copycats inévitables.

**Le travail à faire est de la traduction et de la hiérarchisation, pas de la création algorithmique.** C'est plus rapide, moins risqué, et c'est ce qui sépare un produit qui vend d'un produit qui impressionne.
