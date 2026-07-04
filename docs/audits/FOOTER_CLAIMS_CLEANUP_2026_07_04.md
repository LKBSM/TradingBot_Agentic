# Nettoyage footer & claims légaux/marketing — 2026-07-04

Branche : `fix/footer-claims-cleanup` (depuis main `1526049`).
Règle de mission : le site n'affirme QUE ce qui est vrai et vérifiable
aujourd'hui. En cas de doute sur un claim → suppression, pas reformulation.

## Architecture

Un **seul** footer existe : `webapp/components/Footer.tsx`, rendu une fois dans
`webapp/app/[locale]/layout.tsx` → partagé par toutes les pages. Aucune copie
divergente, rien à unifier.

## Tableau claims avant/après

### Règlement fantôme « UE 2024/2811 » (référence introuvable)

| Localisation | Avant | Après |
|---|---|---|
| `Footer.tsx` | « Conformité UE 2024/2811 par construction. » | Supprimé (le reste du paragraphe posture éducative conservé) |
| `[locale]/layout.tsx` (metadata + OpenGraph) | « …posture éducative, conforme UE 2024/2811. » ×2 | « …posture éducative. » |
| `opengraph-image.tsx` | « …posture éducative · conforme UE 2024/2811 » | « …posture éducative » |
| `FaqSection.tsx` q1 | « (cf. règlement UE 2024/2811 et MiFID II sur les finfluencers) » | Supprimé — le refus des questions « dois-je acheter ? » reste affirmé |
| `FaqSection.tsx` q3 | « Le prompt système inclut les guard-rails compliance UE 2024/2811. » | « Le prompt système lui interdit toute recommandation personnalisée. » |
| `ConversationReplaySection.tsx` | kicker « Compliance UE 2024/2811 » + « prompt système compliance UE 2024/2811 » | kicker « Refus pédagogique » + « prompt système qui interdit toute recommandation personnalisée » |
| `PricingSection.tsx` | « …recommandations personnalisées (UE 2024/2811). » | « …recommandations personnalisées. » |
| `messages/{fr,en,es,de}.json` `disclaimer.long` | « …Conformément à UE 2024/2811 et à la régulation finfluencer mars 2026. » | Phrase supprimée dans les 4 locales (la « régulation finfluencer mars 2026 » est également invérifiable) |
| ~10 commentaires de code (DisclaimerStub, ChatPanel, AppChatSidebar, HonestConfidenceSection, specs…) | citaient UE 2024/2811 / MiFID II 03/2026 | Nettoyés |

### Médiateur, géographie, compteurs

| Localisation | Avant | Après |
|---|---|---|
| `Footer.tsx` | Lien « Médiateur (CM2C) » → `#mediateur` (ancre inexistante, aucune adhésion) | Supprimé |
| `Footer.tsx` | Bloc « Disponibilité géographique · Phase 1 » : 9 pays, « Canada (hors Québec) », « États-Unis et Québec exclus (revue légale Phase 2, indicative M+9) » | Bloc entier supprimé (aucun périmètre annoncé) |
| `FaqSection.tsx` q4 | Liste des 9 pays + « Québec exclu (revue légale en cours) » | « 18 ans minimum + acceptation des conditions + caractère non personnalisé » ; renvoi aux Conditions pour les restrictions juridictionnelles |
| `Footer.tsx` + `PricingSection.tsx` | Badge « Early Access · 50 places » | « Accès anticipé » (vrai) — compteur non décidé supprimé |

### Liens morts

| Lien | Avant | Après |
|---|---|---|
| `/mentions-legales` (footer) | 404 — page inexistante | Retiré (reviendra avec le terminal légal, prompt de service #4) |
| `/cookies` (footer) | 404 — page inexistante | Retiré |
| `#mediateur` (footer) | ancre inexistante | Retiré |
| `/privacy` (FAQ q6) | 404 — la vraie page est `/confidentialite` | Corrigé → `/confidentialite` |
| Ancres produit `#tarifs` etc. (footer) | cassées hors landing (footer global) | Préfixées `/#…` — fonctionnent depuis toutes les pages |
| `/conditions`, `/confidentialite`, `/methodology(#attributions)`, mailto | cibles réelles | Conservés |

### Promesses commerciales invérifiables

| Localisation | Avant | Après |
|---|---|---|
| `FaqSection.tsx` q5 + `PricingSection.tsx` h2 | « Remboursement intégral pendant 30 jours (politique Hamon FR + protection conso UE étendue…) » | Supprimé — aucune politique refund implémentée. « Annulation en un clic » conservé (portail Stripe livré) |
| `FaqSection.tsx` q6 | « exportables au format JSON à tout moment », « effacée sous 30 jours (RGPD art. 17) » | « Vous pouvez demander l'accès, la rectification ou l'effacement de vos données (contact@mia.markets) » — aligné sur la page /confidentialite |
| `PricingSection.tsx` | « Réserver une démo » → `calendly.com/mia-markets/demo` (compte inexistant) | → `mailto:contact@mia.markets?subject=Démo B2B` |
| `PricingSection.tsx` tier gratuit | « 3 questions chatbot par jour » (contredisait le code) | « 5 questions chatbot par jour » — aligné sur `src/api/entitlements.py` (`FREE_CHAT_DAILY_LIMIT=5`) ; « 1 lecture XAU M15 par jour » → « Lecture XAU M15 » (aucune limite /jour codée) |

### Conservé (vrai et aligné sur la ligne)

- Disclaimer footer : « Démonstration en accès anticipé · Lecture algorithmique
  éducative · Ne constitue ni un signal de trading, ni un conseil en
  investissement personnalisé · … risque élevé de perte. »
- `DisclaimerStub` + ligne compliance des panneaux de chat (textes rendus).
- Section Honnêteté (citation publique, « Ce que nous ne ferons jamais »).
- Page `/confidentialite` (placeholder structuré, explicitement marqué
  préliminaire — honnête).

## Non-régression automatisée

`webapp/tests/claims-cleanup.test.ts` (vitest, 16 tests) :

1. **Chaînes interdites** : zéro occurrence de « 2024/2811 », « CM2C »,
   « 50 places », « hors Québec » dans `components/`, `app/`, `messages/`,
   `lib/` (sources rendues + i18n).
2. **Liens du footer** : chaque route interne de `LEGAL_LINKS`/`PRODUCT_LINKS`
   (exportés) doit correspondre à une `page.tsx` existante sous
   `app/[locale]/` ; chaque ancre `/#x` doit correspondre à un `id="x"` réel.

E2E `landing.spec.ts` : le test « 9 Phase-1 countries » (qui verrouillait les
claims faux) est remplacé par un test d'absence de ces claims + présence du
disclaimer honnête et des liens réels.

## Vérifications

- `tsc --noEmit` : 0 erreur.
- `vitest run` : 44 fichiers, **407 tests verts** (dont les 16 nouveaux).
- `next build` : vert.
- Backend non modifié → suite Python non impactée.

## ⚠️ Découverte majeure hors périmètre frontend — décision requise

Le diagnostic partait de « aucune géo-restriction n'existe ». C'est faux au
niveau de l'API : **`GeoBlockMiddleware` existe et est câblé**
(`src/api/app.py:405`, `src/api/middleware/geo_block.py`), actif par défaut
(`GEO_BLOCK_DISABLED=0`), et bloque en HTTP 451 via headers CDN :
US, **GB**, pays OFAC (CU/IR/KP/RU/SY/BY) et la région **CA-QC (Québec)**.

Incohérences à trancher (non traitées ici — décision de périmètre légal) :

1. Le middleware **bloque le Québec** alors que l'entreprise est québécoise et
   que la stratégie légale est Loi 25 + LPC québécoises.
2. L'ancien footer listait le **Royaume-Uni comme ouvert** alors que le
   middleware bloque GB — l'incohérence disparaît côté footer (plus de liste),
   mais le middleware garde sa deny-list.
3. La **CGU canonique** (`docs/legal/conditions-utilisation.md` §4, rendue par
   `/conditions`) et les endpoints legacy `/api/v1/terms|privacy`
   (`src/api/routes/legal.py`) déclarent US/Québec/UK indisponibles. Ces textes
   décrivent fidèlement le comportement implémenté du middleware — les
   « corriger » sans décision sur le middleware rendrait le site MOINS honnête
   (comportement non divulgué). Un test (`test_terms_mention_us_qc_uk_block`)
   verrouille d'ailleurs cette divulgation.

**Suites proposées** (chacune = décision + petit chantier dédié) :
- Décider la deny-list réelle (retirer CA-QC ? garder OFAC seulement ?) puis
  aligner middleware + CGU §4 (avec bump de version = re-consentement) + tests.
- Rebrand des endpoints legacy `/api/v1/terms|privacy` (« Smart Sentinel AI »
  → MIA Markets) ou suppression au profit des pages canoniques.
- Le chatbot peut citer le règlement fantôme : prompts système RAG
  (`src/intelligence/rag/prompts.py`, 4 langues « Tu opères sous UE
  2024/2811 ») + entrée fabriquée `ue-2024-2811` (URL eur-lex inventée) dans
  `data/rag/sources_manifest.yaml` → à purger côté moteur.
