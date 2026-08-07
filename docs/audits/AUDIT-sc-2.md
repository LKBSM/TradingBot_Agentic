# AUDIT — SC-2 · Scanner conversationnel

Branche : `feat/sc-2-scanner-conversationnel` (worktree dédié, depuis `main` à jour).
Maquette de référence : `docs/design/reference-scanner-conversationnel.html` (poussée sur `main`, commit `e4401f0`).

> **L'idée, et la contrainte qui la rend sûre**
> M.I.A EST UN TRADUCTEUR VERS LA PALETTE FERMÉE. ELLE N'EST JAMAIS JUGE. Elle
> transforme une phrase en conditions qui EXISTENT DÉJÀ dans la palette ; toute
> sortie non conforme est rejetée par le code, avant toute évaluation ; elle ne
> consulte aucune donnée de marché pour traduire ; elle n'ordonne rien, ne classe
> rien, ne recommande rien ; elle affiche toujours ce qu'elle a compris, et rien
> ne se lance sans validation.

---

## 1. Schéma de validation retenu

La traduction n'est **jamais crue sur parole**. Trois barrières successives, du
modèle jusqu'à l'évaluateur :

1. **`tool_choice` forcé + `input_schema` strict** (`src/intelligence/scanner_translator.py::build_tool_schema`).
   L'outil `emit_scanner_translation` impose :
   - `conditions[].type` = énumération **exacte** des 22 clés de la palette (dérivée
     de `ALLOWED_CONDITION_TYPES`, garde par test) ;
   - chaque contrôle (`direction`, `max_bars`, `proximity_pct`, `session`…) avec son
     énumération de domaine ;
   - `refusal` (`ranking` | `prediction` | `recommendation`), `assumptions[]`,
     `untranslatable[].category` (énumération fermée).
   Le modèle est **contraint** d'émettre cette forme.

2. **Re-validation serveur** (`sanitize_translation` / `sanitize_condition`), la
   barrière autoritaire, indépendante du modèle :
   - un `type` hors `ALLOWED_CONDITION_TYPES` → **rejeté**, et **nommé** en fragment
     `unsupported` (jamais avalé silencieusement) ;
   - une valeur hors domaine (`proximity_pct = 0.37`) → **rejetée**, jamais snappée
     vers la voisine (`0.25`) ;
   - un champ inapplicable au type → **retiré** (sans effet sur la recherche ; ce
     n'est pas une substitution de condition) ;
   - un booléen déguisé en entier (`True == 1`) → rejeté ;
   - un `refusal` non nul → court-circuite tout, `conditions = []`.

3. **Modèle `ScanCondition` du endpoint de scan** (`_revalidate_conditions` dans
   `src/api/routes/scanner_translate.py`) : chaque condition doit survivre au
   **même** modèle Pydantic (`extra="forbid"`, `Literal` des 22 types, bornes) que
   `POST /api/conditions-scan`. Filet ultime : même conforme jusqu'ici, une
   condition invalide serait de toute façon **422** par le scan — rien d'invalide
   n'atteint l'évaluateur (verrou analogue à l'`id` des zones du chatbot).

**Contrat de l'endpoint** — `POST /api/scanner/translate`
`{ text: str (1..500), locale: "fr"|"en" }` →
`{ outcome, refusal?, conditions[], assumptions[], untranslatable[] }`
avec `outcome ∈ { translated, partial, none, refused, empty, error }`.
Codes : `422` (corps invalide), `503` (traducteur non câblé), `500` (erreur interne,
non fuitée). Bootstrap : `SCANNER_TRANSLATOR_ENABLED` (ou ON par défaut si
`CHATBOT_ENABLED`), modèle **Haiku 4.5**, protégé par un `CircuitBreaker` dédié
(seuil 3 / 60 s) ; échec LLM → `outcome:"error"` (le champ texte reste utilisable).

**M.I.A ne reçoit aucune donnée de marché** : le prompt ne contient que la phrase
et la langue. Le compte de combinaisons et les résultats viennent, eux, de
l'évaluateur SC-1 **inchangé** (`use-live-combo-count` → `POST /api/conditions-scan`).

---

## 2. Formulations naturelles — reconnues vs non reconnues

### Reconnues (traduites vers la palette)

| Formulation (fr/en) | Condition |
|---|---|
| « Order Block », « OB », « jamais testé / vierge » | `price_in_ob`, `zone_untested` |
| « Fair Value Gap », « FVG » | `price_in_fvg` |
| « zone déjà testée une fois / au plus N fois » | `price_in_tested_zone`, `zone_tested_at_most` |
| « zone formée récemment » | `zone_formed_recent` |
| « proche d'un OB/FVG sans y être » | `price_near_ob`, `price_near_fvg` |
| « tendance haussière / baissière / indéterminée » | `trend_is` |
| « le 1 h d'accord / l'unité supérieure va dans le même sens » | `higher_tf_agrees` |
| « BOS / CHOCH récent » | `bos_recent_confirmed`, `choch_recent_confirmed` |
| « dernier événement = BOS/CHOCH ↑/↓ », « remonte à … » | `last_event_is`, `last_event_age` |
| « poche de liquidité prise / balayée » | `liquidity_swept_recent` |
| « poche cassée (clôture au travers) » | `liquidity_broken_recent` |
| « poche intacte proche du prix » | `price_near_liquidity` |
| « creux / sommets égaux, EQH / EQL » | `equal_levels_present` |
| « phase d'accumulation / tendance / range / expansion » | `market_phase_is` |
| « volatilité faible / normale / élevée » | `volatility_is` |
| « tiers haut / milieu / bas du range » | `price_in_range_third` |
| « session Asie / Londres / New York / chevauchement » | `session_is` |

Toute valeur **déduite d'un mot vague** (« récemment » → `max_bars = 10`, « proche »
→ une distance) est annoncée dans le bloc **« Ce que j'ai supposé »** — jamais
silencieuse.

### Non reconnues — refusées explicitement, **jamais** approchées par une voisine

`indicator` (RSI, MACD, stochastique, moyennes mobiles, Bollinger, ATR),
`volume`, `fibonacci` (retracements/extensions), `candle_pattern` (avalement, doji,
marteau, pin bar), `classic_support_resistance` (S/R horizontaux ≠ zones SMC),
`round_number` (chiffres ronds), `sentiment` (COT, positionnement),
`fundamental` (macro/news comme condition scannable). Chaque fragment est **nommé**
avec sa catégorie et la phrase de l'utilisateur reprise textuellement.

### Refus (jamais une recherche)

Demande de **classement** (« les meilleurs setups », « le marché le plus sûr »),
de **prédiction** (« où va le prix », « va-t-il monter »), ou de **conseil**
(« qu'est-ce que je devrais trader », « dois-je acheter/vendre »). Détection
déterministe (`detect_refusal`, fr+en) **avant** tout appel LLM ; le champ
`refusal` du modèle est un second filet. → État 4, `conditions = []`.

---

## 3. Dictée vocale — navigateurs supportés & vie privée

**API** : Web Speech API native du navigateur (`SpeechRecognition` /
`webkitSpeechRecognition`). Aucun serveur tiers contractualisé, aucun audio ne
transite par notre backend, M.I.A ne reçoit que du **texte**.

| Navigateur | Dictée | Comportement |
|---|---|---|
| Chrome (desktop / Android), Edge | ✅ | bouton micro présent |
| Safari 14.1+ (macOS / iOS) | ✅ (`webkit`) | bouton micro présent |
| **Firefox** | ❌ | **bouton micro absent** (pas de bouton mort) |

Intégration Next.js 15 : hook `'use client'` (`use-speech-dictation.ts`),
feature-detect **au montage** (le bouton est absent au rendu serveur puis apparaît
seulement si le navigateur sait dicter — pas de mismatch d'hydratation). HTTPS
requis (localhost exempté). Langue = locale active (`fr-FR` / `en-US`).

**Dégradations** (toutes couvertes par test) : navigateur non supporté → bouton
absent ; permission refusée → message clair, champ texte pleinement utilisable ;
aucune parole → dit, jamais d'écoute indéfinie ; **timeout d'écoute 12 s**.

**Transcription imparfaite** (« Order Block », « CHOCH », « Fair Value Gap ») :
acceptable ici et seulement ici — l'utilisateur voit le texte **avant** de traduire,
puis les cartes **avant** de lancer (deux filets). La transcription n'est jamais
cachée, jamais traduite automatiquement.

### Texte proposé pour la politique de confidentialité (à faire valider)

> **Dictée vocale.** Le scanner conversationnel propose une saisie à la voix. La
> reconnaissance vocale est effectuée par **votre navigateur** ; sur certains
> navigateurs (par ex. Chrome, Edge), l'audio capté peut être transmis aux serveurs
> de l'éditeur du navigateur pour y être transcrit — il s'agit alors d'un transfert
> à un tiers susceptible de se situer hors du Québec. Cette transcription produit du
> **texte**, que vous relisez et pouvez corriger avant tout envoi. **Aucun
> enregistrement audio n'est conservé**, ni sur votre appareil, ni sur nos serveurs :
> nous ne recevons que le texte que vous choisissez de soumettre. La saisie au
> clavier reste disponible à tout moment ; si votre navigateur ne prend pas en
> charge la dictée, le bouton n'apparaît pas.

Une mention discrète reprend l'essentiel sous le bouton micro (`dictation.privacy`,
fr + en).

---

## 4. Couverture de test (les 11 garde-fous du §5)

| Garde-fou | Où |
|---|---|
| Sortie hors palette rejetée par le code | `tests/test_scanner_translator.py` (`sanitize_*`), `test_scanner_translate_endpoint.py` |
| Classement/prédiction → refus, jamais une recherche | idem + Playwright état 4 (scan **non appelé**) |
| Condition non traduisible **nommée**, jamais remplacée | translator tests + Playwright état 3 |
| Toute valeur supposée dans « ce que j'ai supposé » | translator tests + Playwright état 2 (`assumptions-block`) |
| Les 3 blocs de résultat toujours rendus | Playwright état 5 (réutilise `ScanResults` SC-1) |
| Aucun tri par nombre de conditions | hérité de `ScanResults`/`ComboCard` SC-1 (ordre fixe) |
| 0 condition ≠ tous les marchés | hérité SC-1 (`use-live-combo-count` `idle`, `ScanResults`) |
| État sans résultat ne propose jamais d'assouplir | hérité `ScanResults` SC-1 (comptes isolés, aucune suggestion) |
| Micro absent si dictée non supportée | `use-speech-dictation.test.ts` + Playwright « dictation unsupported » |
| Compte de stratégie périmé jamais « actuel » | hérité `StrategyPanel` SC-1 (badge « à réévaluer ») |
| Vocabulaire interdit absent (2 langues) | `scannerchat-vocab.test.ts` (namespace `scannerChat`) |

**Backend** : 33 tests (`test_scanner_translator.py` 24 + `test_scanner_translate_endpoint.py` 9), verts.
**Front vitest** : translate-client 7, dictation hook 4, vocab guard 2, parité 9 locales — verts.
**Playwright** : 6 états + dictée non supportée + permission refusée, `1280×800` et `390×844`.
**tsc** : vert. **build** : vert.

---

## 5. Écarts avec la maquette

1. **Surface = flux, pas onglets.** La maquette navigue par onglets (démo des 6
   états). Le produit est un **flux** : Décrire → (Complète | Partielle | Refus) →
   Résultats, avec accès latéral « Mes stratégies ». Route dédiée
   `/[locale]/scanner/decrire` — **entrée supplémentaire**, jamais un remplacement
   (la palette complète reste accessible en un clic dans la carte « Ajouter »).
2. **Résultats & stratégies réutilisent SC-1.** L'état 5 rend `ScanResults` et
   l'état 6 rend `StrategyPanel` (déjà livrés) : on **hérite** ainsi des garanties
   SC-1 (3 blocs non masquables, aucun tri, état vide « ce n'est pas une erreur »
   avec comptes isolés sans suggestion d'assouplir, comptes périmés « à réévaluer »,
   mention « non synchronisé » + export/import). L'état vide « aucune combinaison »
   de la maquette (illustré sous « Mes stratégies ») vit en réalité dans les
   **Résultats** (là où un scan peut ne rien réunir).
3. **Micro « bientôt » → réel.** La maquette montre un bouton micro décoratif
   (`title="Dictée vocale (bientôt)"`). Livré **fonctionnel** (Web Speech API) avec
   dégradations et mention Loi 25.
4. **Exemple de refus = prédiction, pas « meilleurs setups ».** L'exemple cliquable
   qui démontre le refus est une **prédiction** (« Dis-moi où va le prix de l'or »)
   plutôt que « les meilleurs setups » : même démonstration (déclenche l'état 4),
   sans employer le vocabulaire interdit (`setup`, `meilleur`) que le garde-fou
   proscrit sur toute la surface.
5. **Chiffres réels.** Le compte de combinaisons est **réel** (évaluateur SC-1),
   jamais les valeurs figées de la maquette. Périmètre = 6 combos (XAUUSD/EURUSD ×
   M15/H1/H4).

### Dette assumée (suivi)

- **i18n** : `fr` et `en` **natifs et complets** (98 clés). Les 7 autres locales
  (`de, es, it, pt, nl, pl, ar`) portent les **mêmes clés** (parité stricte, garde
  `locale-parity`) avec **valeurs en repli EN** — à traduire nativement dans une
  passe suivante (même dette documentée que `home`/`regimePanel`, cf.
  `AUDIT-dette-1.md`).
- **Chemin LLM live** : validé avec le fondateur avant merge (les tests mockent le
  tool-call ; le vrai Haiku n'est pas exercé en CI).
