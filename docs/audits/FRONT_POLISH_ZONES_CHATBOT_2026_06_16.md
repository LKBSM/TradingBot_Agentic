# Polish front — zones bornées + rebrand cosmétique du chatbot (2026-06-16)

Branche : `feat/front-polish-zones-chatbot` (tip incluant `33acc3b` mitigated_at +
`224b0a3` zones bornées). Front uniquement. Aucune modif moteur / détection / schéma /
sécurité / prompts chatbot. `mitigated_at` consommé en lecture seule.

---

## Partie A — Zones (boîtes localisées et bornées) — VÉRIFIÉE, déjà livrée

La Partie A était **déjà implémentée sur cette branche** (commit `224b0a3`). Audit de
conformité au cahier des charges, point par point — **aucune correction nécessaire** :

| Exigence | Statut | Référence |
|---|---|---|
| A. Boîtes localisées ancrées à la bougie de formation, plus aucune bande edge-to-edge | ✅ | `ReadingChart.tsx` `recompute()` : `xStart = xAt(z.createdSec)` |
| B. Bornage via `mitigated_at` (active → bougie courante ; testée → point de mitigation) | ✅ | `endSec = z.tested && z.mitigatedSec !== null ? z.mitigatedSec : lastTime` ; clampé, jamais de projection future |
| C. Hiérarchie active nette / testée estompée (~0.05) | ✅ | `ZONE_ALPHA` : active `{fill:0.12, border:0.45}`, tested `{fill:0.05, border:0.18}` ; label uniquement si `!tested` |
| D. Style Direction 1 (bull `#2F9E78` / bear `#C2693E`, OB `#8B95A7`, FVG `#6E84B0`, bordures pointillées fines, labels muted 10px tabular, grille horizontale) | ✅ | `CANDLE`, `ZONE_RGB`, `ZONE_LABEL`, `border 1px dashed`, `text-[10px] tabular-nums`, `horzLines` only |
| E. Curation ~3-4 actives + ~2-3 testées par récence + proximité, reste masqué | ✅ | `ACTIVE_ZONE_CAP=4`, `TESTED_ZONE_CAP=3`, `curateZones()` (rang proximité + récence) |
| F. Un seul petit label par zone visible (bord gauche), testées sans label | ✅ | `<span class="absolute left-1 top-0 ...">` rendu `{!r.tested && ...}` |

`mitigated_at` confirmé jusqu'au front : typé dans `webapp/types/market-reading.ts`
(OB + FVG), parsé en lecture seule par `buildZoneModels()`. Direction 1 + interaction
pan/zoom (`handleScroll`/`handleScale`, pinch, kinetic) déjà présents (`4799c65`).

Couverture de tests A : `webapp/lib/chart/__tests__/zoneLayout.test.ts` (parsing,
read-only `mitigated_at`, exclusion zones consommées, caps, proximité, récence,
déterminisme) — **verts**.

> Pas de nouveau commit pour A : le code est déjà sur la branche et conforme. Seule la
> Partie B introduit des modifications.

---

## Partie B — Rebrand cosmétique du chatbot

### Libellé : « Sentinel » → « M.I.A Agent » (texte affiché uniquement)

Casse exacte respectée : **« M.I.A Agent »**. Identifiants internes (`askSentinel`),
fonctions, prompts système : **inchangés**. Commentaires de code (JSDoc) : **inchangés**
(non visibles). Marque produit historique « Smart Sentinel AI » dans
`content/articles/` : **hors scope** (nettoyage pré-pivot séparé, cf. `_README.md`).

Surfaces visibles renommées :

- **Panneau chat** `components/chat/ChatPanel.tsx` — titre, « réfléchit… », disclaimer, intro.
- **Sidebar dockée** `components/app/AppChatSidebar.tsx` — `aria-label`, titre, « réfléchit… », disclaimer.
- **Saisie** `components/chat/ChatInput.tsx` — placeholder + `aria-label`.
- **Aperçu hero** `components/landing/HeroChatPreview.tsx` — `aria-label`, nom, intro persona, état « réfléchit ».
- **Replay conversations** `ConversationReplaySection.tsx` (titre H2 + note) + `ConversationReplayCard.tsx`.
- **FAQ** `components/landing/FaqSection.tsx` — question + réponse.
- **CTA carte** `components/market-reading/MarketReadingCard.tsx` — « Demander à M.I.A Agent ».
- **SEO / méta** `app/[locale]/layout.tsx`, `app/[locale]/app/page.tsx`, `components/seo/JsonLd.tsx`, `app/opengraph-image.tsx`, `components/Footer.tsx`.

Cohérence des sélecteurs : `ReadingColumn.tsx` cible le `textarea` par son `aria-label` —
mis à jour pour suivre le nouveau libellé (sinon focus chat cassé). Assertions de tests
alignées (`AppWorkspace`, `responsive`, `market-reading-components`, e2e `chatbot`,
`chatbot-backend-integration`, `landing`).

### Logo / avatar : SVG inline 3 chandeliers (Direction 1)

Nouveau composant présentationnel `components/chat/MiaAgentLogo.tsx` : SVG inline,
3 chandeliers sobres (2 hausse `#2F9E78` encadrant 1 baisse `#C2693E`), corps fins, mèches
discrètes, aucune dépendance de thème. Remplace l'icône lucide `Bot` dans :
header `ChatPanel`, header `AppChatSidebar`, avatar `HeroChatPreview`, et les marqueurs de
tours assistant de `ConversationReplayCard`. Lisible sur `bg-primary` clair (navy foncé)
**et** sombre (quasi-blanc) — vérifié contre les tokens `--primary`. Pas de néon / glow /
dégradé.

---

## Discipline & validation

- Front uniquement. Aucune zone recalculée, aucune logique chatbot ni couche de sécurité touchée. Descriptif, aucune projection future.
- `tsc --noEmit` : ✅. `next build` : ✅ (warning `metadataBase` préexistant, non lié). `vitest run` : ✅ **155/155**.
- `next lint` non configuré dans le projet (setup interactif, pas de config ESLint) — posture CI existante inchangée ; les portes vertes sont tsc + build + tests.
- Commits séparés : Partie A déjà sur la branche (pas de nouveau commit) ; Partie B = un commit front + ce rapport.
- Clair + sombre vérifiés (couleurs explicites, indépendantes du thème).
