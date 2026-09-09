# AUDIT LP-2A — Accueil, section « M.I.A Agent » : réduction de la redondance texte/démo

**Branche** : `feat/lp2a-mia-agent-showcase` (worktree dédié `C:\MyPythonProjects\wt-lp-2a`)
**Base** : `origin/main` = `54265ef` (PR #200, MIA-3)
**Date** : 2026-09-08 / 09
**Périmètre** : les 4 cartes de capacité de la section `#mia` de la page d'accueil.

> Discipline d'audit : le HEAD du dépôt principal était **57 commits en retard** (`e0dc69c`,
> branche `docs/preserve-data-1-audit`). Le worktree a donc été créé depuis `origin/main`,
> pas depuis le HEAD local.

---

## 1. Diagnostic — d'où venait l'impression de « trop de texte »

La section disait **trois fois la même chose** : le paragraphe d'intro l'annonce, la démo de
chat la prouve en direct, les 4 cartes la réexpliquaient en prose. Ce n'est pas la longueur
d'une phrase isolée qui pesait, c'est la répétition.

Mesure de la redondance, carte par carte, contre la démo de chat de la même section :

| Carte | Déjà prouvé par la démo | Contenu unique à préserver |
|---|---|---|
| c1 « Elle enseigne le SMC » | `a1` explique un concept sur l'exemple à l'écran → « avec l'exemple sous les yeux » = redite | **la liste des concepts couverts** (OB, FVG, BOS, CHOCH, mitigation, liquidité) |
| c2 « Elle décrit ce qui est là » | `a1` + `a2` donnent niveaux exacts, horodatage, nombre de tests → 1ʳᵉ phrase intégralement redondante | **« une référence inventée est rejetée par le code »** — le mécanisme d'ancrage, jamais montré |
| c3 « Elle pilote ton affichage » | ⚠️ **rien** : cette démo-ci ne contient aucune commande d'affichage (`u2` = « elle a été testée ? ») | **tout** — les 3 exemples de commande sont la seule preuve de pilotage dans cette section |
| c4 « Elle refuse de deviner » | l'encart orange `a3` dit mot pour mot « je ne prédis pas les mouvements de prix… quand acheter ou vendre » → **doublon le plus direct de la section** | **le principe général** : les trois refus + « outil descriptif » |

## 2. Vérification de dépendance (avant toute coupe)

- **Composant** : `webapp/components/landing/lp1/MiaSection.tsx` (`<section id="mia">`), monté par
  `HomeLanding.tsx:80` et par la galerie dev-only `DesignGallery.tsx:226`. Les 4 cartes sont une
  boucle `['c1','c2','c3','c4']` → `styles.capgrid` / `styles.cap` (`lp1.module.css:404-411`).
- **Aucun test ne verrouillait le texte de ces cartes.** `components/__tests__/txt1-copy.test.ts`
  (TXT-1) ne liste que `scanner.*`, `scannerChat.*`, `zones.*`, `calendar.*`, `legal.*` — **pas une
  seule clé de la landing**.
- L'équivalent landing existe mais garde le **vocabulaire, pas le texte** :
  `components/landing/lp1/__tests__/home.test.tsx` interdit sur les 9 locales
  `setup / signal (+ variantes localisées) / opportun / probabilit`, et en FR/EN
  `gagnant, gain, rendement, réussite, meilleur moment, moteur / winner, profit, guaranteed,
  best time, engine`. `tests/claims-cleanup.test.ts` y ajoute `efficace`, `importance élevée`.
- **Clés i18n** `home.miaSection.caps.{c1..c4}.{h,p}` : présentes et non vides dans les **9 locales**
  (`fr, en, de, es, it, pt, nl, pl, ar`). Source unique — aucune duplication en dur ailleurs
  dans le dépôt (vérifié par recherche plein-texte).

## 3. Ce qui a changé

**Les 4 valeurs `caps.<c>.p`, dans les 9 locales. Rien d'autre.**
Les titres `h`, la démo de chat, les 6 questions préfabriquées, `eyebrow`/`title`/`subtitle`/`h3`
et **le paragraphe d'intro `miaSection.p`** sont intacts — le fondateur a explicitement exclu
l'intro du périmètre.

### FR — avant → après

**c1 — Elle enseigne le SMC** (27 → 17 mots)
- *avant* : « Chaque concept — Order Block, Fair Value Gap, BOS, CHOCH, mitigation, liquidité — expliqué **avec l'exemple sous les yeux**, sur ton marché, pas dans un cours théorique. »
- *après* : « Order Block, Fair Value Gap, BOS, CHOCH, mitigation, liquidité — **chaque concept expliqué sur ton propre marché**. »

**c2 — Elle décrit ce qui est là** (28 → 17 mots)
- *avant* : « Niveaux exacts, horodatages, nombre de tests, état de chaque poche de liquidité. **Uniquement des faits détectés** — une référence inventée est rejetée par le code, pas laissée passer. »
- *après* : « Elle ne cite que des niveaux réellement détectés : **une référence inventée est rejetée par le code**. »

**c3 — Elle pilote ton affichage** (28 → 22 mots)
- *avant* : « « Masque les FVG », « isole cette zone », « passe en 4 h ». Tu parles, **le graphique change**. Plus besoin de chercher le bon bouton. »
- *après* : « « Masque les FVG », « isole cette zone », « passe en 4 h » : tu parles, **le graphique change**. »

**c4 — Elle refuse de deviner** (25 → 12 mots)
- *avant* : « Aucune prédiction, aucune indication d'intervention, aucun conseil. **C'est un outil descriptif**, et elle te le dira elle-même si tu lui demandes où va le prix. »
- *après* : « Aucune prédiction, aucune indication d'intervention, aucun conseil : **c'est un outil descriptif**. »

### Les 9 locales — réduction effective (mots, hors balises)

| | fr | en | de | es | it | pt | nl | pl | ar |
|---|---|---|---|---|---|---|---|---|---|
| c1 | 27→17 | 29→17 | 27→17 | 26→17 | 27→16 | 28→17 | 28→17 | 26→17 | 24→16 |
| c2 | 28→17 | 27→16 | 22→13 | 30→13 | 26→12 | 28→12 | 24→14 | 23→11 | 22→11 |
| c3 | 28→22 | 27→20 | 30→23 | 23→15 | 28→21 | 28→20 | 28→21 | 25→20 | 28→20 |
| c4 | 25→12 | 25→12 | 28→14 | 26→12 | 25→12 | 25→12 | 30→15 | 22→12 | 21→11 |

**FR : 108 → 68 mots (−37 %), 8 phrases → 4.** Chaque locale est raccourcie sur les 4 cartes.
Les 7 locales hors FR/EN sont rédigées dans la langue (sens conservé), pas calquées sur le
français raccourci.

### Ligne éditoriale — aucun fait transformé en jugement

Contrôle explicite sur la carte liquidité/zones (c2), la plus exposée au glissement :
elle dit « niveaux **réellement détectés** » et « rejetée par le code » — un fait vérifiable et
un mécanisme, jamais « fiable » ni « solide ». Les trois refus de c4 (prédiction / intervention /
conseil) sont conservés intégralement ; seule la paraphrase mot pour mot de l'encart orange a
été retirée.

## 4. Mise en page — aucune retouche CSS nécessaire

`.capgrid` est une grille `repeat(4, 1fr)` sur **une seule ligne** (2 colonnes < 900 px, 1 colonne
< 520 px), dont les items s'étirent par défaut. Le passage à une phrase ne peut donc pas produire
une ligne dentelée : c'est **asserté** par le test Playwright (mêmes `top` et mêmes hauteurs à
1 px près). Mesuré sur les captures : hauteur de carte 212 px → 190 px, 4/3/3/3 lignes de texte.

## 5. Tests

| Suite | Résultat |
|---|---|
| `components/landing/lp1/__tests__/lp2a-copy.test.ts` (**nouveau garde-fou**) | **8/8** |
| `components/__tests__/txt1-copy.test.ts` | 8/8 |
| `tests/claims-cleanup.test.ts` | 18/18 |
| `components/landing/**` + `components/__tests__/**` | 122/122 |
| **vitest complet** | **1051/1052** — 1 échec pré-existant (voir §6) |
| `npx tsc --noEmit` | 3 erreurs, toutes pré-existantes (`dictation-copy-honesty.test.ts`), **0 nouvelle** |
| `next build` (CI=1) | vert |
| Playwright `lp2a-mia-cards.spec.ts` | **4/4** (1280×800 + 390×844) |
| Playwright `lp1-accueil.spec.ts` + LP-2A, 2 projets | 60 passés, 4 échecs pré-existants (§6) |

### Le nouveau garde-fou (`lp2a-copy.test.ts`) échoue si :
1. une carte regrossit au-delà de sa longueur pré-LP-2A, **dans l'une des 9 locales** ;
2. un mot prédictif ou de jugement est introduit — FR/EN/**ES**, y compris `fiable / solide /
   reliable / solid / cible / target / biais / bias` ;
3. le contenu unique d'une carte disparaît : la liste de périmètre (c1), le mécanisme d'ancrage
   (c2), les 3 exemples de commande (c3), les 3 refus + « outil descriptif » (c4) ;
4. une carte fuit une autre langue (`fr === en`) ;
5. **le paragraphe d'intro est modifié** — hors périmètre, verrouillé sur ses 51 mots.

Le spec Playwright asserte en plus que les 4 cartes tiennent sur une ligne à hauteur égale et
que chaque carte rend **exactement une phrase** (la détection ignore les points d'abréviation
type « 4 Std. »).

## 6. Échecs pré-existants — non causés par LP-2A

1. **`lib/__tests__/markets-guard.test.ts`** → « `lib/market-reading/session.test.ts` → inline
   market array ». Les deux fichiers sont **identiques à `origin/main`** (`git diff origin/main`
   vide) ; le diff LP-2A ne touche que `messages/*.json` et deux fichiers de test neufs.
2. **`lp1-accueil.spec.ts:209`** « nav bar: a visitor gets no App/Zones/Scanner, sees the
   free-trial CTA » (×2 locales ×2 projets). Le test attend un lien « Essayer gratuitement » /
   « Try for free » dans l'entête ; **ce libellé n'existe plus dans `origin/main`**
   (`git grep "Essayer gratuitement" origin/main -- webapp/messages/fr.json` → vide) : l'entête
   servie rend « Créer un compte » / « Se connecter ». Spec périmée sur `main`, sans rapport avec
   les clés modifiées. Son jumeau (`:219`, qui mocke la sonde de session) passe.

## 7. Captures

`docs/audits/lp2a-shots/{before,after}/fr-mia-{desktop,mobile}.png`
— section `#mia` complète, 1280×800 et 390×844, contre le **build de production** (bandeau
cookies pré-décidé pour ne pas masquer la démo).

## 8. Fichiers touchés

```
webapp/messages/{fr,en,de,es,it,pt,nl,pl,ar}.json   4 valeurs chacune (36 lignes)
webapp/components/landing/lp1/__tests__/lp2a-copy.test.ts   (nouveau)
webapp/tests/e2e/lp2a-mia-cards.spec.ts                     (nouveau)
docs/audits/AUDIT-lp-2a-mia-agent.md                        (ce rapport)
docs/audits/lp2a-shots/**                                   (captures avant/après)
```

Aucun composant produit, aucun CSS, aucune route modifiés.

## 9. Reste à faire

**Confirmation visuelle live du fondateur avant merge** (mission §3).
