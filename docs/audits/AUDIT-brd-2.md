# AUDIT BRD-2 — Déploiement du logo M.I.A Markets

**Branche** : `feat/brd-2-logo` (worktree `wt-brd-2`, depuis `origin/main` `e506735`, PR #167).
**Nature** : purement présentationnelle. Aucune logique métier modifiée.
**Statut** : livré, en attente de **confirmation visuelle live** avant merge sur `main`.

---

## 1. Principe directeur tenu — UNE SEULE SOURCE

Avant BRD-2 le logo n'existait pas à un seul endroit : **7 tracés indépendants** pour la
même marque (2 bougies + 5 « M » or), et le composant partagé (`BrandMark`) n'était utilisé
qu'à un seul endroit. C'était la vraie dette.

Après BRD-2 :

- **Le composant unique** : `webapp/components/brand/MiaLogo.tsx` (adapté du `MiaLogo.tsx`
  fourni, coordonnées et couleurs **verbatim**, aucune retouche). Variantes `mark`,
  `horizontal`, `stacked`, `compact` ; tons `auto` (thème) / `color` / `dark` / `mono`.
- **La géométrie** (coordonnées du prisme) vit dans **un seul fichier** :
  `webapp/lib/brand/prism-geometry.ts`, importé par le composant **et** par les 4
  générateurs d'images (favicon, apple-icon, carte sociale, PNG e-mail). Un test échoue si
  un tracé apparaît en double.
- Les fichiers sources bruts sont archivés dans `webapp/public/brand/*.svg` (référence, non
  recopiés dans le code).

---

## 2. Inventaire AVANT → APRÈS par surface

| # | Surface | Avant | Après |
|---|---------|-------|-------|
| 1 | En-tête public (`Nav.tsx`) | `BrandMark` bougies + texte | `MiaLogo variant="horizontal"`, lien accueil, focus clavier |
| 2 | En-tête connecté **desktop** = rail (`ShellRail.tsx`) | **aucune marque** | `MiaLogo variant="horizontal"` en haut du rail, lien accueil |
| 2 | En-tête connecté **mobile** (`ProductShell.tsx`) | **aucune marque** | barre `.shell-mbrand` (<768) `MiaLogo variant="mark"`, lien accueil |
| 2 | `AppHeader.tsx` (chrome legacy) | span « M » or codé main | `MiaLogo variant="mark"` + nom responsive (composant assaini) |
| 2b| Tiroir mobile (`MobileMenu.tsx`) | titre texte seul | `MiaLogo variant="mark"` + nom |
| 3 | Favicon (`app/icon.tsx`) | « M » or 32×32 | prisme **compact** sur tuile sombre |
| 3 | Apple touch (`app/apple-icon.tsx`) | « M » or 180×180 | prisme compact |
| 3 | Manifeste PWA (`public/icon.svg` + `manifest.ts`) | path « M » or + `name:"MIA…"` | prisme compact maskable + `name:"M.I.A Markets"`, `short_name:"M.I.A"` |
| 4 | Aperçu social (`app/opengraph-image.tsx`) | « M » or + « MIA Markets » | prisme (ton fond sombre `#7DA3FF`) + « M.I.A Markets », carte sombre |
| 5 | Connexion (`LoginForm.tsx`) | tuile « M » | prisme **compact** dans la tuile + nom |
| 5 | Inscription / mdp oublié / confirmer / vérifier e-mail | **aucun logo** | `AuthBrandHeader` (verrouillage **empilé**) au-dessus du formulaire |
| 6 | Choix de formule / retour paiement (`abonnement`) | **aucun logo** | `AuthBrandHeader` empilé |
| 7 | Avatar M.I.A (`chat/AgentAvatar.tsx`) | `MiaAgentLogo` bougies, disque or | `MiaLogo variant="compact"` dans disque **teinté marque** (suit le thème) |
| 7 | Aperçus landing (`HeroChatPreview`, `ConversationReplayCard`) | `MiaAgentLogo` | `MiaLogo variant="compact"` |
| 8 | Pied de page (`Footer.tsx`) | texte « MIA Markets » (littéral) | `MiaLogo variant="mark" tone="mono"` discret + `BRAND_NAME` |
| 9 | 404 / erreurs | pas de logo (déjà conforme) | **inchangé** — pas de logo (conforme, testé) |
| 10| E-mails (vérif, reset, renouvellement) | texte brut, aucun logo | alternative **HTML** avec **PNG hébergé** `{APP_PUBLIC_URL}/brand/email-logo.png` + `alt`, texte conservé en repli |
| — | SEO JSON-LD (`JsonLd.tsx`) | `name:"MIA Markets"` | `name:"M.I.A Markets"` + `Organization.logo` |
| — | Nom de marque | « MIA Markets » (sans points) | **« M.I.A Markets »** (avec points) sur toutes les surfaces visibles + 9 locales |

### Choix automatique clair/sombre
Le thème est géré par `next-themes` (`data-design` sur `<html>` : `terminal`/`schema`/`ardoise`
= sombre, `atelier` = clair). Le ton `auto` du logo lit deux variables CSS
(`--brand-mark`, `--brand-word`) définies par thème dans `globals.css` → **#2962FF en clair,
#7DA3FF en sombre**, sans JavaScript, sans décalage de mise en page. Vérifié en Playwright
(couleur résolue du prisme).

### Nom « M.I.A Markets » (renommage inclus)
La règle d'écriture impose « M.I.A Markets » (entreprise), « M.I.A » (agent),
« Multi-asset Intelligence Assistant » (acronyme). Le code écrivait « MIA Markets ». Renommé
sur les **surfaces visibles** : 9 fichiers i18n, `JsonLd`, carte sociale, `package.json`,
e-mails backend, `BRAND_NAME`, titre iOS/PWA. **Non touchés** (hors périmètre logo, non
visibles) : commentaires de code « MIA Markets V2 — Chantier », User-Agent internes, prompt
système du chatbot — pour ne modifier aucune logique.

---

## 3. Fichiers supprimés

- `webapp/components/BrandMark.tsx` (marque bougies dupliquée)
- `webapp/components/chat/MiaAgentLogo.tsx` (avatar bougies)
- Tracés « M » or inline **remplacés** dans `icon.tsx`, `apple-icon.tsx`,
  `opengraph-image.tsx`, `public/icon.svg`, span de `AppHeader.tsx`, tuile de `LoginForm.tsx`.

Aucun ancien fichier de logo « au cas où » : plus aucune référence à `BrandMark` /
`MiaAgentLogo` / au path « M120 384 » / au dégradé or `#FBBF24`/`#B45309` (testé).

---

## 4. Fichiers ajoutés

- `webapp/components/brand/MiaLogo.tsx` — composant unique.
- `webapp/lib/brand/prism-geometry.ts` — géométrie unique.
- `webapp/components/auth/AuthBrandHeader.tsx` — verrouillage empilé des pages auth/formule.
- `webapp/app/brand/email-logo.png/route.tsx` — PNG hébergé stable pour les e-mails.
- `src/api/email_branding.py` — alternative HTML brandée (texte conservé).
- `webapp/public/brand/*.svg` — 7 sources de référence.
- `webapp/components/brand/__tests__/MiaLogo.brand.test.tsx` — 4 gardes.

---

## 5. Tests

- **tsc** : vert.
- **Vitest** : 896 tests verts (suite complète ; les timeouts worker observés sont un artefact
  Windows Defender sur `node_modules` fraîchement installé — les fichiers concernés repassent
  verts à chaud, 42/42).
- **Gardes BRD-2** (10) : tracé unique · variante claire/sombre · plus aucune ancienne
  chaîne « MIA Markets »/dégradé or · logo absent des états chargement/vide/erreur.
- **Pytest** e-mails : renouvellement + auth 22/22, parcours pay3 14/14, imports OK.
- **Playwright** `brd2-logo.spec.ts` + `theme-and-pwa.spec.ts` : **50/50 verts** (desktop
  1280×800 + mobile iPhone-12 390×844). Couvre présence sur accueil/app/scanner/zones/
  actualites/connexion/inscription/abonnement (fr+en), variante claire/sombre (couleur du
  prisme résolue par thème), 404 sans logo, avatar M.I.A dans la conversation.

### Découverte en cours de route
Le shell produit **desktop** (`ShellRail`) et **mobile** (`ProductShell`) n'avaient
**aucune marque** — `AppHeader` (que le diagnostic croyait actif sur /app) n'est pas rendu
par le shell produit. La marque connectée a donc été posée directement sur le rail (desktop)
et via une barre mobile dédiée (couvre toutes les routes produit).

---

## 6. Le logo N'apparaît JAMAIS (règle stricte — vérifié)

Pas de logo en indicateur de chargement, en état vide, en message d'erreur, en filigrane, ni
répété dans un même héros. L'avatar M.I.A dans le chat est l'avatar **de l'auteur** (identité
du locuteur à côté de chaque message), pas une décoration de chargement. `not-found.tsx`,
`error.tsx`, `global-error.tsx`, `ReadingSkeleton.tsx` : aucun logo (testé).

---

## 7. À FAIRE HORS DÉPÔT (toi, hors code)

1. **Tableau de bord Stripe** — remplacer le logo (Checkout, portail client, reçus, e-mails
   Stripe) par le nouveau logo M.I.A Markets.
2. **Images de profil réseaux sociaux** — avatar (prisme compact) + bannière sur X/LinkedIn,
   etc.
3. **Icône du domaine / e-mail** — vérifier que Brevo (ou l'expéditeur SMTP) et l'éventuel
   favicon référencé côté registrar pointent le nouveau logo ; purger tout cache d'ancien
   favicon.
4. **Variables prod** — s'assurer que `APP_PUBLIC_URL` est bien posée (le PNG e-mail en dépend :
   `{APP_PUBLIC_URL}/brand/email-logo.png`). En prod le boot échoue déjà si absente/localhost.
5. **Google / OAuth consent screen** — logo de l'application si présent.

## 8. Points à valider en revue live

- **Carte sociale (OG)** : j'ai gardé la carte riche (titre + accroche conformes) avec le
  prisme + « M.I.A Markets » en haut, plutôt que de la réduire à un logo empilé nu et centré.
  Si tu veux la version « verrouillage empilé centré » stricte, je l'ajuste.
- **Avatar** : disque teinté marque (bleu) au lieu de l'ancien or. À confirmer visuellement.
