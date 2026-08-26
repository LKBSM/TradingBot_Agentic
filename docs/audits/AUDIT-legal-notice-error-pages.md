# AUDIT — Avertissement réglementaire « manquant » (ticket additif TXT-1)

## Position du HEAD

Branche `feat/legal-notice-missing-pages` basée sur `origin/main` (`cccf308`), 0 commit de retard.

## Constat : la prémisse « 0 avertissement » était fausse

Le diagnostic TXT-1 avait signalé « 0 avertissement réglementaire » sur `/compte`,
`/verifier-email`, `/mot-de-passe-oublie` et les pages d'erreur. **Vérification live (captures) :
c'est faux.** Les agents n'avaient inspecté que les composants de page, pas le **chrome** :

| Surface | Avertissement déjà rendu par… | Clé i18n |
|---|---|---|
| Pages **produit** (`/app`, `/compte`, `/scanner`, `/zones`, `/actualites`) | rail `ProductShell` (`ShellRail`) | `nav.legal.disclaimer.chart` — « Lecture algorithmique éducative — ni signal de trading, ni conseil en investissement. » |
| Pages **site** (`/connexion`, `/inscription`, `/abonnement`, `/mot-de-passe-oublie`(+confirmer), `/verifier-email`, `/inscription/google`) | `Footer` (groupe `(site)`) | `footer.disclaimer` — « …Ne constitue ni un signal de trading, ni un conseil en investissement personnalisé · … risque élevé de perte. » |

→ **Chaque page de contenu porte déjà exactement un avertissement.** Ajouter une note par
page y créerait un **doublon** (violation de « jamais plus d'un »). Les 5 ajouts initialement
faits (sur AccountPanel, EmailVerifier, ForgotPasswordForm, ResetPasswordForm,
GoogleFinalizeForm) ont donc été **retirés**, ainsi que le composant partagé et ses tests.

## Le seul vrai manque : les frontières d'erreur

`app/[locale]/error.tsx`, `app/[locale]/not-found.tsx` et `app/global-error.tsx` rendent
**hors** des groupes `(site)`/`(product)` : elles remplacent le sous-arbre en erreur et
n'héritent donc **ni du rail ni du Footer** → aucun avertissement, ce sont les seules pages
réellement à zéro.

**Correctif (le seul justifié)** : une ligne d'avertissement **en dur, auto-contenue** (FR ;
ces pages sont FR-only par conception et doivent rester indépendantes de l'i18n qui peut être
la cause même du crash), reprenant le libellé canonique de `connexion.trust` :

> « Outil éducatif de lecture de marché. Ni signal de trading, ni conseil en investissement. 18+. »

- `error.tsx` — sous les boutons.
- `not-found.tsx` — sous le bouton retour.
- `global-error.tsx` — inline-stylé (boundary sans dépendances).

## Sous-constat non traité (décision fondateur : laisser tel quel)

Les avertissements de chrome (rail + footer) disent « ni signal ni conseil » mais **omettent
« 18 ans et plus »**. Le 18+ est déjà exigé à l'inscription (cases de consentement) et présent
sur `/connexion` (`connexion.trust`). Décision : ne pas modifier le chrome global.

## Tests

- `app/[locale]/error-notice.test.tsx` (2/2) — échoue si l'avertissement disparaît de la 404
  ou de la frontière d'erreur de route. (`global-error.tsx` rend son propre `<html>` → vérifié
  par inspection.)
- `tsc --noEmit` : 0 erreur nouvelle (3 pré-existantes `dictation-copy-honesty`).

## Discipline

Périmètre présentationnel. Staging explicite, pas de force push. Merge après confirmation.
