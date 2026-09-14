# docs/legal — les textes légaux publiés

Ces fichiers **sont** les documents que lit un client. Ils ne sont ni résumés ni
reformulés par le code : `src/api/routes/legal.py` les sert **tels quels** et se
contente de choisir la langue et d'estampiller la version.

> ⚠️ **Non validés par un juriste.** Rédigés en interne (mission LEG-1,
> 2026-09-13). Révision professionnelle à faire avant montée en volume — voir
> `docs/audits/AUDIT-leg-1.md`.

## Les fichiers

| Fichier | Servi par | Page |
|---|---|---|
| `conditions-utilisation.{fr,en,es}.md` | `GET /api/v1/legal/conditions?lang=` (alias `/api/v1/terms`) | `/conditions` |
| `politique-confidentialite.{fr,en,es}.md` | `GET /api/v1/legal/privacy?lang=` (alias `/api/v1/privacy`) | `/confidentialite` |

**Le français fait foi.** L'anglais et l'espagnol sont des traductions fidèles et
le disent dans leur propre en-tête. Les six autres locales de l'interface (de,
it, pt, nl, pl, ar) reçoivent l'**anglais** : on ne publie un texte légal que
dans une langue dont on peut répondre.

## Modifier un texte

1. Modifier **les trois langues** — une clause présente dans une seule langue est
   le défaut qui compte, et un test le refuse.
2. Mettre à jour la date d'en-tête des six fichiers **et** `LAST_UPDATED` dans
   `src/api/routes/legal.py`. Les deux doivent rester identiques : cette chaîne
   est l'horodatage de consentement écrit dans `account_consents`.
3. Un incrément de version signifie que **les clients devront ré-accepter**. Ce
   n'est pas un détail de forme : ne bumper que pour un vrai changement de fond.
4. `pytest tests/test_legal_endpoints.py` et
   `npx vitest run tests/leg1-legal-copy.test.ts` verrouillent le reste
   (structure identique entre langues, vocabulaire interdit, clauses qui ne
   doivent pas disparaître, prix cohérent avec `config/pricing.json`).

## Ce que le rendu accepte

Le rendu web (`webapp/lib/legal/render-markdown.tsx`) gère titres, listes à
puces, citations, gras/italique et filets — **pas les tableaux**. Une liste à
puces à la place d'un tableau, donc.

Le premier bloc de citation de chaque document est **lu par le client** : il
indique quelle langue fait foi. Ne pas y remettre de note de process interne.
