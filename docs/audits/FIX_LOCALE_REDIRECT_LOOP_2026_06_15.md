# Fix — boucle de redirection locale (ERR_TOO_MANY_REDIRECTS)

**Date :** 2026-06-15
**Branche :** `fix/locale-redirect-loop`
**Fichier :** `webapp/middleware.ts` (1 option ajoutée)

## Symptôme
Tout visiteur dont le navigateur annonce `Accept-Language: en` (ou `de`/`es`)
recevait `ERR_TOO_MANY_REDIRECTS` sur `/` et `/<inactive>/*`. Reproduit sous Edge
lors de la vérification visuelle du chart (contourné alors en forçant `fr-FR`).

## Cause racine
`en/de/es` sont dans `SUPPORTED_LOCALES`. La détection de locale next-intl (activée
par défaut) redirige `/` → `/en` pour un navigateur anglophone (locale supportée
mais **inactive**). Le garde du middleware re-strippe `/en` → `/`, que next-intl
re-détecte en `en` → `/en` … → **boucle infinie**.

```
/  --next-intl(detect en)-->  /en  --guard strip-->  /  --next-intl(detect en)-->  /en  …
```

## Correctif
`localeDetection: false` dans `createMiddleware`. V1 est FR-only : `/` doit toujours
servir la locale par défaut, jamais router selon `Accept-Language` vers une locale
dormante. Les visites directes `/en/*` restent redirigées vers FR par le garde
existant (une seule redirection, pas de boucle).

## Vérification (navigateur `Accept-Language: en-US`)
| Requête | Avant | Après |
|---|---|---|
| `/` | boucle ∞ | **200, 0 redirection** |
| `/en/app` | boucle ∞ | **302 → `/app` → 200** (1 redirection) |
| `/en` | boucle ∞ | **302 → `/` → 200** |
| `/app` | 200 | 200, 0 redirection |

`tsc --noEmit` OK · `next build` vert. Aucune autre modif.
