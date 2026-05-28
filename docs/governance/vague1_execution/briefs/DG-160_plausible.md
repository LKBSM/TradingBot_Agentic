# DG-160 — Plausible self-hosted

**Effort** : ~6h · **Sprint** : S3 · **Owner** : code

---

## Objectif

Déployer Plausible Analytics en mode self-hosted sur Fly.io, **sans cookies tiers, CNIL-compatible**, pour rendre observables les conditions de réactivation DEFER (MAU > 200, churn > 20%, engagement chat) en Vague 3.

## Contexte (angle mort plan original)

Sans Plausible + event tracking en V1, **les 10 DEFER deviennent inopérants** car invisibles. Le plan original mettait l'analytique en V2 — corrigé en V1 strict après revue.

## Périmètre

**IN** :
- Déploiement Plausible self-hosted sur même Fly.io org
- Configuration domaine `analytics.mia.markets`
- Snippet `<script defer>` intégré dans tous les pages frontend
- Vérification : pas de cookies tiers posés sur navigateur

**OUT** :
- Cohort analysis dashboard (DG-162 — V2)
- Historique 50 lectures user (DG-163 — V2)
- Email digest hebdo basé sur analytique (DG-176 — V2)

## Dépendances

- DG-022 Fly.io app principale déployée
- Domain `mia.markets` actif avec DNS gérable

## Fichiers à toucher

- `infrastructure/plausible/docker-compose.yml` (à créer)
- `infrastructure/plausible/.env.example` (à créer)
- `frontend/app/layout.tsx` ou équivalent — ajout `<script defer data-domain>`
- `docs/runbooks/plausible_self_hosted.md` (à créer)
- `.env.production` — `PLAUSIBLE_DOMAIN`, `PLAUSIBLE_URL`

## Implémentation

### 1. Déploiement Fly.io de Plausible

```bash
# Dans infrastructure/plausible/
cat > fly.toml <<EOF
app = "mia-analytics"
primary_region = "cdg"

[build]
  image = "ghcr.io/plausible/community-edition:v2.1.0"

[env]
  BASE_URL = "https://analytics.mia.markets"
  DISABLE_REGISTRATION = "invite_only"

[[services]]
  internal_port = 8000
  protocol = "tcp"

  [[services.ports]]
    handlers = ["http"]
    port = 80

  [[services.ports]]
    handlers = ["tls", "http"]
    port = 443
EOF

fly launch --no-deploy
fly secrets set SECRET_KEY_BASE=$(openssl rand -hex 64) \
                DATABASE_URL=postgres://... \
                CLICKHOUSE_DATABASE_URL=http://...
fly deploy
```

### 2. DNS

Pointer `analytics.mia.markets` → Fly.io IP (A record + AAAA record IPv6).

### 3. Snippet front-end (Next.js)

```tsx
// frontend/app/layout.tsx
export default function RootLayout({ children }) {
  return (
    <html lang="fr">
      <head>
        <script
          defer
          data-domain="mia.markets"
          src="https://analytics.mia.markets/js/script.js"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
```

### 4. Variable d'environnement

```bash
# .env.production
PLAUSIBLE_DOMAIN=mia.markets
PLAUSIBLE_URL=https://analytics.mia.markets
```

## Acceptance criteria

- [ ] `analytics.mia.markets` accessible publiquement (auth invite-only)
- [ ] Visite landing depuis browser propre → événement "pageview" visible dans Plausible dashboard dans les 30s
- [ ] DevTools → Application → Cookies → **aucun cookie tiers posé** (vérification CNIL)
- [ ] DevTools → Network → script Plausible chargé en defer, sans bloquer le rendu
- [ ] Dashboard Plausible accessible uniquement avec compte invite (pas de signup public)
- [ ] Backup automatique journalier ClickHouse/Postgres sur Cloudflare R2

## Tests requis

```bash
# Test E2E : visiter landing depuis Playwright headless
# vérifier dans logs Plausible que l'event "pageview" arrive
pytest tests/e2e/test_plausible_tracking.py

# Test cookies absents
pytest tests/e2e/test_no_third_party_cookies.py
```

## Risques / pièges

- ❌ **Mettre Plausible hosted ($9/mo) au lieu de self-hosted** : viole la stratégie bootstrap coût zéro et ajoute un sous-traitant data à mentionner dans Privacy Policy
- ❌ **Oublier `DISABLE_REGISTRATION=invite_only`** : n'importe qui peut créer un compte sur ton instance
- ❌ **Mettre script Plausible non-defer** : bloque le rendu et impacte LCP
- ✅ Backup ClickHouse essentiel sinon perte d'historique à chaque restart
