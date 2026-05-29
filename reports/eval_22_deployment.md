# Eval 22 — Deployment & Infrastructure

**Note globale**: 4.5/10
**Verdict**: Rester sur **Railway** comme primary pour les 90 prochains jours (coût bas, IaC quasi nulle, region EU dispo). Préparer **Fly.io** comme backup/migration cible (machines dédiées + volumes + multi-region). **Bloquant immédiat** : `Procfile` et `railway.toml` lancent encore `parallel_training.py` (ancien pipeline RL) et non `python -m src.intelligence.main` — ce qui veut dire que la production déployée actuellement n'est PAS le SaaS Smart Sentinel AI documenté. À corriger avant toute autre action.

---

## 1. As-is inventory

Source : lecture directe des fichiers du repo.

| Élément | Valeur observée | Fichier:ligne |
|---|---|---|
| Image base | `python:3.11-slim` (builder + runtime) | `infrastructure/Dockerfile:16,41` |
| Multi-stage | Oui — `builder` → `production` → `development` | `Dockerfile:14,39,94` |
| Entry point Docker | `python -m src.intelligence.main` | `Dockerfile:89` |
| Port exposé Docker | `8000` | `Dockerfile:86` |
| User non-root | `sentinel:sentinel` (uid implicite) | `Dockerfile:55-56,78` |
| Healthcheck Docker | `curl -f http://localhost:8000/health` toutes les 30 s | `Dockerfile:81-82` |
| Entry point Procfile | `python parallel_training.py` ⚠️ | `Procfile:1` |
| Entry point Railway | `python parallel_training.py` ⚠️ | `railway.toml:5` |
| Builder Railway | `nixpacks` (donc Dockerfile **ignoré**) | `railway.toml:2` |
| Restart policy Railway | `never` ⚠️ | `railway.toml:6` |
| Volume Railway | `results-volume` → `/app/persistent_results` | `railway.toml:8-10` |
| `.dockerignore` | **absent** (vérifié `ls`) | — |
| `.github/workflows/` | **absent** | — |
| docker-compose | Stack complète locale: bot+redis+postgres+prometheus+grafana+alertmanager, ports `127.0.0.1` only, mots de passe `${VAR:?}` requis | `infrastructure/docker-compose.yml` |
| Port docker-compose | `8080` (différent du Dockerfile `8000`) ⚠️ | `docker-compose.yml:35` |
| Healthcheck docker-compose | `http://localhost:8080/health` | `docker-compose.yml:56` |
| Persistance signaux | SQLite `./data/signals.db` | `src/intelligence/main.py:366` |
| Logging structuré | `JSONFormatter` activé via `LOG_FORMAT=json` | `src/intelligence/main.py:39-52,61` |

### Incohérences critiques détectées (bloquantes)

1. **Procfile / railway.toml lancent l'ancien `parallel_training.py`** (RL training). Le SaaS Sentinel n'est probablement pas en prod actuellement, OU il l'est via un override manuel non versionné.
2. **railway.toml utilise `nixpacks`** : le Dockerfile soigneusement multi-stage **n'est pas utilisé** par Railway.
3. **Restart policy = `never`** : un crash = downtime jusqu'à redéploiement manuel. Pour un scanner long-running, mettre `on-failure` minimum.
4. **Port mismatch** : Dockerfile `8000`, docker-compose `8080`. Source de pannes en stage.
5. **TESTING_MODE=1 par défaut** (`MEMORY.md` + `main.py:17`) → si `SENTINEL_TESTING_MODE=0` n'est pas explicitement set en prod Railway, **l'auth est bypassée silencieusement**. Pas de fail-closed.

### SLA implicites à préserver
- Latence signal < 30 s après close de bougie M15 (ext. SaaS).
- Persistance SQLite des signaux (perte = perte du système d'alerting).
- Connectivité sortante: api.anthropic.com, api.telegram.org, NewsAPI, MT5 broker.
- Single worker (le scanner n'est pas concurrent-safe — confirmé par l'absence de lock dans `main.py`).

---

## 2. Dockerfile audit

| Critère | État | Note | Commentaire |
|---|---|---|---|
| Multi-stage | Oui | + | builder/production/development |
| Base image | `python:3.11-slim` | + | Python 3.11 EOL Oct 2027, OK ; scanner avec `trivy image python:3.11-slim` recommandé mensuellement |
| Layer caching | Bon | + | `requirements.txt` copié avant le code (`Dockerfile:28`) |
| Image size estimée | ~900 MB – 1.2 GB | − | torch CPU + transformers + lightgbm + arch ; `requirements.txt:10-12, 27, 41` |
| Healthcheck | Oui | + | `Dockerfile:81-82` |
| Non-root user | Oui | + | `Dockerfile:78` |
| `.dockerignore` | **Absent** | −− | Risque : `data/*.csv`, `replay_*.json`, `models/`, `.env`, `__pycache__` envoyés au build context. Build lent + risque de leak `.env` dans une layer |
| Build args secrets | Aucun | + | Pas de `ARG ANTHROPIC_API_KEY` (bonne pratique) |
| Pinning versions | Mixte | − | `requirements.txt` mélange `==` (torch, sb3) et `>=` (anthropic, fastapi) → builds non reproductibles |
| Stage `development` | Présent dans la même image | − | Risque : si quelqu'un build `--target development`, `pytest`/`mypy` en prod |
| `apt-get` cleanup | Oui | + | `rm -rf /var/lib/apt/lists/*` |
| CVE scanning | Non versionné | − | Recommander `trivy image --severity HIGH,CRITICAL smart-sentinel:latest` en CI |
| Reproducibilité | Faible | − | Pas de `pip-tools`/`uv lock`, pas de hash pinning |

### Diff suggéré (extrait minimal)

Créer `.dockerignore` à la racine :

```
.git
.github
.venv
venv
__pycache__
**/__pycache__
*.pyc
.env
.env.*
.pytest_cache
.mypy_cache
.ruff_cache
data/*.csv
data/*.parquet
data/*.db
data/*.h5
models/
trained_models/
logs/
*.log
results/
reports/
mockups/
notebooks/
replay_*.json
replay_*.csv
backtest_*.csv
*.zip
*.pkl
*.joblib
README.md
*.md
!requirements.txt
Script*collab*/
infrastructure/grafana/
```

Patch Dockerfile (ne build plus le stage `development` en prod, ajout d'un `pip install --require-hashes` si requirements compilées) :

```dockerfile
# Builder stage — inchangé jusqu'à pip install
RUN pip install --no-cache-dir --upgrade pip==24.0 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip check

# Production stage — supprimer curl si on remplace le HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Healthcheck via Python (évite la dépendance curl)
HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0) if urllib.request.urlopen('http://localhost:8000/health',timeout=5).status==200 else sys.exit(1)"

# Tag explicite pour éviter de pull prod accidentellement comme dev
# Build prod uniquement: docker build --target production -t smart-sentinel:1.0.0 .
```

Et `parallel_training.py` ne doit plus être l'entry point Railway (cf. §9).

---

## 3. Comparatif providers (4 × 6 critères)

Spec cible : **2 vCPU, 4 GiB RAM, 1 worker, EU region, persistent disk 5–10 GB pour `signals.db`**.
Pricing à **vérifier 2026** (les pages tarifaires bougent ; chiffres ci-dessous = ordres de grandeur publiés mi-2025/début 2026).

| Critère | Railway (baseline) | Fly.io | Render | AWS ECS Fargate |
|---|---|---|---|---|
| **$/mois (spec)** | ~$5 starter + $20–40 usage = **$25–45** *(vérifier 2026)* | shared-cpu-2x 4GB ~$30 + 5GB volume ~$0.75 + bandwidth = **~$32–40** *(vérifier 2026)* | Standard 2 vCPU/4GB **$25** ; Pro **$85** *(vérifier 2026)* | Fargate 2vCPU/4GB ~24/7 ≈ **$60–75** + ALB **$18** + EBS = **~$90** *(vérifier 2026)* |
| **Cold-start** | ~3–8 s (scale-to-zero off par défaut) | <1 s sur Machine "always on", 5–10 s si auto-stop | 0 s (Standard) | 30–60 s première tâche (Fargate provisioning) |
| **IaC ease** | `railway.toml` + CLI ; pas de Terraform officiel mature | `fly.toml` simple, provider Terraform officiel | `render.yaml` (Blueprint) | Terraform/CDK natif, mais verbeux (~200 lignes pour égaler) |
| **Régions** | US-east, US-west, EU-west (Amsterdam), Asia-southeast | 35+ régions (`cdg`, `fra`, `ams`, `iad`, `nrt`, …) | US, EU (Frankfurt), Singapore | 30+ régions, `eu-west-3` (Paris), `eu-central-1` |
| **SQLite persistance** | Volume Railway natif (déjà utilisé `railway.toml:8-10`) | Fly Volumes (NVMe local), 1 volume = 1 machine pinning | Disk add-on Standard+ (≥$1/GB/mo) | EFS ($0.30/GB/mo) ou EBS attaché ; Fargate + EBS récent |
| **Lock-in** | Faible (Dockerfile portable, mais `nixpacks` actuel = couplé) | Faible (Dockerfile + fly.toml) | Moyen (Blueprint propriétaire) | Élevé (ALB + IAM + VPC + Fargate task def) |

### Recommandation

- **Primary** : **Railway** — coût et simplicité imbattables pour solo founder en phase test. Prérequis : passer en **Dockerfile** (pas nixpacks) et corriger `startCommand`.
- **Backup / migration cible quand >100 users payants** : **Fly.io** — meilleure granularité region (latence LLM EU↔Anthropic optimisable), `fly.toml` clean, machine `auto_stop_machines = "off"` pour le scanner long-running.
- **À éviter pour ce stade** : Fargate (overkill, $90+ baseline) ; Render Standard est OK mais disk plus cher au GB.

---

## 4. CI/CD pipeline (YAML)

Aucun workflow GitHub Actions n'existe (`ls .github/workflows` → vide). Skeleton à placer dans `.github/workflows/ci-cd.yml` :

```yaml
name: ci-cd

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  IMAGE_NAME: smart-sentinel
  PYTHON_VERSION: "3.11"

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - run: pip install ruff==0.4.10
      - run: ruff check src/ tests/

  test:
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: pip
      - run: pip install -r requirements.txt
      - name: Run pytest (fail fast)
        env:
          SENTINEL_TESTING_MODE: "1"
        run: pytest -x --maxfail=3 --tb=short -q tests/

  build:
    runs-on: ubuntu-latest
    needs: test
    permissions:
      packages: write
      contents: read
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - name: Build image
        uses: docker/build-push-action@v6
        with:
          context: .
          file: infrastructure/Dockerfile
          target: production
          push: ${{ github.event_name == 'push' }}
          tags: |
            ghcr.io/${{ github.repository_owner }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
            ghcr.io/${{ github.repository_owner }}/${{ env.IMAGE_NAME }}:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max

  smoke-image:
    runs-on: ubuntu-latest
    needs: build
    if: github.event_name == 'push'
    steps:
      - uses: actions/checkout@v4
      - name: Pull image
        run: docker pull ghcr.io/${{ github.repository_owner }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
      - name: Run container
        run: |
          docker run -d --name sentinel \
            -e SENTINEL_TESTING_MODE=1 \
            -e ANTHROPIC_API_KEY=dummy-for-smoke \
            -e NARRATIVE_MODE=template \
            -p 8000:8000 \
            ghcr.io/${{ github.repository_owner }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
      - name: Wait & probe /health
        run: |
          for i in $(seq 1 20); do
            if curl -fs http://localhost:8000/health > /dev/null; then
              echo "OK"; exit 0
            fi
            sleep 3
          done
          docker logs sentinel
          exit 1

  deploy-staging:
    runs-on: ubuntu-latest
    needs: smoke-image
    if: github.ref == 'refs/heads/main'
    environment: staging
    steps:
      - uses: actions/checkout@v4
      - uses: superfly/flyctl-actions/setup-flyctl@master   # or railway up
      - run: flyctl deploy --remote-only --config fly.staging.toml
        env:
          FLY_API_TOKEN: ${{ secrets.FLY_API_TOKEN_STAGING }}

  smoke-staging:
    runs-on: ubuntu-latest
    needs: deploy-staging
    steps:
      - name: Probe staging /health
        run: |
          for i in $(seq 1 20); do
            curl -fs https://sentinel-staging.fly.dev/health && exit 0
            sleep 5
          done
          exit 1

  deploy-prod:
    runs-on: ubuntu-latest
    needs: smoke-staging
    environment: production       # GitHub-protected, manual approval
    steps:
      - uses: actions/checkout@v4
      - uses: superfly/flyctl-actions/setup-flyctl@master
      - run: flyctl deploy --remote-only --config fly.toml
        env:
          FLY_API_TOKEN: ${{ secrets.FLY_API_TOKEN_PROD }}
      - name: Post-deploy smoke
        run: |
          sleep 15
          curl -f https://sentinel.fly.dev/health
```

### Rollback recipes

- **Railway** :
  ```
  railway status                           # list deployments
  railway redeploy --deployment <id>       # roll back to known-good
  ```
- **Fly.io** :
  ```
  flyctl releases --app sentinel
  flyctl releases rollback <version> --app sentinel
  ```
- **Image-level (universel)** : tagger chaque release `ghcr.io/.../smart-sentinel:<sha>` et redéployer le SHA précédent — déjà couvert par le workflow.

### Staging/prod separation

- Aujourd'hui : **single env** (constaté ; pas de fichier staging dans le repo).
- Cible : `fly.staging.toml` (1 machine shared-cpu-1x, scale-to-zero on, faux ANTHROPIC_API_KEY, NARRATIVE_MODE=template) ↔ `fly.toml` prod. Branch protection sur `main` + GH Environment `production` avec required reviewer (toi).

---

## 5. Secrets & rotation

Inventaire des env vars (cross-référencé avec `MEMORY.md` + `src/intelligence/main.py:1-22`).

| Var | Usage | Logué en clair ? | Rotation possible | Stockage actuel |
|---|---|---|---|---|
| `ANTHROPIC_API_KEY` | LLM narratives | **Non** — masqué `****` (`src/intelligence/security.py:258`) | Oui, console Anthropic, propagation immédiate | Railway Vars (env-injected) |
| `TELEGRAM_BOT_TOKEN` | Notifier | **Risque** — interpolé dans URL `f"https://api.telegram.org/bot{token}/sendMessage"` (`src/live_trading/alerting.py:255`). Si exception/HTTP error log inclut l'URL → leak. | Non sans recréer le bot ; rotation = nouveau bot, ré-inscription utilisateurs | Railway Vars |
| `TELEGRAM_CHAT_ID` | Destinataire | Non sensible (id, pas un secret) | N/A | Railway Vars |
| `SIGNAL_DB_PATH` | Path SQLite | Non | N/A | Railway Vars (default `./data/signals.db`) |
| `DATA_DIR` | OHLCV path | Non | N/A | Railway Vars |
| `DB_PASSWORD` (compose) | Postgres local uniquement | Non, `${VAR:?}` (`docker-compose.yml:108`) | Manuel | `.env` local, jamais commité (`.gitignore:67`) |
| `GRAFANA_PASSWORD` | Grafana local | Non, `${VAR:?}` | Manuel | `.env` local |
| `MT5_PASSWORD` | Broker | À auditer (hors scope ici) | Côté broker | Railway Vars |

### Findings concrets

1. **Pas de leak direct des secrets dans les logs `src/`** — seuls "initialized"/"not set" sont écrits. Bon.
2. **Risque potentiel Telegram** : si `requests`/`httpx` log une `RequestException`, l'URL complète (avec token) peut apparaître dans une stack-trace. Mitigation : ajouter un `logging.Filter` qui masque `bot[A-Za-z0-9:_-]{20,}` ou utiliser le client `python-telegram-bot` qui n'expose pas le token dans les exceptions.
3. **Railway Vars** : env-injected, pas d'audit log, pas de versioning. Acceptable pour solo founder, insuffisant pour SOC2.
4. **Pas de calendrier de rotation** documenté.

### Recommandations

- **Court terme (gratuit)** : créer `docs/SECRETS_ROTATION.md` avec un calendrier — Anthropic key tous les 90 j, Telegram bot recreation annuelle ou sur incident, NewsAPI 90 j. Ajouter un Calendar reminder.
- **Moyen terme (~$20/mo)** : **Doppler** ou **Infisical Cloud** (free tier suffit pour <5 secrets), intégration Railway/Fly.io native, audit log + versioning + sync automatique.
- **Anti-leak Telegram** : ajouter dans `src/intelligence/main.py` après `setup_logging`, un filter :
  ```python
  class SecretMaskFilter(logging.Filter):
      _PATTERNS = [re.compile(r"bot\d{8,12}:[A-Za-z0-9_-]{20,}")]
      def filter(self, record):
          msg = record.getMessage()
          for p in self._PATTERNS:
              msg = p.sub("bot***:***", msg)
          record.msg = msg; record.args = ()
          return True
  ```

---

## 6. IaC skeleton (Terraform extract)

Cible : **Fly.io** (provider Terraform officiel `fly-apps/fly`). À placer dans `infrastructure/terraform/main.tf`.

```hcl
terraform {
  required_version = ">= 1.6"
  required_providers {
    fly = {
      source  = "fly-apps/fly"
      version = "~> 0.0.23"
    }
  }
  backend "s3" {
    bucket = "sentinel-tfstate"
    key    = "fly/prod.tfstate"
    region = "eu-west-3"
  }
}

provider "fly" {
  fly_api_token = var.fly_api_token
}

variable "fly_api_token"        { type = string  sensitive = true }
variable "anthropic_api_key"    { type = string  sensitive = true }
variable "telegram_bot_token"   { type = string  sensitive = true }
variable "telegram_chat_id"     { type = string }
variable "region"               { type = string  default = "cdg" } # Paris
variable "image_tag"            { type = string  default = "latest" }

resource "fly_app" "sentinel" {
  name = "smart-sentinel"
  org  = "personal"
}

resource "fly_volume" "signals_db" {
  app    = fly_app.sentinel.name
  name   = "signals_data"
  size   = 10            # GB
  region = var.region
}

resource "fly_machine" "scanner" {
  app      = fly_app.sentinel.name
  region   = var.region
  name     = "sentinel-prod"
  image    = "ghcr.io/<owner>/smart-sentinel:${var.image_tag}"
  cpus     = 2
  cputype  = "shared"
  memorymb = 4096

  services = [{
    ports = [
      { port = 443, handlers = ["tls", "http"] },
      { port = 80,  handlers = ["http"] }
    ]
    protocol      = "tcp"
    internal_port = 8000
  }]

  mounts = [{
    volume = fly_volume.signals_db.id
    path   = "/app/data"
  }]

  env = {
    SENTINEL_TESTING_MODE = "0"          # ⚠️ fail-closed en prod
    LOG_FORMAT            = "json"
    LOG_LEVEL             = "INFO"
    NARRATIVE_MODE        = "template"
    SIGNAL_DB_PATH        = "/app/data/signals.db"
    DATA_DIR              = "/app/data"
    SYMBOLS               = "XAUUSD"
    VOL_MODE              = "hybrid"
  }

  secrets = {
    ANTHROPIC_API_KEY  = var.anthropic_api_key
    TELEGRAM_BOT_TOKEN = var.telegram_bot_token
    TELEGRAM_CHAT_ID   = var.telegram_chat_id
  }
}
```

### Effort migration estimé (jours-homme)

| Tâche | Effort |
|---|---|
| Créer `.dockerignore`, fixer Dockerfile, builder localement | 0.5 j |
| Créer `fly.toml` + `fly.staging.toml` minimal (sans Terraform) | 0.5 j |
| Pousser image vers GHCR via CI | 0.5 j |
| Migrer SQLite Railway → Fly volume (dump + restore) | 0.5 j |
| Bascule DNS / canary (10% → 100%) | 1 j |
| Terraform full (au-dessus du déploiement manuel) | 1.5 j |
| **Total migration Railway → Fly** | **~4 j** |

---

## 7. Latence régionale

Hypothèse : Anthropic API n'a pas de region EU dédiée publique (vérifier 2026 ; aujourd'hui les calls sortent vers `api.anthropic.com` US). Impact : **+80–120 ms** par call LLM depuis EU vs US.

| Provider | Régions EU | Régions US | Asia | Latence depuis FR vers app | Latence app→Anthropic (si app EU) |
|---|---|---|---|---|---|
| Railway | Amsterdam | US-East/West | Singapore | 20–35 ms (Amsterdam) | ~90 ms |
| Fly.io | `cdg` (Paris), `fra`, `ams`, `lhr`, `mad` | `iad`, `sjc`, `ord` | `nrt`, `sin`, `hkg` | **5–15 ms (Paris)** | ~85 ms |
| Render | Frankfurt | Oregon, Ohio | Singapore | 25–40 ms | ~95 ms |
| AWS Fargate | `eu-west-3` (Paris), `eu-central-1` | nombreuses | nombreuses | 5–20 ms | ~85 ms |

### Calcul SLA "signal < 30 s après bar close"

Décomposition end-to-end :
- Détection algo (ConfluenceDetector + VolForecaster) : ~200–500 ms par symbole.
- LLM call (Claude Sonnet, NARRATIVE_MODE=llm) : 1.5–3 s typique + ~85 ms RTT EU→US.
- Telegram envoi : 200–400 ms.
- **Total** : 2–4 s. Marge confortable face au SLA 30 s.

Si NARRATIVE_MODE=template (défaut prod actuel — cf. `MEMORY.md`/eval_05) : LLM enlevé → **~1 s end-to-end**.

### Recommandation

**Région cible = `cdg` (Paris) sur Fly.io**, ou `eu-west` Amsterdam sur Railway. Pas besoin de multi-region pour la phase actuelle (1 utilisateur founder + early testers EU).

---

## 8. Red-Team

| Objection | Réponse honnête |
|---|---|
| **"Le rollback est-il réellement testé ?"** | Non. Aucun test versioné. Action : exécuter `flyctl releases rollback` une fois en staging et chronométrer. KPI cible : MTTR < 5 min. |
| **"La migration Postgres (Prompt 12) casse le comparator ?"** | Risque réel : si on remplace SQLite par Postgres managed, le critère "SQLite persistance" devient caduc et **AWS RDS / Fly Postgres / Railway Postgres** entrent en jeu. Re-faire le comparatif après décision Postgres. |
| **"Fly.io machine = long-running ?"** | Oui. Configurer `auto_stop_machines = "off"` et `min_machines_running = 1`. Le scanner Sentinel est un loop, **pas** un handler request/response — un autoscaler request-based casserait le scan. |
| **"TESTING_MODE=1 par défaut → leak en prod ?"** | Très probable aujourd'hui si `SENTINEL_TESTING_MODE=0` n'est pas explicitement set sur Railway. Le `main.py:17` documente le default, et on n'a aucun guard "raise if PROD and TESTING_MODE". **À ajouter :** `if os.getenv("ENVIRONMENT")=="production" and os.getenv("SENTINEL_TESTING_MODE","1")=="1": sys.exit("Refusing to start: TESTING_MODE in PROD")`. |
| **"Railway nixpacks ignore le Dockerfile soigné"** | Confirmé : `railway.toml:2` = `nixpacks`. Tout le travail multi-stage est inutilisé en prod. Ajouter `[build] builder = "dockerfile"` + `dockerfilePath = "infrastructure/Dockerfile"`. |
| **"Procfile + railway.toml lancent `parallel_training.py`"** | Vrai et **majeur**. Si la prod tourne réellement, c'est l'ancien training RL, pas le SaaS. Si elle ne tourne pas, le SaaS n'est pas déployé. À clarifier avant tout autre chantier. |
| **"Pas de CI = chaque push peut casser la prod"** | Vrai. Il n'y a aucun guard. Le workflow §4 doit être posé avant la prochaine modif `src/`. |
| **"Image ~1 GB = cold-start lent"** | Probable. `torch+cpu` (~200 MB) + `transformers` (~500 MB) sont les coupables. Si FinBERT n'est pas utilisé en prod Sentinel (vérifier), retirer `transformers` → −500 MB. |
| **"Volume Fly = pinned à 1 machine"** | Vrai — Fly Volumes sont locaux NVMe. Pour HA, passer en LiteFS ou Postgres. Acceptable en single-instance. |
| **"Pas de backup automatique SQLite"** | Critique. Railway volume = pas de snapshot natif. Action : cron `litestream replicate` vers S3/R2 ($0–5/mo). |

---

## 9. Plan migration phasé

### J0 (cette semaine) — Stop the bleeding (effort: 1 j)
1. Corriger `Procfile` → `web: python -m src.intelligence.main` (Railway respecte web pour http).
2. Corriger `railway.toml` :
   ```
   [build]
   builder = "dockerfile"
   dockerfilePath = "infrastructure/Dockerfile"

   [deploy]
   startCommand = "python -m src.intelligence.main"
   restartPolicyType = "on_failure"
   restartPolicyMaxRetries = 5

   [[deploy.volumes]]
   name = "sentinel-data"
   mountPath = "/app/data"
   ```
3. Sur le dashboard Railway : set `SENTINEL_TESTING_MODE=0`, `LOG_FORMAT=json`, `NARRATIVE_MODE=template`, `ENVIRONMENT=production`.
4. Ajouter un guard fail-closed dans `main.py` pour TESTING_MODE en prod.
5. Créer `.dockerignore` (cf. §2).

### J+30 — Hardening (effort: 3 j cumulés)
6. Créer le workflow `.github/workflows/ci-cd.yml` (lint+test+build+smoke vers GHCR).
7. Mettre en place staging Railway (2e env, même image, secrets dédiés). Branch protection main.
8. Litestream backup SQLite → Cloudflare R2 (~$0.015/GB/mo).
9. Doppler/Infisical free tier pour secrets.
10. Trivy scan en CI (fail si HIGH/CRITICAL nouveau).
11. Documenter rollback dans `docs/RUNBOOK.md` + tester une fois.

### J+90 — Migration Fly.io (si nécessaire) (effort: 4 j)
12. Passer le projet Terraform Fly (cf. §6).
13. Bascule DNS canary (10% trafic Fly, 90% Railway) pendant 1 semaine.
14. Décommissioner Railway après validation.
15. Multi-region read-replicas si users APAC/US (LiteFS).

---

## 10. Top 5 actions (effort × impact)

| # | Action | Effort | Impact | Priorité |
|---|---|---|---|---|
| 1 | Corriger `Procfile` + `railway.toml` (entry point + Dockerfile builder + restart policy) | 0.5 j | **Bloquant** : sinon prod = ancien RL, pas le SaaS | P0 |
| 2 | Set `SENTINEL_TESTING_MODE=0` + guard fail-closed en prod | 0.25 j | Critique sécurité (auth bypass actuellement possible) | P0 |
| 3 | Créer `.dockerignore` + retirer `transformers` si FinBERT non utilisé | 0.5 j | Image −40 % (~600 MB), cold-start −30 % | P1 |
| 4 | GitHub Actions CI/CD (lint+test+build+smoke) | 1 j | Empêche les pushs cassés ; permet rollback automatique | P1 |
| 5 | Litestream backup SQLite vers R2/S3 | 0.5 j | Évite perte totale `signals.db` sur incident volume | P1 |

---

## 11. KPIs

| KPI | Cible | Mesure actuelle |
|---|---|---|
| Deploy frequency | ≥ 1×/semaine | inconnue (pas de CI) |
| MTTR (Mean Time To Recovery) | < 10 min | inconnu (pas de rollback testé) |
| Image size | < 600 MB | ~900 MB–1.2 GB estimé |
| Image build time (cache hit) | < 90 s | inconnu |
| Image build time (cold) | < 6 min | ~5–8 min estimé (torch + transformers) |
| Healthcheck success rate | > 99.5 % / 30 j | inconnu |
| $/mois infra | < $50 (phase test), < $150 (50 users payants) | ~$25–45 Railway estimé |
| Vulnérabilités HIGH/CRITICAL | 0 (Trivy) | inconnu |
| Cold-start | < 15 s | inconnu |
| Coverage tests CI | > 70 % | non gated |

---

## 12. Trade-offs assumés

1. **Solo founder** : on accepte de **rester sur Railway** (managed, 0 ops) plutôt que d'aller direct sur Fly+Terraform. Coût opportunité = -1 jour d'apprentissage IaC, gain = 4 j non passés à debug Fly.
2. **SQLite > Postgres** : tant qu'on a 1 worker et < 100 users payants, SQLite + Litestream = $0 vs $15–25/mo Postgres managed. À réévaluer si Prompt 12 force la migration.
3. **NARRATIVE_MODE=template** par défaut prod (déjà constaté eval_05) → **on assume** que le LLM est OFF en prod, ce qui rend Anthropic key non-bloquante. Si on rallume `llm`, refaire la latence §7.
4. **`transformers` (FinBERT)** : à supprimer du `requirements.txt` pour Sentinel SaaS si non utilisé — gain 500 MB. Risque : casser un feature legacy non audité ici.
5. **Pas de multi-region** : 1 EU (`cdg`/`ams`) est suffisant pour les early testers FR/EU. Multi-region à >100 users ou si les tests latence US dégradent.
6. **Pas de Vault complet** : Doppler/Infisical free tier remplace HashiCorp Vault tant qu'on a < 10 secrets et 1 dev. `src/security/secrets_manager.py` existe mais sur-dimensionné pour ce stade.
7. **Single env Railway court terme** : staging viendra à J+30 ; risque assumé d'un déploiement direct prod pendant 1 mois.
8. **Image dev/prod dans le même Dockerfile** : on accepte le risque qu'un dev build `--target development` accidentellement ; mitigé par le fait que le CI build explicitement `--target production`.
