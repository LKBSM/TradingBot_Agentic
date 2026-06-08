# Déploiement Docker — Smart Sentinel AI

**Date de bascule** : 2026-05-16
**Remplace** : Railway (config legacy dans `.deploy_legacy/`)
**Statut** : canonique

---

## Pourquoi Docker (vs Railway)

| Critère                         | Railway                      | Docker (choisi)              |
| ------------------------------- | ---------------------------- | ---------------------------- |
| Portabilité                     | Vendor lock-in               | ✅ Tout cloud / on-prem      |
| Coût à scale                    | Forfait Railway              | ✅ AWS/GCP/Azure spot etc.   |
| Contrôle infra                  | Limité (PaaS)                | ✅ Full (CPU, RAM, GPU)      |
| Stack obs                       | Plugins payants              | ✅ Prometheus/Grafana inclus |
| Reproductibilité prod = local   | Build dépend Railway         | ✅ Image identique partout   |
| Multi-region / failover         | Premium plan                 | ✅ K8s, ECS, Nomad, etc.     |

Le brief §6 critère #3 demande *"tout signal des 12 derniers mois rejouable à l'identique"*. Une image Docker `smart-sentinel:institutional` versionnée est plus solide qu'un build Railway opaque.

---

## Quick start

### 1. Setup `.env`

```bash
cp .env.example .env
# Edit .env : remplir ANTHROPIC_API_KEY, TELEGRAM_BOT_TOKEN, etc.
```

### 2. Build l'image

```bash
docker build -t smart-sentinel:institutional .
```

Première fois : ~5-10 min (download base + install deps). Builds suivants : ~30s grâce au cache layer.

### 3. Lancer le stack

**Mode minimal (juste l'app)** :

```bash
docker compose up -d
```

**Mode full stack** (app + Prometheus + Grafana + Redis + Postgres + Alertmanager) :

```bash
docker compose -f infrastructure/docker-compose.yml up -d
```

### 4. Vérifier

```bash
docker compose ps
docker compose logs -f sentinel
curl http://localhost:8000/health
```

Endpoint healthy attendu : `{"status": "ok", "scanner_running": true, ...}`.

### 5. Stop / clean

```bash
docker compose down                # Stop containers, garde volumes
docker compose down -v             # Stop + delete volumes (perte data)
```

---

## Configuration via env vars

Toutes les env vars du brief sont passées via `.env` (chargé automatiquement par docker compose) :

```
# .env
ANTHROPIC_API_KEY=sk-...                # Required for LLM narratives
TELEGRAM_BOT_TOKEN=...                  # Required for Telegram delivery
TELEGRAM_CHAT_ID=...
VOL_MODE=har                            # har/lgbm/hybrid (default har)
SYMBOLS=XAUUSD,EURUSD                   # comma-separated
SENTINEL_TESTING_MODE=1                 # 1=auth bypass, 0=tier gated
LOG_LEVEL=INFO
LOG_FORMAT=json
CORS_ALLOWED_ORIGINS=http://localhost:3000
```

---

## Build pour production

### Tag versionné (recommandé)

```bash
docker build -t smart-sentinel:v1.0.0-institutional .
docker tag smart-sentinel:v1.0.0-institutional smart-sentinel:latest
```

### Push vers registry privé (ECR / GCR / Harbor / Docker Hub)

```bash
docker tag smart-sentinel:v1.0.0-institutional REGISTRY/smart-sentinel:v1.0.0-institutional
docker push REGISTRY/smart-sentinel:v1.0.0-institutional
```

### Reproductibilité bit-à-bit

L'image doit être identique à chaque build sur le même tag git :

```bash
git checkout v0.9.0-pre-institutional
docker build --pull --no-cache -t smart-sentinel:v0.9.0 .
docker image inspect smart-sentinel:v0.9.0 --format '{{.Id}}'
# → digest doit matcher entre deux builds sur la même machine
```

---

## Volumes persistants

Trois volumes pour les données stateful :

| Volume                  | Contenu                              | Backup recommandé        |
| ----------------------- | ------------------------------------ | ------------------------ |
| `smart-sentinel-data`   | OHLCV CSV, SQLite signals.db         | Quotidien                |
| `smart-sentinel-logs`   | Logs JSON applicatifs                | 30 jours rétention       |
| `smart-sentinel-models` | HMM persisté, LGBM .pkl, scoring v3  | À chaque train run       |

### Backup d'un volume

```bash
docker run --rm \
  -v smart-sentinel-data:/source:ro \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/data_$(date +%F).tar.gz -C /source .
```

### Restore

```bash
docker run --rm \
  -v smart-sentinel-data:/dest \
  -v $(pwd)/backups:/backup:ro \
  alpine tar xzf /backup/data_2026-05-16.tar.gz -C /dest
```

---

## Déploiement cloud (production)

### Option A — AWS ECS Fargate

```bash
# 1. Push to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker tag smart-sentinel:institutional <account>.dkr.ecr.<region>.amazonaws.com/smart-sentinel:institutional
docker push <account>.dkr.ecr.<region>.amazonaws.com/smart-sentinel:institutional

# 2. Define task in ECS console (2 vCPU, 4 GB RAM)
# 3. Create service with ALB on port 8000
```

### Option B — GCP Cloud Run

```bash
gcloud builds submit --tag gcr.io/<project>/smart-sentinel:institutional .
gcloud run deploy smart-sentinel \
  --image gcr.io/<project>/smart-sentinel:institutional \
  --port 8000 \
  --cpu 2 --memory 4Gi \
  --min-instances 1 --max-instances 5 \
  --set-env-vars VOL_MODE=har,SYMBOLS=XAUUSD,EURUSD
```

### Option C — Self-host (VPS Hetzner / OVH)

```bash
# Sur le VPS
docker compose up -d
# Configure reverse proxy (Caddy / nginx) port 80/443 → 8000
```

---

## Migration depuis Railway

Le `railway.toml` legacy est dans `.deploy_legacy/`. Pour réactiver :

```bash
git mv .deploy_legacy/railway.toml.disabled railway.toml
git mv .deploy_legacy/Procfile.disabled Procfile
```

**Données à migrer** depuis le volume Railway `sentinel-data` :

1. Snapshot du volume Railway via leur dashboard → `signals.db` + `data/*.csv`.
2. Restore dans le volume Docker local :
   ```bash
   docker cp signals.db smart-sentinel:/app/data/signals.db
   docker cp data/. smart-sentinel:/app/data/
   ```

---

## Logging + observability

L'app log en JSON structuré (`LOG_FORMAT=json`). Pour parser :

```bash
docker compose logs sentinel | jq 'select(.level == "ERROR")'
```

Métriques Prometheus exposées sur `/metrics` (port 8000). Pour scrape :

- Stack complet : `docker compose -f infrastructure/docker-compose.yml up -d` (inclut Prometheus + Grafana avec dashboards pré-provisionnés).
- Externe : pointer un Prometheus existant sur `http://localhost:8000/metrics`.

---

## CI/CD

Le workflow GitHub Actions `algo_tests.yml` (Sprint 0) build l'image au push :

```yaml
# Suggestion d'extension Sprint 6 batch 6.4 :
- name: Build Docker image
  run: docker build -t smart-sentinel:${{ github.sha }} .

- name: Test container starts
  run: |
    docker run -d --name test-sentinel -p 8000:8000 \
      -e SENTINEL_TESTING_MODE=1 \
      smart-sentinel:${{ github.sha }}
    sleep 10
    curl -f http://localhost:8000/health
    docker stop test-sentinel
```

---

## Troubleshooting

| Symptôme                                    | Cause probable             | Fix                                            |
| ------------------------------------------- | -------------------------- | ---------------------------------------------- |
| `docker compose up` : ports already in use | Service local sur 8000     | `lsof -i :8000` puis kill, ou changer le port |
| Image build échoue sur `pip install`       | Réseau / proxy             | `docker build --network=host`                  |
| Health check fail                           | App pas encore démarrée    | Attendre 30s (start_period configuré)         |
| Volume permission denied                    | UID host ≠ UID conteneur   | `chown -R 999:999 ./data` ou volumes nommés    |
| Out of memory                               | Backtest 7y consomme RAM   | Augmenter `mem_limit` dans docker-compose      |

---

## Désactivation Railway (effectuée 2026-05-16)

Fichiers archivés :

- `.deploy_legacy/railway.toml.disabled`
- `.deploy_legacy/Procfile.disabled`

Le repo ne propose plus Railway par défaut. Si tu redéploies sur Railway, déplacer les fichiers à la racine et adapter les env vars.

---

**Maintainer** : Backtest Infrastructure agent (`agents/backtest_infrastructure/CHARTER.md` extends RACI to deployment).
