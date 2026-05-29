# Plan de Commercialisation — Catégorie 19 : MLOps & Deployment

> **Périmètre** : Dockerfile, docker-compose, CI/CD pipeline, model versioning/registry, train/serve consistency, deploy pipeline, infrastructure (cloud choice), secrets management, rollback strategy, observability deploy.
>
> **Sources audit** :
> - `reports/eval_22_deployment.md` (note 4.5/10 — Procfile/railway.toml lancent encore `parallel_training.py` legacy)
> - `reports/eval_23_mlops.md` (note 4.5/10 — Maturity Level 1, 4 skews train/serve identifiés)
> - `infrastructure/Dockerfile:1-112`, `infrastructure/docker-compose.yml:1-216`
> - `.github/workflows/ci.yml:1-150`, `.github/workflows/algo_tests.yml:1-46`
> - `src/intelligence/main.py:76-…` (`assert_safe_production_config`)
> - `models/*.pkl|*.lgb` (5 artefacts pickle non versionnés)
> - `requirements.txt` (mix `==` / `>=`, builds non reproductibles)

---

## 1. État actuel (Audit)

### 1.1 Inventaire factuel

| Élément | État | Référence |
|---|---|---|
| `infrastructure/Dockerfile` | **OK** (multi-stage, non-root, healthcheck, port 8000) | `Dockerfile:14,41,55,78,81,89` |
| `infrastructure/docker-compose.yml` | OK pour local stack ; restart `unless-stopped`, fail-closed `SENTINEL_TESTING_MODE=0`, ports `127.0.0.1` | `docker-compose.yml:33,41` |
| `Procfile` | **ABSENT** à la racine (purgé depuis eval_22 ?) | `ls Procfile → not found` |
| `railway.toml` | **ABSENT** à la racine | `ls railway.toml → not found` |
| `.dockerignore` | **OK** présent (secrets, data, models, reports exclus) | `.dockerignore:1-103` |
| `.github/workflows/ci.yml` | Présent mais **partiel** (tests data-sprints uniquement, pas de build/deploy) | `ci.yml:91-150` |
| `.github/workflows/algo_tests.yml` | Tests algo `not slow/integration/skip_on_ci`, pas de deploy | `algo_tests.yml:1-46` |
| Model registry | **ABSENT** — 5 `.pkl/.lgb` posés à plat dans `models/` (a1_stack_v1, factor_model_v1, scoring_v2/v3) | `ls models/` |
| Train/serve fingerprint | **ABSENT** — 4 skews concrets (sessions, currency, blend, hybrid residual) | `eval_23:#3` |
| Secrets management | `.env` local + Railway Vars env-injected ; pas de Doppler/Vault | `eval_22:§5` |
| Restart policy prod | **INCONNU** (railway.toml retiré, fly.toml jamais créé) | — |
| Healthcheck endpoint | `/health` via `curl` (Dockerfile) | `Dockerfile:81-82` |
| Image scanning | **ABSENT** (pas de Trivy en CI) | — |
| SBOM / supply-chain | **ABSENT** | — |
| Database migrations | **ABSENT** (Alembic non installé, SQLite `signals.db` créée à la volée) | `data/signals.db` |
| Blue/green ou canary | **ABSENT** | — |
| Infrastructure as Code | **ABSENT** (skeleton Terraform Fly proposé eval_22:§6 mais non écrit) | — |
| Rollback documenté | **ABSENT** (`docs/RUNBOOK.md` à créer) | — |
| Backups SQLite | **ABSENT** (pas de litestream, pas de snapshot Railway natif) | — |
| Multi-region | N/A (single instance) | — |

### 1.2 Bloqueurs critiques (P0 absolu)

1. **Aucun fichier deploy à la racine** : ni `Procfile`, ni `railway.toml`, ni `fly.toml`. Le repo **n'est plus déployable** par un PaaS auto-build sans intervention manuelle. Soit la prod tourne via override manuel non versionné, soit elle ne tourne pas du tout.
2. **5 artefacts modèles `.pkl` non versionnés** dans `models/` (`a1_stack_v1.pkl` 1.4 MB, `factor_model_v1.pkl` 1.2 MB, etc.) — pas de SHA, pas de manifest, pas de lineage. Re-train = perte de traçabilité.
3. **CI `ci.yml`** ne builde **pas** d'image Docker, ne pousse rien vers un registry, ne déploie nulle part. C'est une simple gate test unitaire (55 % coverage data-sprint).
4. **4 skews train/serve** (sessions/currency/blend/hybrid residual — `eval_23:§3`) saignent silencieusement 15-25 % RMSE en prod. Aucun fingerprint feature, aucun test de cohérence.
5. **Secrets en clair via env Railway/local `.env`** — pas d'audit log, pas de rotation, risque leak Telegram token via stack-traces (`eval_22:§5`).
6. **`SENTINEL_TESTING_MODE=1` est le défaut** (`MEMORY.md`) — `main.py:76` ajoute un fail-closed via `assert_safe_production_config()` mais dépend que `ENVIRONMENT=production` soit set.

### 1.3 Score MLOps maturity

| Niveau | Description | Notre cas |
|---|---|---|
| Level 0 | Notebooks manuels, pas de pipeline | — |
| **Level 1** | Scripts versionnés, deploy manuel, CI tests | **← nous (4.5/10)** |
| Level 2 | CI build + push image + deploy staging auto | cible J+30 |
| Level 3 | Model registry + regression gate + drift monitor + canary | cible J+90 |
| Level 4 | Automated retrain on drift, A/B testing, shadow deploy | hors scope v1 |

---

## 2. Vision cible (CI/CD ironclad, model registry, blue/green, MLOps Level 3)

### 2.1 Pipeline cible end-to-end

```
[dev push]
   → [CI: lint → test → security → build image]
   → [push GHCR ghcr.io/<owner>/smart-sentinel:<sha>]
   → [deploy staging auto via Fly.io machines]
   → [smoke staging /health + /metrics + 1 signal de bout-en-bout]
   → [manual approval (GH Environments)]
   → [deploy prod canary 10 % → 100 % en 30 min]
   → [post-deploy rollback drill mensuel chronométré]
```

### 2.2 Train / serve cible

```
[Colab seeded + data SHA manifest]
   → [export model.pkl + meta.json (git_sha, data_sha, features_fingerprint)]
   → [PR draft → CI regression gate (RMSE ≤ 1.02 × baseline)]
   → [merge → CI promeut artefact en GHCR OCI layer ou R2]
   → [main.py:_calibrate_system disabled → load_state(model.pkl) at boot]
   → [drift cron daily PSI vs train snapshot]
   → [retrain candidate si PSI > 0.25 OR RMSE 14d > 1.10 × baseline]
```

### 2.3 Stack cible (coût/mo)

| Composant | Choix recommandé | Coût |
|---|---|---|
| Hosting prod | **Fly.io** (Paris `cdg`) | $30–40/mo |
| Hosting staging | **Fly.io** shared-cpu-1x scale-to-zero | $2-5/mo |
| Registry image | **GHCR** (gratuit pour repos publics ; payant si privé) | $0–4/mo |
| Model registry / artefact storage | **Cloudflare R2** | $1/mo |
| Backups SQLite (litestream) | **Cloudflare R2** | $0.5/mo |
| Secrets manager | **Doppler free tier** (≤ 5 users, audit log inclus) | $0 |
| Drift monitor + alerting | Cron Fly.io machine + Telegram existant | $0 |
| MLflow tracking | **DEFER** ; print metrics + git tags suffisent v1 | $0 |
| IaC | Terraform fly-apps/fly provider | $0 |
| **TOTAL infra prod + staging** | | **~$35–50/mo** |

---

## 3. Gap analysis

| Domaine | Cible | Actuel | Gap | Priorité |
|---|---|---|---|---|
| Entry deploy versionné | `fly.toml` + `Dockerfile` source of truth | Aucun fichier deploy | **CRITIQUE** | P0 |
| CI build + push image | `ci-cd.yml` build → GHCR push | `ci.yml` test seul | Manque build + smoke | P0 |
| Hébergement production | Fly.io `cdg` Dockerfile mode, restart `on_failure` | Inconnu (peut-être Railway nixpacks) | À choisir + provisionner | P0 |
| Model registry | Manifest JSON + R2 + git SHA tag | 5 pkl à plat | Manque metadata + lineage | P0 |
| Secrets management | Doppler free tier sync vers Fly secrets | Env vars Railway/.env | Manque audit + rotation | P0 |
| Restart policy | `restart on_failure` minimum | Inconnu (eval_22 mentionne `never` historique) | À confirmer | P0 |
| Train/serve consistency | `features_fingerprint` SHA des 10 features clés | 4 skews silencieux | Manque shared `features/` module | P0 (cf. eval_23 cat. 4) |
| Healthcheck robuste | `/health` + composant-level + start-period 45 s | `curl /health` 30 s | OK mais ajouter Python-based (drop curl) | P1 |
| Blue/green ou canary | Fly machines + DNS canary 10 % → 100 % | Single instance | À ajouter J+30 | P1 |
| DB migrations | Alembic SQLite → eventuelle Postgres | Pas d'Alembic | À ajouter avant Postgres | P1 |
| Image scanning | Trivy en CI (fail HIGH/CRITICAL) | Aucun | À ajouter | P1 |
| SBOM + signature | Syft + cosign sign images | Aucun | À ajouter | P2 |
| Multi-region failover | LiteFS ou Fly multi-region | Single | Hors scope v1 | P2 |
| IaC Terraform | `infrastructure/terraform/main.tf` cf. eval_22:§6 | Skeleton non écrit | À écrire | P2 |
| Rollback drill | Mensuel chronométré, MTTR < 10 min | Jamais testé | À documenter + drill | P1 |

---

## 4. Plan d'exécution

### Matrice de priorités synthèse

| ID | Tâche | Prio | Effort | Coût/mo | Dépendances |
|---|---|---|---|---|---|
| MLOPS-01 | Choix hébergement final (matrice) | P0 | 0.5 j | — | — |
| MLOPS-02 | `fly.toml` + `fly.staging.toml` + kill Procfile/railway legacy | P0 | 1 j | $35-50 | MLOPS-01 |
| MLOPS-03 | GitHub Actions `ci-cd.yml` build → test → security → deploy | P0 | 2 j | $0 | MLOPS-02 |
| MLOPS-04 | Model registry minimal (R2 + manifest JSON) | P0 | 1.5 j | $1 | — |
| MLOPS-05 | Doppler free tier + sync Fly secrets | P0 | 0.5 j | $0 | MLOPS-02 |
| MLOPS-06 | Restart policy `on_failure` + healthcheck Python-based | P0 | 0.25 j | — | MLOPS-02 |
| MLOPS-07 | Blue/green via Fly machine clone + canary DNS 10 % | P1 | 2 j | — | MLOPS-02, MLOPS-03 |
| MLOPS-08 | Alembic migrations SQLite | P1 | 1 j | $0 | — |
| MLOPS-09 | Train/serve fingerprint + regression gate | P1 | 2 j | $0 | eval_23:§5 |
| MLOPS-10 | Trivy image scanning + Syft SBOM en CI | P1 | 0.5 j | $0 | MLOPS-03 |
| MLOPS-11 | Rollback runbook + drill mensuel | P1 | 0.5 j | $0 | MLOPS-02 |
| MLOPS-12 | Litestream backup SQLite → R2 | P1 | 0.5 j | $0.5 | MLOPS-02 |
| MLOPS-13 | Multi-region failover (LiteFS ou Fly multi-region) | P2 | 3 j | $20-30 | MLOPS-07, MLOPS-12 |
| MLOPS-14 | IaC Terraform fly-apps/fly + backend S3 | P2 | 2 j | $0-2 | MLOPS-02 |
| MLOPS-15 | Cosign sign + GitHub attestation | P2 | 0.5 j | $0 | MLOPS-10 |

---

### P0 — MLOPS-01 : Choix hébergement final (matrice coût/complexité)

**Livrables**
- `docs/INFRASTRUCTURE_CHOICE.md` : matrice 5 providers × 8 critères, décision documentée et signée.
- Compte créé chez provider retenu, billing alert configurée.

**Spec cible** : 2 vCPU partagé, 4 GiB RAM, 1 worker long-running (scanner Sentinel), région EU (FR/AMS/FRA pour GDPR + latence Anthropic), volume persistant ≥ 5 GB pour `signals.db`, healthcheck HTTP `/health`.

#### Matrice 5 providers × 8 critères (à vérifier 2026)

| Critère | **Railway** | **Fly.io** | **Render** | **Hetzner Cloud** | **AWS ECS Fargate** |
|---|---|---|---|---|---|
| **$/mois (spec cible)** | $25–45 (starter + usage) | **$30–40** (shared-cpu-2x 4G + 5G volume) | $25 (Standard) → $85 (Pro) | **$8–12** (CX21 2vCPU/4G) + setup | $90+ (Fargate + ALB + EBS) |
| **Setup time** | ~30 min | ~1 h | ~30 min | ~6 h (cloud-init, fail2ban, traefik) | ~1 j (Terraform/CDK) |
| **EU region GDPR** | Amsterdam | `cdg`/`fra`/`ams`/`mad` (35+ régions) | Frankfurt | Nuremberg/Falkenstein/Helsinki | `eu-west-3` Paris, `eu-central-1` Francfort |
| **IaC maturity** | `railway.toml` + CLI (pas de Terraform officiel) | `fly.toml` + provider Terraform officiel | `render.yaml` Blueprint | hcloud Terraform officiel | Terraform/CDK natif mais verbeux |
| **Cold-start** | 3–8 s | <1 s (always-on) ; 5-10 s si auto-stop | 0 s Standard | 0 s (machine dédiée) | 30-60 s Fargate provisioning |
| **Persistant volume** | Railway Volume natif | Fly Volumes NVMe (pinned 1 machine) | Disk add-on (≥$1/GB/mo) | LV/LVM sur disk SSD inclus | EFS ($0.30/GB) ou EBS |
| **Lock-in** | Moyen (nixpacks particulier) | **Faible** (Dockerfile + fly.toml portable) | Moyen (Blueprint propriétaire) | **Très faible** (Linux pur, port n'importe où) | Élevé (ALB + IAM + VPC + task def) |
| **Audit/SOC2 ready** | Non | Soc2 Type II en cours | Soc2 Type II | Iso 27001 (datacenter) | SOC2 Type II ✓ |
| **Scalabilité horizontale** | Replicas manuels | `fly scale count N` natif + regions | Replicas avec disk constraints | Manuel (Terraform + LB) | Auto-scaling natif via ECS Service |

#### Recommandation

**Décision : Fly.io `cdg` (Paris) production + staging.**

**Justification**
- **Coût $30-40/mo** prod aligné avec spec ($35-50 incluant backups + staging scale-to-zero).
- **Région EU obligatoire** pour GDPR clients FR-first (cf. ICP wedge `eval_25_pmf_icp`).
- **Faible lock-in** : Dockerfile portable, peut être migré vers Hetzner en 1 j si revenue justifie.
- **`auto_stop_machines = "off"`** + `min_machines_running = 1` est natif — convient parfaitement au scanner long-running (PAS un handler request/response).
- **Terraform officiel** disponible pour IaC (MLOPS-14).
- **Multi-region** disponible nativement pour J+90 (LiteFS pour SQLite ou bascule Postgres).

**Pourquoi pas les autres ?**
- Railway : `nixpacks` ignore notre Dockerfile soigné (eval_22:§9). Bloquant immédiat.
- Render : disk add-on cher au GB, Frankfurt-only EU.
- Hetzner : 50-70 % moins cher mais ~6 h setup + ops permanent. **Cible de migration phase 2** (>100 users payants, $500+ ARR/mo justifie un sysadmin partiel).
- AWS Fargate : 2-3× le coût pour solo founder sans bénéfice côté ops.

**Acceptance**
- Compte Fly.io créé sous email pro, MFA activé.
- App `smart-sentinel-prod` + `smart-sentinel-staging` provisionnée région `cdg`.
- Volume `signals_data` 10 GB attaché à prod.
- Billing alert $50/mo configurée.

**Dépendances** : aucune.

---

### P0 — MLOPS-02 : `fly.toml` + `fly.staging.toml` + suppression Procfile/railway legacy

**Livrables**
- `fly.toml` à la racine (prod, `cdg`, 2 vCPU/4G).
- `fly.staging.toml` (shared-cpu-1x, scale-to-zero).
- Confirmation que `Procfile` et `railway.toml` sont définitivement absents (déjà vérifié) ou supprimés du git history si présents dans une branche oubliée.
- README `docs/DEPLOY.md` avec procédure `flyctl deploy --remote-only --config fly.toml`.

**Contenu `fly.toml` (prod)** :
```toml
app = "smart-sentinel"
primary_region = "cdg"

[build]
dockerfile = "infrastructure/Dockerfile"
build-target = "production"

[env]
ENVIRONMENT = "production"
SENTINEL_TESTING_MODE = "0"     # fail-closed
LOG_FORMAT = "json"
LOG_LEVEL = "INFO"
NARRATIVE_MODE = "template"
SIGNAL_DB_PATH = "/app/data/signals.db"
DATA_DIR = "/app/data"
SYMBOLS = "XAUUSD"
VOL_MODE = "har"                # cf. eval_04 — lgbm/hybrid latency hors cible

[http_service]
internal_port = 8000
force_https = true
auto_stop_machines = "off"       # scanner long-running
auto_start_machines = true
min_machines_running = 1

  [http_service.checks]
    [http_service.checks.health]
    interval = "30s"
    timeout = "10s"
    grace_period = "45s"
    method = "get"
    path = "/health"

[[mounts]]
source = "signals_data"
destination = "/app/data"

[[vm]]
cpu_kind = "shared"
cpus = 2
memory_mb = 4096

[deploy]
strategy = "rolling"             # bumped à "bluegreen" en MLOPS-07
```

**Contenu `fly.staging.toml`** : idem mais `app = "smart-sentinel-staging"`, `min_machines_running = 0`, `auto_stop_machines = "stop"`, `NARRATIVE_MODE = "template"`, faux ANTHROPIC_API_KEY.

**Heures** : 1 j (incluant test bascule + smoke /health post-deploy).
**Coût** : $35-50/mo (prod + staging combinés).
**Acceptance**
- `flyctl deploy --config fly.toml` réussit, container reste UP > 30 min sans crash.
- `curl https://smart-sentinel.fly.dev/health` retourne 200.
- Logs JSON visibles via `flyctl logs --app smart-sentinel`.
- Volume `signals.db` persiste après `flyctl machine restart`.

**Dépendances** : MLOPS-01.

---

### P0 — MLOPS-03 : GitHub Actions CI/CD (build → test → security → deploy staging → manual prod)

**Livrables**
- `.github/workflows/ci-cd.yml` (nouveau, distinct de `ci.yml`).
- GH Environments `staging` (auto) + `production` (required reviewer = loukmanebessam).
- Branch protection sur `main` : status checks `lint`, `test`, `build`, `smoke-image` requis.
- Secrets GH Actions : `FLY_API_TOKEN_STAGING`, `FLY_API_TOKEN_PROD`, `DOPPLER_TOKEN`.

**Pipeline 7 jobs séquentiels + parallélisme** :
1. **lint** (ruff + black, advisory au début, blocking après J+30).
2. **test** (réutilise `algo_tests.yml` curated set, fail-fast).
3. **security** (Trivy filesystem scan + `pip-audit` — cf. MLOPS-10).
4. **build** (`docker/build-push-action@v6`, target `production`, push GHCR, tags `:sha` + `:latest` + `:vX.Y.Z` si tag git).
5. **smoke-image** (docker run local, wait 60 s, probe `/health` 20 retries × 3 s).
6. **deploy-staging** (auto sur push main, `flyctl deploy --config fly.staging.toml`).
7. **smoke-staging** (curl https://smart-sentinel-staging.fly.dev/health en boucle).
8. **deploy-prod** (manual approval via GH Environment, `flyctl deploy --config fly.toml`).
9. **smoke-prod** (post-deploy probe + rollback auto si fail).

**Squelette YAML complet** : cf. `reports/eval_22_deployment.md:§4` (déjà fourni, 130 lignes).

**Améliorations vs squelette eval_22**
- Job `security` ajouté entre `test` et `build` (Trivy + pip-audit + Bandit).
- Cache pip + cache buildx GHA mode=max (build time -60 %).
- Tag image `:sha` (immutable) + `:latest` (mutable). Rollback = redeploy `:<previous_sha>`.
- Job `release` final si git tag : crée GitHub Release + attache SBOM + image SHA.

**Heures** : 2 j (incluant debug sur premier run + tuning timeouts).
**Coût/mo** : $0 (GH Actions free 2000 min/mo pour public, 3000 min/mo pour Team).
**Acceptance**
- Un push main vert déclenche un déploiement staging automatique.
- Un push main rouge (test fail) bloque le merge.
- Manual approval prod fonctionne (badge GH "Waiting for review").
- Image GHCR taggée `ghcr.io/<owner>/smart-sentinel:<sha>` accessible.
- Smoke staging fail = pas de propagation vers prod.

**Dépendances** : MLOPS-02 (fly.toml existe).

---

### P0 — MLOPS-04 : Model registry minimal (R2 + manifest JSON + SQLite metadata)

**Livrables**
- `infrastructure/model_registry/` : helpers Python `publish_model.py`, `fetch_model.py`, `list_versions.py`.
- `data/models/manifest.json` (committé) — index immutable des versions.
- Bucket Cloudflare R2 `sentinel-models` : artefacts `.pkl/.lgb` versionnés.
- `models/REGISTRY.md` documentant le contrat.

**Schema manifest.json** :
```json
{
  "models": {
    "scoring_v3_lgbm": {
      "versions": [
        {
          "version": "1.0.0",
          "git_sha": "f0d41df...",
          "data_sha256": "a1b2c3...",     // hash du CSV d'entraînement
          "features_fingerprint": "sha256:...", // ordre + types des features
          "metrics": {"auc": 0.62, "brier": 0.21, "rmse_holdout": 0.034},
          "trained_at": "2026-05-16T01:16:00Z",
          "trained_by": "colab_v3_lgbm.py",
          "r2_url": "https://r2.../models/scoring_v3_lgbm/1.0.0/model.pkl",
          "r2_sha256": "...",
          "promoted": "production",     // candidate | staging | production | retired
          "rollback_to": "0.9.0"
        }
      ]
    }
  }
}
```

**Pourquoi PAS MLflow v1**
- Solo founder, 3 modèles actifs, 1 prod déployé. MLflow tracking server = 200 LOC pour rien.
- Re-évaluer à 2ᵉ architecture modèle ou $1k MRR (cf. `eval_23:§9` recommandation DEFER).

**Train/serve fingerprint** (cf. MLOPS-09 et eval_23:§3) :
- `features_fingerprint = sha256(json.dumps(sorted([(name, dtype) for name, dtype in feature_schema])))`.
- Calculé à `fit` ET à `serve`. Si mismatch → `RuntimeError("Features schema drift detected")` à `load_state`.

**Heures** : 1.5 j (R2 setup + scripts + premier upload + README).
**Coût/mo** : $1 R2 (10 GB free, $0.015/GB après ; 5 modèles × ~1 MB = négligeable).
**Acceptance**
- `python infrastructure/model_registry/publish_model.py --model scoring_v3_lgbm --version 1.0.0 --pkl models/scoring_v3_lgbm.pkl` push vers R2 + update manifest.
- `python infrastructure/model_registry/fetch_model.py --model scoring_v3_lgbm --version production` télécharge + vérifie SHA256.
- `main.py` charge le modèle depuis manifest au lieu de hard-code path `models/scoring_v3_lgbm.pkl`.

**Dépendances** : aucune (peut démarrer en parallèle de MLOPS-02).

---

### P0 — MLOPS-05 : Secrets management (Doppler free tier)

**Livrables**
- Doppler workspace `smart-sentinel` avec projets `dev`, `staging`, `prod`.
- Tous les secrets migrés depuis `.env` local + Fly Vars dashboard vers Doppler.
- CI/CD `ci-cd.yml` job `deploy-*` lit `DOPPLER_TOKEN` puis `doppler run -- flyctl deploy`.
- `.env` local ne contient plus que `DOPPLER_TOKEN` personnel (chacun le sien).
- Suppression des secrets du git history (BFG ou `git-filter-repo` si historique pollué).
- `docs/SECRETS_ROTATION.md` : calendrier rotation (Anthropic 90 j, NewsAPI 90 j, Telegram bot annuel).

**Pourquoi Doppler vs Vault/SSM**
- Free tier ≤ 5 users couvre solo founder + 1 dev futur.
- Audit log + versioning + sync natif Fly.io, GitHub Actions, Railway.
- Pas de cluster Vault à opérer ($30+ /mo minimum).
- Migration future possible vers AWS Secrets Manager / Vault sans réécrire le code (env-injected reste).

**Anti-leak Telegram** (cf. eval_22:§5) :
```python
# src/intelligence/main.py — ajouter après setup_logging()
class SecretMaskFilter(logging.Filter):
    _PATTERNS = [
        re.compile(r"bot\d{8,12}:[A-Za-z0-9_-]{20,}"),  # Telegram
        re.compile(r"sk-ant-[A-Za-z0-9_-]{40,}"),       # Anthropic
        re.compile(r"sk-[A-Za-z0-9]{40,}"),             # generic OpenAI-style
    ]
    def filter(self, record):
        msg = record.getMessage()
        for p in self._PATTERNS:
            msg = p.sub("***REDACTED***", msg)
        record.msg = msg
        record.args = ()
        return True
```

**Heures** : 0.5 j.
**Coût/mo** : $0 (Doppler free tier).
**Acceptance**
- `doppler secrets --project smart-sentinel --config prod` liste 8+ secrets attendus.
- `doppler run --config prod -- python -m src.intelligence.main` boot OK.
- Aucun secret en clair dans `git log -p`.
- `pytest tests/test_secret_mask_filter.py` passe (à créer).

**Dépendances** : MLOPS-02 (fly.toml existe pour intégration deploy).

---

### P0 — MLOPS-06 : Restart policy `on_failure` + healthcheck Python-based

**Livrables**
- `fly.toml` : `[deploy].strategy = "rolling"`, healthcheck `grace_period = "45s"` (cf. boot calibrate ~30 s), `restart_policy = "on-failure"` implicite via Fly Machines.
- `Dockerfile` HEALTHCHECK Python-based (drop curl pour réduire image -3 MB) :
  ```dockerfile
  HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
      CMD python -c "import urllib.request,sys; \
          sys.exit(0) if urllib.request.urlopen('http://localhost:8000/health',timeout=5).status==200 else sys.exit(1)"
  ```
- `/health` endpoint déjà OK (`src/api/routes/health.py` audit eval_22:§1).
- Documenter `restart_policy_max_retries = 5` dans `fly.toml` ou équivalent Railway si fallback.

**Heures** : 0.25 j.
**Coût** : nul.
**Acceptance**
- `flyctl machine restart` puis `flyctl status` affiche `passing` après ≤ 60 s.
- Crash forcé (`kill -9` PID dans le container) déclenche un restart automatique en ≤ 30 s.

**Dépendances** : MLOPS-02.

---

### P1 — MLOPS-07 : Blue/green ou canary deploy

**Livrables**
- `fly.toml` : `[deploy] strategy = "bluegreen"` (Fly Machines supporte nativement).
- Pour canary plus fin : `flyctl deploy --config fly.toml --strategy canary` (10 % puis 100 %).
- Health-gated rollback : si `/health` < 95 % success sur 5 min après deploy, `flyctl releases rollback` automatique (script `scripts/auto_rollback.sh` lancé en post-deploy GHA).
- `docs/RUNBOOK_DEPLOY.md` : procédure canary + monitoring + rollback.

**Heures** : 2 j (canary plus complexe que rolling — nécessite split DNS via Fly Anycast ou poids `prefer_regions`).
**Coût** : 0 (Fly facture la machine extra pendant le canary, ~$0.50/h × 30 min = $0.25 par deploy).
**Acceptance**
- Un deploy bluegreen avec `bug-introduit-volontairement` (fail healthcheck) ne route AUCUN trafic vers nouvelle version.
- Rollback drill : déploiement bluegreen avec MAUVAISE version, restauration automatique en < 5 min mesurée.

**Dépendances** : MLOPS-02, MLOPS-03.

---

### P1 — MLOPS-08 : Database migrations (Alembic propre)

**Livrables**
- `alembic.ini` + `migrations/` à la racine.
- 1ʳᵉ migration `001_initial_schema.py` capturant l'état actuel des tables SQLite (`signals`, `vol_predictions`, `narrative_cache`, `kill_switch`, etc.).
- `main.py` `_init_db()` remplacé par `alembic upgrade head` au boot.
- CI : nouveau job `db-migration-test` qui applique toutes les migrations sur DB vide et vérifie schema final.
- `docs/MIGRATIONS.md` : workflow ajouter une migration.

**Justification** : actuellement les schemas sont créés à la volée via `CREATE TABLE IF NOT EXISTS` éparpillés dans `src/`. Bascule future SQLite → Postgres (à >100 users payants) sera impossible sans migrations propres.

**Heures** : 1 j.
**Coût** : 0.
**Acceptance**
- `alembic upgrade head` sur DB vide produit le schema identique à l'actuel (diff via `sqlite3 .schema`).
- Un downgrade `alembic downgrade -1` est possible (autogen reverse OK).

**Dépendances** : aucune (peut commencer en parallèle).

---

### P1 — MLOPS-09 : Model train/serve consistency tests (fingerprint features)

**Livrables**
- `src/intelligence/features/` shared module (cf. `eval_23:§6` — 10 features listées).
- `tests/test_train_serve_consistency.py` : pour chaque modèle (scoring_v3_lgbm, factor_model, a1_stack, vol_har, vol_lgbm), assert que `features_fingerprint(train)` == `features_fingerprint(serve)`.
- Corrections explicites des 4 skews train/serve (cf. eval_23:§3) :
  - Skew #1 : session hours hybrid (`colab_hybrid_vol_poc.py:104` 7h → 8h).
  - Skew #2 : currency filter ajouté à `volatility_forecaster.py:743`.
  - Skew #3 : `_calibrate_blend_weight` re-applique cal_mult × regime_mult.
  - Skew #4 : `_fit_lgbm_on_residuals` calcule `har_blended` au lieu de `har raw`.
- `tests/test_model_regression.py` (cf. `eval_23:§5`) : RMSE candidat ≤ 1.02 × baseline.

**Heures** : 2 j (1 j shared module + 1 j tests + corrections).
**Coût** : 0.
**Acceptance**
- 4 skews fermés (PR review-able, RMSE prod 14 j vs Colab dans ±5 %).
- `pytest tests/test_train_serve_consistency.py` passe pour les 5 modèles.
- CI fail si quelqu'un modifie `volatility_forecaster.py` features sans toucher `colab_*.py` (et inversement).

**Dépendances** : MLOPS-04 (manifest avec `features_fingerprint`).

---

### P1 — MLOPS-10 : Image scanning + SBOM en CI

**Livrables**
- CI job `security` :
  - `trivy image --severity HIGH,CRITICAL --exit-code 1 ghcr.io/<owner>/smart-sentinel:<sha>`.
  - `trivy fs --severity HIGH,CRITICAL --exit-code 1 .` (scan requirements.txt + python deps).
  - `pip-audit` pour CVE Python.
  - `bandit -r src/ -lll` pour static-analysis.
  - `syft packages ghcr.io/.../smart-sentinel:<sha> -o spdx-json > sbom.json` puis upload artefact.
- Job `release` (sur tag git) : attache SBOM à la GitHub Release.
- `cosign sign --key-ref doppler://cosign_key ghcr.io/.../smart-sentinel:<sha>` (MLOPS-15).

**Heures** : 0.5 j.
**Coût** : 0 (Trivy/Syft/cosign sont OSS).
**Acceptance**
- Premier run identifie au moins 2-3 CVE HIGH dans torch/transformers (à fixer ou accept-list).
- SBOM SPDX-JSON valide downloadé depuis Release.

**Dépendances** : MLOPS-03.

---

### P1 — MLOPS-11 : Rollback runbook + drill mensuel

**Livrables**
- `docs/RUNBOOK.md` couvrant :
  - Procédure rollback Fly (`flyctl releases rollback <version>`).
  - Procédure rollback model (modifier `manifest.json` `promoted: production` → version précédente, push, redeploy).
  - Procédure rollback DB migration (`alembic downgrade -1`).
  - Procédure suppression user/secret incident.
  - Procédure incident sévère (`flyctl machine stop` + ouverture status page).
- Calendrier mensuel (1ʳᵉ semaine) : drill chronométré, viser MTTR < 10 min.
- `tests/test_rollback_smoke.sh` : script exécutable qui déploie une mauvaise version sur staging, mesure le temps de rollback.

**Heures** : 0.5 j initial + 0.5 j par drill mensuel.
**Coût** : 0.
**Acceptance**
- 1ᵉʳ drill chronométré documenté avec capture d'écran.
- MTTR mesuré < 10 min (cible) ou < 30 min (acceptable au début).

**Dépendances** : MLOPS-02, MLOPS-07.

---

### P1 — MLOPS-12 : Litestream backup SQLite → R2

**Livrables**
- `infrastructure/litestream.yml` : config replicate `/app/data/signals.db` → `s3://sentinel-backups/signals.db` (R2 endpoint).
- Sidecar litestream dans Dockerfile (option a) OU systemd timer sur Fly Machine (option b).
- Backup verification script weekly : restore vers un fichier temporaire et `sqlite3 PRAGMA integrity_check`.

**Heures** : 0.5 j.
**Coût** : $0.5/mo (R2 storage).
**Acceptance**
- `litestream restore` produit une DB identique à la prod (hash row count match).
- RPO (Recovery Point Objective) < 30 s (lag de réplication observable).
- RTO (Recovery Time Objective) < 5 min (restore + redeploy).

**Dépendances** : MLOPS-02.

---

### P2 — MLOPS-13 : Multi-region failover

**Livrables**
- `fly.toml` : `regions = ["cdg", "ams", "iad"]` ; `fly scale count 3 --max-per-region 1`.
- LiteFS (Fly natif) pour réplication SQLite read-only multi-region OU bascule Postgres managed Fly + read replica.
- DNS Anycast Fly géré automatiquement (Anycast natif).
- Drill failover : `flyctl machine stop --region cdg` + vérifier que `ams` ou `iad` prend le relais < 60 s.

**Heures** : 3 j.
**Coût** : +$20-30/mo (2 machines extra shared-cpu-1x + 2 volumes 5 GB).
**Acceptance**
- Kill primary region → traffic ré-routé < 60 s.
- Cohérence eventual SQLite : writes < 5 s lag entre regions.

**Dépendances** : MLOPS-07, MLOPS-12.

---

### P2 — MLOPS-14 : Infrastructure as Code (Terraform fly-apps/fly)

**Livrables**
- `infrastructure/terraform/main.tf` (squelette dans `eval_22:§6`).
- `infrastructure/terraform/variables.tf`, `outputs.tf`, `versions.tf`.
- Backend S3 (Cloudflare R2 S3-compatible) pour tfstate.
- `terraform plan` en CI (advisory, sur PR), `terraform apply` manuel (jamais auto).
- Modules réutilisables : `modules/sentinel_app/`, `modules/observability/`.

**Heures** : 2 j.
**Coût** : 0 (R2 backend gratuit).
**Acceptance**
- `terraform apply` provisionne app + volume + machines + secrets en < 5 min.
- `terraform destroy` puis re-apply = même état (idempotent).
- tfstate stocké chiffré sur R2.

**Dépendances** : MLOPS-02.

---

### P2 — MLOPS-15 : Cosign sign + supply-chain attestation

**Livrables**
- Clé cosign générée, stockée Doppler.
- CI : `cosign sign --key doppler://cosign_key ghcr.io/.../smart-sentinel:<sha>`.
- GitHub attestation `actions/attest-build-provenance@v1` pour SLSA Level 2.
- Vérification cosign au déploiement Fly : `cosign verify --key cosign.pub ghcr.io/.../smart-sentinel:<sha>`.

**Heures** : 0.5 j.
**Coût** : 0.
**Acceptance**
- `cosign verify` retourne OK pour le dernier deploy.
- SLSA provenance JSON attaché à chaque Release GH.

**Dépendances** : MLOPS-10.

---

## 5. Tests & validation

### 5.1 Tests deploy

| Test | Outil | Fréquence | Acceptance |
|---|---|---|---|
| Smoke `/health` post-deploy | curl + retry | Chaque deploy | 200 OK en ≤ 60 s |
| Smoke E2E (1 signal de bout-en-bout) | `tests/test_smoke_e2e.py` | Chaque deploy prod | scanner produit ≥ 1 signal en ≤ 5 min sur fixture XAU |
| Boot time | `time flyctl deploy` | Chaque deploy | < 90 s |
| Restart resilience | `flyctl machine restart` + probe | Hebdo | Up en < 60 s |
| Rollback drill | `scripts/rollback_drill.sh` | Mensuel | MTTR < 10 min |
| Backup restore | Litestream `restore` vers staging | Hebdo | SQLite intégrité OK |

### 5.2 Tests infra-as-tests

- `terraform validate` + `terraform plan` en CI.
- `terratest` Go-based pour Fly app provisioning (P2, optionnel).
- `docker-compose config` validation en CI pour bloc local.
- `actionlint` pour valider syntaxe GH Actions.

### 5.3 Tests modèles (MLOps)

- `tests/test_model_regression.py` : RMSE candidat ≤ 1.02 × baseline (eval_23:§5).
- `tests/test_train_serve_consistency.py` : features fingerprint identique.
- `tests/test_model_load.py` : tous les `.pkl` du manifest se chargent sans erreur.
- `tests/test_drift_monitor.py` : PSI baseline calcule correctement sur fixture.

---

## 6. Sécurité

### 6.1 Image security

- **Trivy** scan en CI (MLOPS-10) : fail sur HIGH/CRITICAL CVE non accept-listed.
- **Distroless base image** : envisager `gcr.io/distroless/python3-debian12` pour réduire surface attack (−40 MB, plus de shell, plus d'apt) — **DEFER P2**, casse `curl healthcheck` (déjà migré Python en MLOPS-06).
- **Pinning hash** : `pip install --require-hashes -r requirements-prod.lock` via `pip-tools compile --generate-hashes` — P1 J+30.

### 6.2 Supply-chain attestation

- **SBOM Syft** (MLOPS-10) : SPDX-JSON attaché à chaque Release GH.
- **Cosign signature** (MLOPS-15) : SLSA Level 2 attestation.
- **GH branch protection main** : status checks requis, no force-push, signed commits requis (à activer dans Settings).
- **Dependabot** : `.github/dependabot.yml` weekly pour Python deps + GH Actions versions.

### 6.3 Runtime security

- **Non-root user** déjà OK (`Dockerfile:78`).
- **Read-only filesystem** : `fly.toml` `[[mounts]]` seuls `/app/data` et `/app/logs` writable. Le reste read-only (`docker run --read-only` flag).
- **Capabilities drop** : `cap_drop = ["ALL"]` sauf `NET_BIND_SERVICE` (FastAPI bind 8000).
- **AppArmor / SELinux** : Fly Machines tournent en gVisor par défaut, isolation kernel forte. Pas d'action requise.
- **Secret mask filter** Telegram/Anthropic dans logs (MLOPS-05).

### 6.4 Network security

- HTTPS forcé via `force_https = true` (Fly.io fournit cert Let's Encrypt auto).
- CORS strict via `CORS_ALLOWED_ORIGINS` env (déjà existant cf. MEMORY.md).
- Rate-limit 100 req/min per-IP déjà en place (cf. MEMORY.md Production Wiring §3).
- Webhook signatures HMAC pour B2B (déjà couvert par eval_24/architecture).

---

## 7. Métriques (DORA)

### 7.1 DORA metrics cibles

| Metric | Cible v1 (J+30) | Cible v2 (J+90) | Elite (benchmark) | Mesure |
|---|---|---|---|---|
| **Deploy frequency** | ≥ 1×/sem | ≥ 1×/jour | Multiple per day | GH Actions `deployments` API |
| **Lead time for changes** | ≤ 1 j (commit → prod) | ≤ 4 h | < 1 h | GHA timestamp PR merged → deploy-prod success |
| **MTTR (Mean Time to Recovery)** | ≤ 30 min | ≤ 10 min | < 1 h | Pager incident → resolved, log dans `runbook.md` |
| **Change failure rate** | ≤ 20 % (rollback ou hotfix dans 24 h) | ≤ 10 % | < 15 % | Compteur rollbacks / total deploys |

### 7.2 Métriques MLOps spécifiques (cf. eval_23:§12)

| Metric | Cible | Mesure |
|---|---|---|
| Train/serve skew count | 0 | grep diff `features/` ↔ `colab_*.py` en CI |
| Time-to-deploy retrain | ≤ 30 min | Colab → PR → CI → merge → restart |
| Regression gate pass rate | ≥ 80 % | `test_model_regression.py` pass over last 10 PRs |
| Drift detection latency | ≤ 24 h | PSI > 0.25 → alert delivered |
| Hybrid-vs-Colab RMSE gap | ≤ 5 % | 30-day rolling prod RMSE / Colab |
| Reproducibility rate | ≥ 95 % | Re-run 5 colab notebooks blind, compare metrics |

### 7.3 Métriques infra

| Metric | Cible | Mesure |
|---|---|---|
| Image size | < 600 MB | `docker image inspect` |
| Image build time (cache hit) | < 90 s | GHA timestamp |
| Image build time (cold) | < 6 min | GHA timestamp |
| Healthcheck success rate | > 99.5 % / 30 j | Fly metrics |
| Cold-start P95 | < 15 s | Fly logs `Starting machine` → first 200 |
| $/mo infra | < $50 (phase test), < $150 (50 users payants) | Fly billing |
| Vulnérabilités HIGH/CRITICAL | 0 (Trivy) | CI report |

---

## 8. Risques & mitigations

| Risque | Probabilité | Impact | Mitigation |
|---|---|---|---|
| **Cloud lock-in Fly.io** | Moyenne | Moyen | Dockerfile portable, IaC Terraform, migration Hetzner planifiée si revenue > $500/mo |
| **Cost overrun (LLM Anthropic + Fly scale)** | Élevée | Élevé | Billing alerts $50/$100/$200 ; `NARRATIVE_MODE=template` par défaut (eval_05) ; rate-limit déjà en place |
| **Drift modèle silencieux** | Élevée | Élevé | Drift monitor PSI cron J+30 (eval_23:§7) + regression gate avant promotion |
| **Secret leakage via logs** | Moyenne | Élevé | SecretMaskFilter (MLOPS-05) + Doppler audit log + rotation 90 j |
| **SQLite corruption sur volume Fly** | Faible | Élevé | Litestream backup R2 (MLOPS-12) + integrity_check hebdo |
| **Volume Fly pinned à 1 machine** | N/A (design) | Moyen | Documenté ; bascule Postgres ou LiteFS si HA requise (>100 users payants) |
| **Train/serve skew après refacto** | Moyenne | Élevé | `test_train_serve_consistency.py` (MLOPS-09) ; shared `features/` module |
| **CI/CD GHA quota dépassé** | Faible | Faible | 2000 min/mo free largement suffisant ; passer Team ($4/mo) si dépassement |
| **Image build flaky (timeout pip torch)** | Moyenne | Moyen | Cache GHA mode=max + buildkit ; supprimer `transformers` si FinBERT non utilisé (eval_22:§10 −500 MB) |
| **MTTR > 30 min sur incident réel** | Élevée (init) | Élevé | Runbook documenté + drill mensuel + on-call rotation solo (Pushover/PagerDuty $10/mo P2) |
| **TESTING_MODE=1 leak en prod** | Moyenne (déjà été le cas) | Critique | `assert_safe_production_config()` (`main.py:76`) + `SENTINEL_TESTING_MODE=0` codé en dur dans `fly.toml` |
| **GHCR registry indisponibilité** | Faible | Moyen | Backup tag image sur Docker Hub mensuel ; rollback peut fonctionner sur image locale Fly |
| **Doppler indisponibilité** | Faible | Moyen | Fallback env vars Fly natives (sync hebdo) ; sealed-secrets si critique |

---

## 9. Dépendances (catégories connexes)

| Catégorie | Dépendance | Direction | Détail |
|---|---|---|---|
| **17. Testing** | Forte | ↔ | CI/CD MLOPS-03 exécute la suite testing ; regression gate MLOPS-09 utilise fixtures testing |
| **16. Observability** | Forte | → | Healthcheck + metrics Prometheus consommés par fly.io probes ; alerting Telegram déjà partagé |
| **15. Security** | Forte | ← | MLOPS-05 (Doppler) + MLOPS-10 (Trivy/SBOM) + MLOPS-15 (cosign) sont des sous-systèmes sécurité |
| **04. Volatility forecasting** | Moyenne | → | MLOPS-09 corrige skews train/serve volatilité (4 skews eval_23) |
| **02. Confluence detector** | Moyenne | → | Model registry MLOPS-04 héberge `scoring_v3_lgbm.pkl` |
| **18. Backtest** | Moyenne | → | Fixtures de regression testing héritées du backtest harness |
| **22. Deployment ops** | Forte | ↔ | Cette catégorie EST le sous-ensemble deployment de l'ops |
| **23. MLOps** | Forte | ↔ | Cette catégorie EST le sous-ensemble MLOps |
| **24. Unit economics** | Faible | → | Coût infra $35-50/mo entre dans marges (eval_24 marges 78-98 % à valider) |
| **29. Compliance** | Moyenne | → | Région EU (cdg) obligatoire GDPR ; audit log Doppler aide SOC2 |
| **08. Data providers** | Moyenne | → | Pipeline data fetch via cron Fly machine (XAU/EUR Dukascopy + FF calendar) |

---

## 10. Estimation totale & timeline

### 10.1 Total effort

| Tranche | Tâches | Effort |
|---|---|---|
| **P0 (Sprint 1 — J+0 à J+10)** | MLOPS-01, 02, 03, 04, 05, 06 | **6 j-h** |
| **P1 (Sprint 2 — J+11 à J+30)** | MLOPS-07, 08, 09, 10, 11, 12 | **6.5 j-h** |
| **P2 (Sprint 3 — J+31 à J+90)** | MLOPS-13, 14, 15 | **5.5 j-h** |
| **TOTAL** | 15 tâches | **18 j-h** |

À raison d'un dev solo founder ~30 h/sem dédié, c'est ≈ **4-5 semaines plein temps** ou **6-8 semaines à mi-temps** combiné aux autres catégories du sprint commercialisation.

### 10.2 Timeline détaillée

```
Semaine 1 (J0-J7) — Sprint 1.A "Unblock deploy"
├── J1   MLOPS-01 (choix hosting, account Fly.io)              0.5 j
├── J1   MLOPS-04 (model registry R2 manifest)                 1.5 j
├── J2   MLOPS-02 (fly.toml prod + staging, kill legacy)       1 j
├── J3   MLOPS-05 (Doppler secrets + SecretMaskFilter)         0.5 j
├── J3   MLOPS-06 (restart on_failure + Python healthcheck)    0.25 j
└── J4-5 MLOPS-03 (CI/CD ci-cd.yml build+test+security+deploy) 2 j
       → Smoke staging green, premier deploy prod manuel OK

Semaine 2 (J8-J14) — Sprint 1.B "Reliability"
├── J8    MLOPS-10 (Trivy + SBOM en CI)                        0.5 j
├── J8-9  MLOPS-11 (Runbook + 1er drill rollback chronométré)  0.5 j
├── J9    MLOPS-12 (Litestream backup R2)                      0.5 j
├── J10-11 MLOPS-08 (Alembic migrations)                       1 j
└── J12-14 MLOPS-09 (train/serve fingerprint + 4 skews fixes)  2 j

Semaine 3 (J15-J21) — Sprint 2 "Blue/green"
├── J15-16 MLOPS-07 (bluegreen + canary + auto-rollback)       2 j
└── J17-21 buffer / drill mensuel / docs

Semaine 4-8 (J22-J56) — Sprint 3 conditionnel
├── MLOPS-14 (Terraform IaC)                                   2 j
├── MLOPS-15 (cosign sign supply-chain)                        0.5 j
└── MLOPS-13 (multi-region) — DEFER tant que < 100 users      3 j
```

### 10.3 Critères go/no-go fin de sprint

**Fin Sprint 1 (J+10) — go/no-go commercial soft launch :**
- [ ] Premier deploy prod via CI/CD vert.
- [ ] `flyctl machine restart` ne perd pas de données (SQLite persiste).
- [ ] Doppler secrets opérationnel, `.env` purgé git history.
- [ ] Healthcheck UP > 99 % sur 24 h.

**Fin Sprint 2 (J+21) — go/no-go onboarding payants :**
- [ ] Blue/green deploy testé sur staging et prod, MTTR < 10 min.
- [ ] Litestream restore validé (drill).
- [ ] 4 skews train/serve fermés, RMSE prod aligné Colab ±5 %.
- [ ] Trivy zero HIGH/CRITICAL CVE non accept-listé.

**Fin Sprint 3 (J+90) — go/no-go scale > 100 users :**
- [ ] IaC Terraform reproduit infra depuis zéro en < 30 min.
- [ ] Cosign + SBOM sur chaque image.
- [ ] Multi-region failover testé (drill, < 60 s recovery).

### 10.4 Coût infra total recommandé

| Composant | Mensuel | Annuel |
|---|---|---|
| Fly.io prod (cdg, 2vCPU/4G + 10G volume) | $35 | $420 |
| Fly.io staging (shared-cpu-1x scale-to-zero) | $3 | $36 |
| Cloudflare R2 (models + backups, ~20 GB) | $1.5 | $18 |
| Doppler (free tier) | $0 | $0 |
| GHCR (repo privé ou public) | $0-4 | $0-48 |
| GH Actions (free 2000 min) | $0 | $0 |
| Sentry/observability (cf. catégorie 16) | hors scope ici | — |
| **TOTAL CATÉGORIE 19** | **~$40-45/mo** | **~$480-520/an** |

Avec montée à 100 users payants (J+180) :
- Fly.io scale 3 machines + multi-region : +$30/mo.
- Postgres managed Fly (si bascule SQLite) : +$15-25/mo.
- Doppler Team : +$15/mo (si > 5 users).
- **TOTAL J+180** : ~$100-110/mo.

---

## Synthèse exécutive

- **Chemin livrable** : `C:\MyPythonProjects\TradingBOT_Agentic\reports\commercialization_sprint\19_mlops_deployment.md`
- **Top 3 P0** : (1) Choix hosting Fly.io + `fly.toml` + `fly.staging.toml` [MLOPS-01/02, 1.5 j] ; (2) CI/CD GitHub Actions build+test+security+deploy [MLOPS-03, 2 j] ; (3) Model registry R2 + manifest JSON + Doppler secrets [MLOPS-04/05, 2 j].
- **Heures totales** : 18 j-h (P0: 6 j, P1: 6.5 j, P2: 5.5 j) ≈ 4-5 semaines plein temps.
- **Coût infra recommandé** : **~$40-45/mo** (Fly prod cdg + staging + R2 backups + Doppler free + GHCR), montée à ~$100/mo à 100 users.
- **Bloqueurs critiques actuels** : aucun fichier deploy versionné (ni Procfile, ni railway.toml, ni fly.toml), 5 modèles `.pkl` à plat sans lineage, CI ne build aucune image, 4 skews train/serve silencieux (~15-25 % RMSE perdus).
