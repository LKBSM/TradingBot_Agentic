# Deploy Legacy — Railway

Configurations Railway préservées après bascule Docker (2026-05-16, demande user).

Fichiers archivés :

- `railway.toml.disabled` — config Railway (build via Dockerfile, healthcheck `/health`, volume `sentinel-data` sur `/app/data`).
- `Procfile.disabled` — `web: python -m src.intelligence.main`.

## Pour réactiver Railway

```bash
git mv .deploy_legacy/railway.toml.disabled railway.toml
git mv .deploy_legacy/Procfile.disabled Procfile
```

Puis re-connecter le projet Railway via `railway login` + `railway link`.

## Pourquoi le switch ?

Voir `docs/deployment/docker.md` § "Pourquoi Docker (vs Railway)" — vendor lock-in, contrôle infra, multi-cloud, reproductibilité.
