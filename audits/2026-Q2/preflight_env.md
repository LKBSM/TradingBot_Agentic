# Pre-flight — Environnement Python

**Date** : 2026-05-15
**Batch** : Sprint 0 — 0.0

## Versions runtime

| Composant | Version  |
| --------- | -------- |
| Python    | 3.12.6   |
| pytest    | 9.0.2    |
| pytest-cov | 7.1.0   |

## Libs algo critiques

| Lib            | Version  | Statut       |
| -------------- | -------- | ------------ |
| pandas         | 3.0.0    | ✅ très récent |
| numpy          | 1.26.4   | ✅           |
| scipy          | 1.15.3   | ✅           |
| scikit-learn   | 1.8.0    | ✅           |
| lightgbm       | 4.6.0    | ✅           |
| hmmlearn       | 0.3.3    | ✅           |
| pydantic       | 2.11.7   | ✅           |
| pydantic_core  | 2.33.2   | ✅           |
| fastapi        | 0.129.0  | hors algo    |
| matplotlib     | 3.10.8   | ✅           |
| arch (GARCH)   | **non installé** | ⚠️ Warning runtime au load de `risk_manager.py`. Fallback vol activé. |

## Notes

- **`arch` manquant** : `src/environment/risk_manager.py:14` émet un `UserWarning` au load. Le fallback est utilisé. Action : décider Sprint 1 si on installe `arch` ou si on retire le fallback. Logged dans `OUT_OF_SCOPE.md` (mineur).
- **pandas 3.0.0** : extrêmement récent. Le script `audit_xau_coverage.py` a déjà rencontré une deprecation (`Timestamp.utcnow`). Surveiller les autres modules au backtest.
- **Python 3.12** : OK avec l'ensemble du stack.

## Verdict

✅ Environnement opérationnel. Pas de bloquant pour Sprint 0.
