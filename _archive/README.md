# `_archive/` — Convention d'archivage M.I.A. Markets

Ce dossier contient du code et des documents **désactivés** mais conservés pour traçabilité historique. Rien ici n'est exécuté en production, importé par le pipeline actif, ni considéré comme source de vérité.

## Pourquoi ce dossier existe

Le repo a connu un pivot stratégique majeur le **2026-05-27** :

> **Vision A** (origine 2025-Q4 → 2026-Q1) : bot de trading RL autonome (PPO + multi-agent + curriculum/ensemble/meta/EWC). Stack institutionnelle "TradingBOT Agentic" — modèle qui prend des décisions d'exécution.
>
> **Vision B** (depuis 2026-Q1, formalisée 2026-05-27) : **M.I.A. Markets** — indicateur de marché conversationnel (Multi-asset Intelligence Assistant for Markets). L'IA explique, ne décide pas. Pipeline déterministe (SmartMoneyEngine → ConfluenceDetector → CalibratedConviction → InsightAssembler) + chatbot Sentinel comme moat. Aucune exécution automatique, aucun signal binaire BUY/SELL.

Le pivot est documenté dans :
- `docs/governance/decisions/2026-05-27_pivot_positioning_audit.md`
- `docs/governance/AUDIT_ALGO_2026_05_27.md` (Verdict 3/10 sur scoring RL — scénario C réparation puis pivot B éducatif honest)
- `docs/architecture/MIA_MARKETS_ARCHITECTURE.md` (architecture cible 12-18 mois)
- `PROJET_VISION_INDICATEUR_CHATBOT.md` (document fondateur Vision B, racine repo)

Le code RL stocké ici **n'est pas mort techniquement** (les tests passaient au moment de l'archivage), il est **stratégiquement obsolète** : la roadmap M.I.A. Markets ne le réactive jamais.

## Convention de nommage

```
_archive/
├── README.md                           ← ce fichier
└── <YYYY-MM-DD>_<motif-court>/
    ├── ARCHIVE_NOTE.md                 ← raison + contexte + ce qui aurait pu être réutilisé
    ├── <chemin/d/origine/préservé>/
    │   └── fichier.py
    └── ...
```

**Règles** :

1. **Sous-dossier daté** : `YYYY-MM-DD` correspond à la date du `git mv` (pas à la date du dernier commit du contenu).
2. **Motif court** : 1-3 mots kebab-case (`pivot_vision_b`, `legacy_dashboards`, `experimental_garch`, …).
3. **Préservation de l'arborescence d'origine** : si on archive `src/training/meta_learner.py`, il devient `_archive/2026-05-27_pivot_vision_b/src/training/meta_learner.py`. **Ne pas aplatir.** Cela permet à `git log --follow` de tracer l'historique.
4. **`ARCHIVE_NOTE.md` obligatoire** dans chaque sous-dossier : explique pourquoi, ce qui est dedans, ce qui aurait pu être récupéré et a été refait propre ailleurs.
5. **Jamais d'éditions** sur les fichiers archivés. Si une réutilisation est nécessaire, copier dans le repo actif et adapter, **ne pas** reactiver depuis `_archive/`.

## Ce qui ne va PAS dans `_archive/`

- ❌ **Historique précieux à consulter** (rapports d'audit, méthodologies, plans dépassés mais référencés) → va dans **`docs/archive/`** (visible, classifié, indexé).
- ❌ **Code mort temporaire** (commit de WIP cassé, refactor en cours) → ne va pas dans le repo du tout, reste sur une branche.
- ❌ **Données** (CSV, JSON de replay, modèles `.pkl`) → vont dans `data/_archived/` ou `models/_archived/`.

`_archive/` est réservé au **code source désactivé pour raison stratégique**, avec preuve documentée du choix.

## Vérification d'intégrité (rappel pour future-Claude)

Avant tout `git mv` vers `_archive/` :

1. **Imports croisés** : `grep -r` depuis le code actif (`src/intelligence/`, `src/api/`, `src/delivery/`, `webapp/`) vers le module à archiver. Doit être **vide** ou wrappé en try/except avec dégradation gracieuse.
2. **Tests** : tout test qui importe le module va dans l'archive avec lui.
3. **Patches collatéraux** : si un module conservé garde une référence (try/except, message d'erreur en string, bloc `__main__`), patcher dans le **même** PR.
4. **Gate tests** : `pytest tests/` doit rester ≥ 1 200 verts, 0 ImportError nouveau.

## Restauration (en cas d'erreur)

```bash
git mv _archive/<dossier-daté>/<chemin>/<fichier> <chemin>/<fichier>
git commit -m "revert(archive): restore <fichier> (motif)"
```

L'historique git est préservé par `git mv`, donc `git log --follow <fichier>` continue de fonctionner après archivage et après restauration.

---

**Mainteneur** : Loukmane Bessam (solo founder).
**Dernière mise à jour de la convention** : 2026-05-27.
