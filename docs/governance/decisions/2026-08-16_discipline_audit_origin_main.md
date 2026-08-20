# Règle permanente — Tout diagnostic s'effectue contre `origin/main` à jour

**Date** : 2026-08-16
**Origine** : incident DATA-1 (un audit produit sur un HEAD détaché à −90 commits a conclu à tort
qu'un correctif de performance mergé — `SENTINEL_INTERACTIVE_SERVE_STORED`, PERF-2, PR #136 — était
absent du code).
**Statut** : **RÈGLE PERMANENTE**

---

## La règle

> **TOUT DIAGNOSTIC S'EFFECTUE CONTRE `origin/main` À JOUR.**
>
> **Première action de toute mission** : vérifier la position du HEAD local et l'écart en commits
> avec `origin/main`.
>
> Si l'écart n'est pas nul, **l'annoncer AVANT de rapporter quoi que ce soit** — un diagnostic sur
> une version périmée produit des conclusions fausses.

## Procédure de démarrage (à exécuter en premier)

```sh
git fetch origin
git rev-parse --short HEAD
git log -1 --format='%h %s' origin/main
echo "retard: $(git rev-list --count HEAD..origin/main) / avance: $(git rev-list --count origin/main..HEAD)"
```

- **Écart = 0** → poursuivre.
- **Écart ≠ 0** → l'annoncer explicitement, et soit se mettre à jour sur `origin/main`, soit auditer
  directement `origin/main` via `git show origin/main:<chemin>` / `git grep <motif> origin/main`.
- **Ne jamais** conclure « absent du code » sur la seule base d'un `grep` du working tree local sans
  avoir confirmé la position vs `origin/main`.

## Piège spécifique à ce dépôt

Le dépôt utilise **de nombreux worktrees** (une branche par terminal). Le working tree principal peut
rester en **HEAD détaché très en retard** pendant que `origin/main` avance de dizaines de commits.
La branche `main` locale peut elle-même être montée dans un autre worktree (`wt-run-main`) et être en
retard. **La source de vérité du « code courant » est `origin/main`, pas le checkout local.**

## Incident de référence

DATA-1, 2026-08-16 : HEAD détaché sur `6cfe8cf` (~PR #120), `origin/main` sur `771ccc0` (PR #156),
écart 90 commits. Le flag `SENTINEL_INTERACTIVE_SERVE_STORED` (commit `944b401`, merge `a464f80`)
existait dans `origin/main` (défaut activé) mais était absent du checkout périmé → conclusion initiale
fausse, corrigée après vérification contre `origin/main`.
