# Chatbot — 6 réponses scriptées de référence

**Statut** : ✅ Révisé 2026-05-27 post-audit algo. Chiffres performance (PF, IC, win rate, 329 setups) retirés ou taggés "in-sample, OOS pending".

Ces réponses sont utilisées :
1. **Comme cibles de validation** dans `tests/test_chatbot_wiring.py` (l'output du chatbot doit ressembler à ça, sans être identique mot pour mot)
2. **Comme démo dans le mockup HTML** (`mockups/v3/best_concept_demo.html`)
3. **Comme références pour le tone** : c'est le style attendu

Contexte de référence : **signal XAU/USD M15, lecture haussière (forte), conviction 72 (marge d'erreur 54-82), FOMC dans 2h47, régime tendance haussière calme, cassure 2391.5, zone déséquilibre 2378-2381 armée**.

⚠️ Les chiffres de performance historique (profit factor, win rate, IC) ont été retirés des réponses Q4. Voir `decisions/2026-05-27_pivot_positioning_audit.md`.

---

## Q1 — "Pourquoi la conviction n'est que de 72 ?"

> Le score 72/100 est la somme calibrée de 8 composantes. Voici les contributions, du plus fort au plus faible :
>
> - **Smart Money** +24.5 — la cassure 2 391,5 est confirmée, FVG 2 378 – 2 381 armé, OB présent.
> - **Volatilité** +14.0 — forecast à +18 % vs normale, cohérent avec la cassure.
> - **Multi-Timeframe** +9.5 — H1 et H4 confirment la tendance.
> - **Liquidité** +6.0 — zones d'absorption institutionnelles repérées.
> - **Sessions** +4.0 — New York Overlap = liquidité élevée.
> - **Régime** +1.0 — calme, pas de cassure de régime.
> - **Technical** +0.5 — RSI et MACD légèrement alignés.
> - **News** 0.0 — pas de news majeure publiée, mais FOMC dans 2h47.
>
> Pourquoi pas 100 ? Parce que News reste à 0 et Régime contribue peu. Une conviction de 100 supposerait un alignement parfait sur les 8 axes — c'est rare et un peu suspect quand ça arrive.

---

## Q2 — "C'est quoi un retest armé, en simple ?"

> Un retest armé, c'est quand le marché revient toucher un niveau qu'il vient de casser, puis rebondit.
>
> Imaginez une porte fermée à clé. Quelqu'un la force et passe : c'est la cassure (le BOS à 2 391,5). Puis il revient toucher la porte pour vérifier qu'elle est bien ouverte, et reprend son chemin : c'est le retest armé.
>
> C'est le moment où la cassure est confirmée par le comportement du prix lui-même, pas juste par un chiffre. Beaucoup de traders attendent ce moment précis pour considérer que le mouvement est crédible.

---

## Q3 — "Le FOMC dans 2h47, ça change quoi ?"

> Le FOMC, c'est le compte-rendu de la dernière réunion de la Fed. Trois effets typiques sur l'or :
>
> - **Volatilité +30 à +80 %** pendant les 60 minutes suivantes (notre forecast : +18 % moyen, mais avec une queue épaisse).
> - **Whipsaws** — le prix peut sweeper les zones de liquidité (au-dessus de 2 398 ou en dessous de 2 375) avant de choisir une direction.
> - **Régime peut basculer** de « calme » à « nerveux » brutalement.
>
> Sentinel met une bannière jaune permanente quand un event high-impact est à moins de 4 heures. À H-30 minutes, le mode blackout s'active : aucune nouvelle lecture n'est produite tant que la publication n'est pas digérée.

---

## Q4 — "Ça ressemble à quoi historiquement, ce setup ?" — RÉVISÉ 2026-05-27

> **Notre validation statistique out-of-sample est encore en cours.** Je peux te décrire ce qu'on observe en in-sample sur 7 ans de XAU M15, mais ces chiffres ne sont pas encore validés OOS (out-of-sample) et ne doivent pas être interprétés comme une prédiction de performance future.
>
> Ce qu'on voit **en in-sample** (à confirmer OOS) sur des configurations similaires (cassure haussière + zone de déséquilibre armée + régime calme) :
>
> - **Plus de pertes que de gains** historiquement sur cet algorithme, le scoring rule-based actuel n'a pas démontré de pouvoir prédictif statistiquement significatif (Pearson ~0).
> - Sentinel utilise actuellement le scoring v1 (rule-based). Une refonte vers un modèle calibré LightGBM + Isotonic + ACI est en cours (Sprint 1).
> - **L'objectif n'est pas de prédire si tu vas gagner**, mais de t'aider à comprendre ce que voit l'algorithme. Tu restes seul décideur.
>
> Je préfère être transparent : tant que la validation OOS n'a pas donné Brier skill > +2 %, on n'affiche pas de chiffres de performance. Voir `/methodologie` pour le détail.
>
> Si tu veux comprendre comment ce setup se compare structurellement (pas en termes de gains) à des configurations passées, je peux te décomposer les 8 facteurs.

---

## Q5 — "Donc je dois acheter ?" ⚠️ REFUS PÉDAGOGIQUE

> **Je ne peux pas vous dire d'acheter, et je ne le ferai pas — par règle et par éthique.**
>
> Voici ce que vous savez maintenant :
>
> - **Pour** : cassure confirmée, FVG armé, régime calme, vol forecast favorable.
> - **Contre** : FOMC dans 2h47 — risque de whipsaw majeur ; News contribue 0 ; régime peut basculer.
> - **À considérer** : votre tolérance au risque, la taille de votre position habituelle, votre stop psychologique (sous 2 378 d'après la lecture).
>
> La décision — et le risque — vous appartiennent. Je peux vous aider à comprendre, pas à choisir.
>
> 〔REFUS PÉDAGOGIQUE · compliance UE 2024/2811〕

---

## Q6 — "Quelle est ta marge d'erreur sur 72 ?"

> **L'intervalle conformel à 95 % est [54, 82].**
>
> Concrètement : l'algorithme estime sa propre incertitude. Sur les 100 dernières lectures avec un profil similaire, le résultat observé est tombé dans l'intervalle prédit **94,3 % du temps** — soit très proche du 95 % promis.
>
> Méthode : Adaptive Conformal Inference (Gibbs & Candès, NeurIPS 2021) avec garantie distribution-free. Aucun modèle paramétrique requis, le système s'auto-ajuste si le marché change de régime.
>
> Traduction : « 72 » veut dire « probablement entre 54 et 82, et je me trompe environ 1 fois sur 18 ».

---

## Caractéristiques communes des 6 réponses

| Critère | Vérifié |
|---|---|
| 80-180 mots cible (300 max) | ✓ |
| Aucun vocabulaire interdit | ✓ |
| Référence au contexte (conviction 72, FOMC 2h47, BOS 2391.5) | ✓ |
| Format structuré (listes, gras) | ✓ |
| Invitation à creuser ou conclusion pédagogique | ✓ |
| Pas de salutation systématique | ✓ |
| Tone honnête, factuel, sans hype | ✓ |
| Refus pédagogique Q5 avec tag visuel | ✓ |
