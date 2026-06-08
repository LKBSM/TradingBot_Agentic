"""Conjunctive rules engine — Plan B (pure algorithm, no ML scoring).

Mission
-------
Remplacer la somme pondérée additive du :class:`ConfluenceDetector` par un
**moteur de règles conjonctives** (AND/OR/NOT) sur les détections ICT/SMC
déterministes. Aucune ML — juste de la logique formelle.

Activation
----------
Ce module est activé **conditionnellement** :
- Si le Sprint 4 ML scoring (logistic L1 / LGBM) franchit Brier skill OOS
  > +0.03 → l'algo principal reste sur scoring ML.
- Sinon → bascule sur ce moteur conjonctif (décision user 2026-05-16).

Pourquoi conjonctif
-------------------
1. **Transparent / auditable** : chaque trade a une liste binaire de
   conditions cochées, pas un score opaque.
2. **Vendable B2B** : *"L'indicateur ne fire un signal QUE si TOUTES les
   conditions ICT sont simultanément validées."*
3. **Zéro drift** : pas de ré-entraînement.
4. **Bit-by-bit reproductible** sans seed ni training data.

Le risque c'est la **rigidité** : aucune règle conjonctive n'a forcément
d'edge OOS. On le mesure via le sweep + gates CPCV/DSR/PBO.

Composants
----------
- :class:`Rule` — une condition unitaire évaluable sur des features SMC.
- :class:`RuleSet` — un AND/OR de rules avec verdict booléen.
- :class:`ConjunctiveDetector` — drop-in replacement de
  :class:`ConfluenceDetector` qui retourne un :class:`ConfluenceSignal`
  uniquement si toutes les rules sont satisfaites.

Status
------
Scaffold — implémentation Sprint 4 décision branchée.
"""

from src.intelligence.rules_engine.rules import Rule, RuleSet  # noqa: F401
from src.intelligence.rules_engine.conjunctive_detector import ConjunctiveDetector  # noqa: F401

__all__ = ["Rule", "RuleSet", "ConjunctiveDetector"]
