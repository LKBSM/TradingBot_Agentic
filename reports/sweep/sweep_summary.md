# Sweep paramétrique state machine — 48 cells

**Mode** : QUICK (30k bars)
**Grid** : enter ∈ [55, 60, 65, 70], exit ∈ [35, 40, 45], confirm ∈ [1, 2]
**Cells avec trades** : 33 / 48
**Cells qui passent les gates** : 0

## Top 20 cells par profit factor

| cell | trades | PF | PF_lo | DSR | PBO | gates |
| --- | --- | --- | --- | --- | --- | --- |
| `xau_m15_E70_X35_C1` | 1 | inf | 0.000 | 0.000 | 0.500 | ❌ |
| `xau_m15_E70_X40_C1` | 1 | inf | 0.000 | 0.000 | 0.500 | ❌ |
| `xau_m15_E70_X45_C1` | 1 | inf | 0.000 | 0.000 | 0.500 | ❌ |
| `xau_m15_E65_X35_C1` | 6 | 3.802 | 0.000 | 0.000 | 0.500 | ❌ |
| `xau_m15_E65_X40_C1` | 6 | 3.802 | 0.000 | 0.000 | 0.500 | ❌ |
| `xau_m15_E65_X45_C1` | 6 | 3.802 | 0.000 | 0.000 | 0.500 | ❌ |
| `xau_m15_E60_X35_C1` | 22 | 0.603 | 0.000 | 0.000 | 0.500 | ❌ |
| `xau_m15_E60_X40_C1` | 22 | 0.603 | 0.000 | 0.000 | 0.500 | ❌ |
| `xau_m15_E55_X35_C1` | 69 | 0.542 | 0.000 | 0.000 | 0.500 | ❌ |
| `xau_m15_E55_X40_C1` | 69 | 0.542 | 0.000 | 0.000 | 0.500 | ❌ |
| `xau_m15_E55_X45_C1` | 70 | 0.489 | 0.000 | 0.000 | 0.500 | ❌ |
| `xau_m15_E60_X45_C1` | 22 | 0.483 | 0.000 | 0.000 | 0.500 | ❌ |
| `xau_m15_E55_X35_C2` | 18 | 0.240 | 0.000 | 0.000 | 0.500 | ❌ |
| `xau_m15_E55_X40_C2` | 18 | 0.240 | 0.000 | 0.000 | 0.500 | ❌ |
| `xau_m15_E55_X45_C2` | 18 | 0.240 | 0.000 | 0.000 | 0.500 | ❌ |
| `eurusd_m15_E55_X35_C1` | 7 | 0.082 | 0.000 | 0.000 | 0.500 | ❌ |
| `eurusd_m15_E55_X40_C1` | 7 | 0.082 | 0.000 | 0.000 | 0.500 | ❌ |
| `eurusd_m15_E55_X45_C1` | 7 | 0.000 | 0.000 | 0.000 | 0.500 | ❌ |
| `eurusd_m15_E60_X35_C1` | 3 | 0.000 | 0.000 | 0.000 | 0.500 | ❌ |
| `eurusd_m15_E60_X40_C1` | 3 | 0.000 | 0.000 | 0.000 | 0.500 | ❌ |

## ❌ Aucune cell ne passe toutes les gates

Le sweep a généré des trades mais aucune configuration ne franchit simultanément DSR ≥ 1.5, PBO ≤ 0.35, PF_lo > 1.0, DM_p < 0.05.
Recommandation : pivot ou actions Sprint 4 (logistic L1 sur composantes).