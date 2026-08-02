# Étendre le calendrier économique à un nouveau marché

> Note d'architecture (NW-4, chantier 6). **Documentation seule — rien n'est
> implémenté ici.** Décrit ce qu'il en coûte d'ajouter un marché au calendrier
> des publications, et le seul angle mort connu du modèle.

## 1. Le rattachement événement → marché est de la configuration pure

Un événement du calendrier est rattaché à un marché **par sa devise**, pas par du
code. La table de vérité est un seul fichier versionné :

`config/event_market_map.json`

```json
{
  "version": 1,
  "markets": {
    "XAUUSD": { "driver_currencies": ["USD"] },
    "EURUSD": { "driver_currencies": ["USD", "EUR"] }
  }
}
```

Règle (écrite dans le fichier lui-même) : un événement est attaché à un marché si
la **devise** de l'événement figure dans les `driver_currencies` de ce marché. Un
événement dont la devise n'est motrice d'aucun marché suivi n'est **pas** rattaché,
donc **pas** affiché — jamais rattaché par défaut.

Le code se contente de lire cette table :

```python
# src/intelligence/calendar_service.py
def attach_markets(self, currency: str) -> List[str]:
    cur = (currency or "").upper()
    return [m for m, drivers in self._market_map.items() if cur in drivers]
```

Le champ `markets` d'un événement est une **liste** : la relation est
plusieurs-à-plusieurs. Le même `us_cpi` (devise USD) s'attache aujourd'hui à
`XAUUSD` **et** `EURUSD` sans être dupliqué.

## 2. Ajouter un marché piloté par une devise déjà couverte = une ligne JSON

Si le nouveau marché est mû par une devise déjà présente au catalogue (USD, EUR),
l'ajout est **purement de la configuration**, sans toucher au code :

```json
"markets": {
  "XAUUSD": { "driver_currencies": ["USD"] },
  "EURUSD": { "driver_currencies": ["USD", "EUR"] },
  "GBPUSD": { "driver_currencies": ["USD", "GBP"] }
}
```

- `GBPUSD` capte immédiatement toutes les publications USD existantes.
- Pour la composante GBP, il faut en plus des **événements** de devise GBP au
  catalogue (donc des entrées `calendar_catalog.json` + un adaptateur de dates
  pour l'organisme émetteur, p. ex. l'ONS). Sans eux, seule la jambe USD s'affiche
  — honnêtement, sans rien inventer.

## 3. Cas concret du Bitcoin (BTCUSD)

Le Bitcoin n'a **pas d'émetteur** : aucun organisme officiel ne publie de
statistique sur lui. Mais des événements officiels et programmés le concernent.

### 3.1 Rattacher les publications américaines existantes — trivial et propre

Une seule ligne suffit :

```json
"BTCUSD": { "driver_currencies": ["USD"] }
```

Aussitôt, **toutes** les fiches USD (décision FOMC, IPC, NFP, PCE…) s'attachent à
`BTCUSD` — **sans duplication** (le champ `markets` est une liste). C'est
exactement le comportement voulu, et le modèle le permet déjà. **Zéro code.**

### 3.2 Échéances de décision SEC — entrent presque telles quelles

Les échéances de décision de la SEC sur les produits crypto sont des **événements
officiels et programmés** (publiés au Federal Register). Elles rentrent dans le
schéma actuel avec **un bémol** :

| Champ | SEC deadline | Remarque |
|---|---|---|
| `organism` | SEC | nouvel organisme → entrée `sources` + adaptateur de dates (ou planning curé) |
| `currency` | USD | s'attache à BTCUSD via la règle existante |
| `scheduled_at` | date connue | OK |
| `series_code` | `null` | pas de série chiffrée |
| `value_unit` / `actual` / `previous` | — | **pas de valeur numérique** : issue binaire (approbation / rejet / report) |

Le schéma **tolère déjà les événements sans valeur** : c'est le cas de la décision
FOMC aujourd'hui (`series_code=null` → état `unavailable`). Une échéance SEC est
donc un événement « moment seul » : **config + petit adaptateur de dates**, pas de
changement de schéma.

### 3.3 Le halving — le seul angle mort du modèle

Le halving est **déterministe et vérifiable sur la chaîne**, sa date est connue à
l'avance. Mais il n'a **aucun émetteur institutionnel**. Or tout le modèle
présuppose un **publieur** :

- `source` / `organism` — qui publie ? *Personne.* C'est un fait de protocole.
- `license_label` / `policy_url` — quelle licence de réutilisation ? *Aucune* au
  sens d'un organisme ; la donnée est la chaîne elle-même.
- `series_code` / `value_unit` / `actual` — *sans objet* (ni série, ni valeur
  publiée).

La **date** s'insère sans peine (le schéma date + `markets` suffit), mais
l'attribution ne mappe pas. C'est le **seul cas** qui demande une **extension de
schéma**, pas juste de la configuration.

#### Ce que coûterait un « type de source déterministe »

Introduire une catégorie de source **non institutionnelle**, dont l'attribution
est la **chaîne** plutôt qu'un organisme :

1. **Schéma** — rendre facultatifs (ou porter une variante) `organism`,
   `license_label`, `series_code`, `value_unit`. Aujourd'hui plusieurs sont déjà
   optionnels ; il faut surtout un `source_kind` explicite : `official` |
   `protocol` (déterministe, vérifiable, sans émetteur). ~½ journée.
2. **Attribution** — pour `source_kind = "protocol"`, l'attribution cite la règle
   de protocole et un moyen de vérification on-chain (hauteur de bloc), pas une
   `policy_url` d'organisme. ~½ journée.
3. **Dates** — un `date_source` déterministe qui calcule le halving par hauteur de
   bloc (pas un fetch réseau : un calcul vérifiable). ~1 journage.
4. **UI** — la fiche d'un événement de protocole n'affiche ni « valeur publiée »
   ni « précédente » (aucune) ; elle affiche la règle et la vérifiabilité. Réutilise
   le rendu « événement sans valeur » déjà en place. ~½ journée.

Aucune de ces étapes ne touche la détection ni le rattachement marché→devise. Le
halving resterait attaché à `BTCUSD` par une devise fictive dédiée **ou**, plus
proprement, par un rattachement direct au marché (extension mineure de la règle
pour les sources `protocol`, qui n'ont pas de devise).

## 4. Résumé décisionnel

| Opération | Coût |
|---|---|
| Nouveau marché mû par USD/EUR (déjà couverts) | **1 ligne de config** |
| Nouveau marché mû par une devise neuve (GBP, JPY…) | config **+** événements catalogue + adaptateur de dates de l'organisme |
| Rattacher les publications US à BTCUSD | **1 ligne de config**, zéro duplication |
| Échéances SEC | config **+** petit adaptateur de dates (événement sans valeur, déjà toléré) |
| **Halving** | **extension de schéma** : type de source `protocol` (~2,5 j), seul cas hors config |

**Conclusion :** l'architecture de rattachement est saine et extensible par
configuration pour tout marché piloté par des devises déjà couvertes. Le seul
angle mort conceptuel est l'événement **sans émetteur** (halving), qui ne rentre
pas dans le modèle « un organisme publie une valeur » et demande un type de source
déterministe.
