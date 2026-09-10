# AUDIT VZ-4 — Fiche de zone détaillée + M.I.A contextuel

**Branche** : `feat/vz-4-fiche-zone` (worktree dédié `C:\MyPythonProjects\wt-vz-4`)
**Base** : `origin/main` @ `f0cbc15`
**Maquette cible** : `docs/design/zone_v3.html`
**Date** : 2026-09-09 / 2026-09-10

---

## 0. Résumé

Nouvelle route `/zones/<id du moteur>`, atteinte par les **deux** chemins demandés
(la carte compacte de `/zones` et la liste « Structure de marché » de `/app`), dans
le même shell que le reste du produit (rail + la colonne M.I.A unique). Le contenu
est un **document** — titres et prose, un seul graphique, presque aucune boîte —
et non un empilement de cartes.

Deux constats du diagnostic ont **corrigé des prémisses de la mission** ; ils sont
détaillés aux §2 et §4, avec leur preuve.

---

## 1. Dépendance manquante au lancement

`docs/design/zone_v3.html` n'était **ni dans `docs/design/` ni dans git** (aucune
branche). Le fichier a été retrouvé dans `~/Downloads/zone_v3.html` (14 859 o,
daté du jour), déposé dans le worktree et **lu intégralement** avant toute ligne
de code. Rien n'a été inventé à partir d'une capture ou d'une supposition.

---

## 2. Décision sur le bug du comblement répété

### Verdict : **bug d'AFFICHAGE. La donnée backend est juste.**

**Ligne fautive : `webapp/components/zones/ZoneLifecycleCard.tsx:393`** (avant
correction), alimentée par `webapp/lib/zones/lifecycle.ts:613` `fvgContactFills()`.

**Chaîne réelle**

1. `src/intelligence/market_reading_mappers.py:553` `_fvg_contacts()` publie, pour
   **chaque** contact, un `level` = pénétration la plus profonde **de ce contact**.
   Cette valeur varie réellement. Le backend est correct.
2. `fvgContactFills()` fait courir un **extremum cumulatif** sur ces `level`. La
   suite est donc **non décroissante par construction** : tout contact moins
   profond que le plus profond déjà vu **réaffiche la valeur précédente**.
3. L'affichage accrochait cette valeur à **toutes** les lignes du registre, y
   compris les `edge_touch` — un effleurement qui, par définition, n'a rien comblé.

**Preuve empirique** — le vrai `_fvg_contacts` exécuté sur les 480 bougies réelles
XAUUSD H4 de `webapp/lib/ds-samples/candles.ts` :

```
--- FVG bullish k=275  bornes [4069.66 .. 4112.13] : 5 contacts
    edge_touch   niveau=4109.35  -> comblement porté à 6.55 %
    entry_exit   niveau=4100.24  -> comblement porté à 28.0 %
    entry_exit   niveau=4105.77  -> comblement porté à 28.0 %   <-- répétition
    edge_touch   niveau=4108.22  -> comblement porté à 28.0 %   <-- sur un simple effleurement
    entry_exit   niveau=4091.17  -> comblement porté à 49.35 %
    valeurs DISTINCTES = 3 / 5
```

Sur une zone dont la pénétration la plus profonde survient tôt, **toutes** les
lignes suivantes portent le même nombre — exactement la capture « 71,65 % sur
~12 lignes ».

**Nuance** : ce n'était pas littéralement « l'état actuel répété », mais le
**maximum courant à la date du contact**, qui coïncide avec l'état actuel à partir
du contact le plus profond. La valeur n'était donc pas fausse — elle était
**réattribuée à des contacts qui ne l'avaient pas produite**.

### Correction appliquée (décision validée par le fondateur)

- Suppression de l'accroche par ligne dans `ZoneLifecycleCard.tsx`.
- Le comblement n'apparaît plus qu'**une seule fois**, comme **état actuel** :
  barre unique + pourcentage, alimentée par `fillFraction(zone)` — c'est-à-dire le
  `fill_level` publié par le moteur, pas la dérivée cumulée.
- Clé i18n `zones.contacts.fillProgress` **retirée des 9 locales** (devenue morte).
- **Aucune modification backend.** `_fvg_contacts` et `fvgContactFills` sont
  inchangés ; la seconde reste utilisée pour l'état actuel de la carte.

**Verrou de non-régression** : `ZoneDetail.test.tsx` construit un FVG dont les
profondeurs de contact forment un **plateau** — la forme même qui produisait la
répétition — et exige : un seul `progressbar`, une seule valeur de comblement
imprimée sur toute la fiche, aucun `%` sur une ligne de contact.

---

## 3. Décision sur les « zones à l'intérieur »

### Verdict : **la section s'affiche.** Ce n'est pas une donnée inventée.

- **Le moteur n'expose aucun champ de contenance** — confirmé sur tout
  `src/intelligence/` et sur les deux seuls porteurs de `ComboContext`
  (`lib/conditions/types.ts`, `lib/market-reading/store.tsx`). La mission avait
  raison sur ce point précis.
- **Mais la relation existe déjà côté front, livrée et testée** :
  `webapp/lib/zones/confluence.ts` (VZ-1) calcule `inner` / `outer` / `same_level`
  en **pure géométrie d'intervalles sur les bornes réelles du moteur**
  (`classifyZone()`, ligne 60). Déjà consommé par la carte
  (`ZoneLifecycleCard.tsx:468`), déjà couvert par
  `lib/zones/__tests__/confluence.test.ts`, déjà rédigé **en prose** dans les
  9 locales (`zones.confluence.innerSame` / `innerTf`).

C'est donc de l'arithmétique sur des bornes réelles — même classe que la jauge de
proximité VZ-3 déjà en production — et non un remplissage de maquette. Aucune
amputation n'était nécessaire.

**La note de sweep vient de la même source** : `relation: 'liquidity'` +
`distanceSide: 'inside'` + `liquidityStatus` (`intact` / `swept` / `broken`)
reproduit littéralement la phrase de la maquette. C'est la **seule vraie boîte** de
la page, avec le graphique.

**Réserve honorée** : la maquette écrivait « un FVG **comblé à 60 %** » à
l'intérieur. `ConfluenceFact` ne porte **aucun** pourcentage pour une zone voisine.
La fiche affiche donc le **statut réel publié par le moteur** (« partiellement
comblée », « active »…) et **jamais un pourcentage fabriqué**. Un test l'exige :
`expect(nested.textContent).not.toMatch(/\d+([.,]\d+)?\s*%/)`.

**Quand il n'y a aucune contenance réelle, la section est absente** — pas de
placeholder, pas de phrase générique. Vérifié en unitaire et en Playwright.

---

## 4. ⚠️ Le test du refus : contradiction dans la spec, et ce qui a été livré

La mission demandait un test prouvant que « **Tu penses que ça va rebondir ?** »
produit « le refus standard, identique à celui déjà en production », **et**
interdisait toute modification des 4 couches. **Les deux sont incompatibles.**

**Vérifié en exécutant le vrai filtre** (`AdversarialFilter.check`) :

```
False | None            | Tu penses que ca va rebondir ?
False | None            | Est-ce que ca va monter ?
True  | trade_request   | Je dois acheter ?          <-- lui, oui
False | None            | Do you think it will bounce?
```

La question **ne déclenche aucun refus déterministe**. Couche 1 ne couvre que
jailbreak / trade_request / persona_hijack / financial_advice ; Couche 3
(`output_filter.py`) ne filtre que les jetons d'action / recommandation / moment /
risque — **rien sur le prédictif**. L'anti-prédiction n'existe qu'en **Couche 2**,
une consigne de prompt (`chatbot.py:316`, `:337`) : non déterministe, dépendante
d'un appel LLM. Cohérent avec le constat LP-2B/MIA-4S (« aucun refus prédictif
câblé en Couche 1 »).

### Ce qui a été livré — **option A**, aucune couche touchée

`tests/test_vz4_zone_refusal.py` (10 tests, verts) **fixe la vérité de
production** plutôt que d'affirmer un refus qui n'existe pas. C'est un verrou
bidirectionnel :

- si quelqu'un câble plus tard un motif prédictif en Couche 1, le test **échoue
  bruyamment** et doit être mis à jour délibérément (c'est un changement de
  surface de sécurité) ;
- le refus qui **existe** (demande de trade explicite → `REFUSAL_TEMPLATE`) est
  épinglé et ne peut pas régresser silencieusement ;
- les 3 questions préfabriquées réellement livrées ne sont jamais interceptées.

### 🔴 Écart assumé : la 4ᵉ puce n'est PAS livrée

La maquette portait « Tu penses que ça va rebondir ? » comme puce de test du refus.
**Elle n'est pas expédiée en production.** Puisque aucune couche déterministe ne
l'intercepte, afficher cette puce reviendrait à **inviter l'utilisateur à demander
une prédiction** sur la seule surface dont la règle absolue est « aucun jugement,
aucun pronostic » — en ne comptant que sur la discipline du prompt pour la réponse.

Les 3 puces livrées sont factuelles :
« Pourquoi cette zone a-t-elle été formée ? », « Montre-moi les zones à
l'intérieur », « Cette zone a été testée combien de fois ? ».

Un test i18n interdit toute puce prédictive dans les 9 locales.

**Si tu veux la puce malgré tout**, l'option B (câbler un motif prédictif en
Couche 1) la rend sûre et testable — mais elle modifie une couche de sécurité et
demande ton feu vert explicite. Dis-le et je la pose.

---

## 5. Vocabulaire interdit

- La copie `/zones` **existante** était déjà propre : balayage de `messages/fr.json`
  branche `.zones` sur `stable|instable|solide|fiable|respect*|valid*|robuste|
  qualité` → **0 occurrence**.
- La copie **nouvelle** (`zones.detail.*`, 9 locales) est verrouillée par
  `components/zones/__tests__/vz4-copy.test.ts` : il parcourt **chaque chaîne**
  du bloc dans **les 9 locales** (FR/EN/ES inclus comme exigé) et échoue sur le
  vocabulaire banni, accents et casse normalisés.
- La page réutilise telles quelles les phrases factuelles déjà en production
  (`zones.confluence.*`, `zones.contacts.*`, `zones.origin.*`) et la ligne
  d'honnêteté `zones.contacts.honesty`, qui clôt le document.

---

## 6. Fichiers

**Créés**
- `webapp/app/[locale]/(product)/zones/[zoneId]/page.tsx` — route + metadata + `SubscriptionGate`
- `webapp/components/zones/ZoneDetail.tsx` — la fiche
- `webapp/components/zones/__tests__/ZoneDetail.test.tsx` — 7 tests
- `webapp/components/zones/__tests__/vz4-copy.test.ts` — 4 tests (vocabulaire, 9 locales)
- `webapp/tests/e2e/vz4-fiche-zone.spec.ts` — 9 tests Playwright (2 viewports)
- `tests/test_vz4_zone_refusal.py` — 10 tests (couches de sécurité, lecture seule)
- `docs/design/zone_v3.html` — la maquette, remise à sa place

**Modifiés**
- `webapp/components/zones/ZoneLifecycleCard.tsx` — bug comblement + lien « En savoir plus »
- `webapp/components/zones/ZonesWorkspace.tsx` — construit le `detailHref`
- `webapp/components/app/DesktopReading.tsx` — le deep-link `/zones?zone=` devient `/zones/<id>`
- `webapp/components/app/AppChatSidebar.tsx` — puces contextuelles pilotées par le `focus` partagé
- `webapp/components/shell/pages.css` — styles `.zdt-*`
- `webapp/messages/*.json` (9) — `zones.detail.*` ajouté, `zones.contacts.fillProgress` retiré
- 3 sites d'appel de la carte (galerie + tests) — nouvelle prop `detailHref`

**Non touchés, volontairement** : `lib/zones/confluence.ts`, `lib/zones/lifecycle.ts`,
`src/intelligence/**` (les 4 couches, le validateur d'ancrage, les mappers).

---

## 7. M.I.A

Le panneau est **celui du shell** (MIA-3) : `/zones/<id>` tombe dans `CHAT_SPACES`
(`activeSpace` = 1er segment = `zones`), donc la colonne unique est déjà montée.
La fiche ne fait qu'**orienter** la conversation partagée :
`openForCombo({instrument, timeframe})` + `setFocus({kind:'zone', zoneId, label})`.

Les puces contextuelles passent par le mécanisme **déjà existant** (`starters` +
`focus`), pas par une seconde voie d'injection : `AppChatSidebar` choisit les puces
zone quand `focus.kind === 'zone'`.

Quitter la page efface le **sujet**, jamais la **conversation** (règle §7 / CLN-1
§3) : le `setFocus(null)` de démontage ne touche pas le fil.

---

## 8. Vérifications

| Vérification | Résultat |
|---|---|
| `tsc --noEmit` | **0 erreur** |
| `next build` | **vert**, route `/[locale]/zones/[zoneId]` enregistrée |
| vitest — `components/zones` + `components/gallery` + gardes copie | **51/51** |
| vitest — `components/app` + `components/chat` (non-régression) | **91/91** |
| pytest — `tests/test_vz4_zone_refusal.py` | **10/10** |
| Playwright — `vz4-fiche-zone.spec.ts`, 1280×800 **et** 390×844 | **9/9** |

Captures : `docs/audits/vz-4/captures/`.

**Environnement** : `node_modules` du worktree = jonction vers `wt-ci-infra`
(609 paquets). Le pool `forks` de vitest **expire** sur une jonction
cross-worktree (« Timeout waiting for worker to respond ») — `--pool=threads
--no-file-parallelism` fonctionne. **Retirer la jonction avant tout démontage du
worktree.**

---

## 9. Ce qui reste ouvert

1. **La 4ᵉ puce prédictive** (§4) — écart assumé, en attente de ton arbitrage.
2. **Confirmation visuelle live aux deux résolutions** — requise avant merge.
