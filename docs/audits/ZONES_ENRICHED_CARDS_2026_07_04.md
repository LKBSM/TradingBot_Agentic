# Page « Zones » — cartes enrichies + « Analyser la zone » (2026-07-04)

Branche : `feat/zones-page-enriched-cards` (worktree dédié, depuis main consolidé `1526049`).
Mission : enrichir les cartes de la page /zones avec des FAITS passés/présents uniquement,
et transformer « Analyser » en « Analyser la zone » (focus graphique à l'arrivée).

Ligne inviolable respectée : descriptif présent/passé uniquement ; aucun score, aucune
« force de zone », aucun classement qualité ; chevauchement multi-TF formulé en géométrie
pure ; dégradation honnête quand la donnée n'existe pas ; **détection inchangée** (aucun
fichier backend touché — tout le calcul d'affichage vit dans le frontend).

---

## 1. Diagnostic — tableau donnée → dispo / manquante → où

| # | Donnée | Verdict | Où / forme exacte |
|---|--------|---------|-------------------|
| a | Prix courant du combo | ✅ Dispo | `header.close_price` (consommé par /zones) ; plus frais : `useLatestPrice` (lecture M15 cache, poll 45 s), le modèle du header /app. **Décision GO : useLatestPrice avec fallback close_price.** |
| b | `fill_level` FVG | ✅ Dispo bout-en-bout | Moteur `_fvg_lifecycle` → `FairValueGap.fill_level` (un PRIX : pénétration max des mèches, clampé dans la bande) → `fillFraction()` front dérive la fraction par géométrie pure (bornes réelles + fill_level réel, jamais une estimation). |
| c | Tests multiples | ❌ Manquant | `tested: bool` + `mitigated_at: datetime\|null` = timestamp du **premier contact uniquement** (`first_tap_idx` OB / `first_entry_idx` FVG, `market_reading_mappers.py:297-401`). Aucun compteur, aucun historique par test. Un compteur serait calculable dans le mapper (même couche que `fill_level`) mais = changement de schéma. **Décision GO : booléen seul — la carte n'affiche jamais « ×N ».** |
| d | Timestamp de formation | ✅ / ⚠️ | `created_at` (ISO) → âge en durée = exact. Âge en **bougies** : pas d'index de barre dans le payload → compté sur la fenêtre de bougies réelles (`useCandles`, 400 bars, cache SQLite). Fenêtre trop courte → durée seule (jamais durée÷TF : weekends/gaps fausseraient le compte). |
| e | Événement structurel de formation | ❌ Manquant | `bos_events`/`choch_events` existent (direction+level+broken_at) mais **aucun lien zone↔événement** dans les faits. Association par proximité temporelle = inférence → **omis de la carte** (conforme mission §1.e). |
| f | Zones des autres TF | ✅ Accessible | Même endpoint `fetchMarketReading(instrument, tf)` — lectures cache-servies, 6 combos toujours chauds. Précédent : `useMtfTrends`. Chevauchement = intersection d'intervalles `[level_low, level_high]`. |
| g | `?focus=` à l'arrivée | ✅ Existait bout-en-bout | `buildAppHref` → `/app?…&focus=` → `AppWorkspace.initialFocusZoneId` → dispatch unique `focus_zone`+`highlight_zone` via `coerceViewActions` (verrou d'id) → `ReadingChart` recadre la fenêtre temporelle sur la zone (prix par autoscale) + highlight. **Réutilisé tel quel, aucun second mécanisme.** Manques comblés : libellé bouton + message honnête sur id périmé. |

## 2. Ce qui a été livré

### A. Carte enrichie (`ZoneLifecycleCard`) — compact d'abord, riche en dépliant
Compact (max 2 accents visuels) :
- **Badge relation au prix** : « prix actuellement dans la zone » (accent ambre — même teinte
  que l'état « en test » du graphe) sinon « à X,X pts au-dessus/en dessous du prix »
  (fond neutre). Recalculé à la fraîcheur des lectures ; omis sans prix utilisable.
- **Âge** : « formée il y a 26 bougies (6 h 30) » — bougies comptées sur la fenêtre de
  candles réelle (`barsSince`), sinon durée seule. « · testée » / « · pénétrée » = fait
  booléen moteur (jamais ×N).
- **Barre de comblement FVG** (accent 2) : inchangée, `fill_level` moteur uniquement.
- En-tête type/direction/état et bornes+importance : inchangés.
- La phrase de narration qui dupliquait l'en-tête (`narrateZone`) est **supprimée**.

Dépliable (chevron « Détails ») :
- **Timeline** complète (événements moteur uniquement) ; les étapes d'interaction portent
  la mention « · premier contact » (honnêteté : `mitigated_at` = premier contact, pas « le » test).
- **Chevauchements multi-TF** : « chevauche un OB H1 haussier (2 375,00 – 2 378,00) » —
  géométrie pure, un TF sœur indisponible n'apporte simplement rien.
- Pas d'événement de formation (lien absent des faits — point e du diagnostic).

### B. « Analyser la zone »
- Bouton renommé ; le deep-link transportait déjà instrument + timeframe + zone_id réel.
- Verrou d'id inchangé (`coerceViewActions` + `validZoneIds`).
- **Nouveau** : à l'arrivée dans /app, si le `?focus=` ne résout pas dans la lecture chargée
  (zone consommée/périmée), bandeau discret « Cette zone n'est plus détectée — la lecture
  actuelle du marché est affichée. » (fermable). L'app s'ouvre normalement sur le combo ;
  la zone n'est JAMAIS redessinée de mémoire.

### C. Tri
- Défaut : **Proximité** (distance du prix à la BANDE, 0 si dedans — le même fait que le badge).
- Options : **Fraîcheur** (created_at desc) / **État** (statut moteur).
- Le tri « Importance » est **supprimé** (classement par force = recommandation implicite).
- Sans prix utilisable, Proximité dégrade vers l'ordre par état (aucune distance inventée).

## 3. Fichiers modifiés (frontend uniquement)

- `webapp/lib/zones/lifecycle.ts` — +`priceRelation`, `barsSince`, `formatDurationShort`,
  `findOverlaps`, `SiblingZone` ; tri `proximity|recency|state` ; −`narrateZone`, −tri importance.
- `webapp/lib/zones/use-sibling-zones.ts` — **nouveau** hook (lectures TF sœurs → zones).
- `webapp/components/zones/ZoneLifecycleCard.tsx` — carte compacte/dépliable, badge relation,
  âge, chevauchements, « Analyser la zone ».
- `webapp/components/zones/ZoneTimeline.tsx` — mention « premier contact » sur les interactions.
- `webapp/components/zones/ZonesWorkspace.tsx` — `useLatestPrice`+fallback, `useCandles`,
  `useSiblingZones`, tris refaits (défaut proximité).
- `webapp/components/app/AppWorkspace.tsx` — bandeau « zone n'est plus détectée » (une seule
  vérification, après le dispatch du focus).
- Tests : `lifecycle.test.ts`, `ZonesWorkspace.test.tsx`, `zone-focus-deeplink.test.tsx`.

Backend : **aucun fichier touché.**

## 4. Cas d'honnêteté testés

- Distance prix↔zone : dedans (bords inclus) / au-dessus / en dessous, mesurée au bord le
  plus proche ; null sans prix (badge omis, jamais deviné).
- `fill_level` affiché = valeur moteur ; pas de barre sans `fill_level` (pas de % inventé).
- Tests multiples non tracés → rien de type « ×N » dans tout le rendu (test texte).
- Âge en bougies : compté sur candles réelles ; fenêtre trop courte / pas de candles /
  timestamp invalide → durée seule (jamais durée÷TF).
- Chevauchement : intersection stricte (bords qui se touchent ≠ chevauchement) ; TF sœur en
  échec → aucun fait inventé.
- Deep-link : id valide → focus + highlight, aucun bandeau ; id inconnu → app normale +
  bandeau « n'est plus détectée », aucune commande de focus ; pas de `?focus=` → aucun bandeau.
- Non-régression texte : aucun « score / confluence / renforc- / fiab- / étoile / classement »
  dans le rendu de la page.

## 5. Validation

- Vitest : **407 tests verts, 43 fichiers** (0 échec).
- `tsc --noEmit` : **0 erreur**.
- `next build` : **OK** — `/[locale]/zones` 7,4 kB (First Load 132 kB), `/[locale]/app` 12,9 kB (170 kB).
