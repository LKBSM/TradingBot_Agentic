# Audit qualité de détection — revue lecture seule (2026-06-12)

**Branche** : `audit/detection-quality-review` · **Périmètre** : pipeline /app (MarketReading) — OB, FVG, BOS, CHOCH, Phase/Régime, News
**Discipline** : AUCUNE modification de code, de seuil ou d'heuristique. Diagnostic + cartographie uniquement.
**Sources de preuve** : code (`src/environment/strategy_features.py`, `src/intelligence/market_reading_*`, `src/intelligence/data_providers/twelve_data_provider.py`, `src/intelligence/news_pipeline.py`, `src/api/bootstrap.py`), bases live `data/market_readings.db` (121 lectures, dont 45 XAUUSD M15) et `data/candles.db` (1 688 bougies), reproduction du moteur en mémoire (lecture seule).

---

## 0. Résumé exécutif

Le « manque d'analyse » perçu est un **mélange identifié et quantifié** :

1. **2 bugs transverses prouvés** (corrigeables maintenant, sur preuve) : timestamps du provider décalés de **+10 h** stockés comme UTC, et analyse exécutée sur la **bougie en cours de formation** (signaux non finaux, lectures non reproductibles).
2. **1 manque structurel dominant** : le pipeline ne regarde que **la dernière bougie** (`market_reading_assembler.py:118`) → cardinalité maximale de **1 OB et 1 FVG par lecture**, durée de vie 1 bougie, aucun suivi ni invalidation. Sur la fenêtre de 200 bougies du cas flaggé, le moteur calcule pourtant **40 patterns OB et 34 FVG** ; le produit en montre 0 ou 1, choisi par récence et non par importance.
3. **Des écarts de définition** (le moteur suit sa règle, mais la règle est discutable) : OB = simple pattern 2-bougies (pas ICT), importance = taille de bougie/ATR, phase/régime = heuristique fenêtre fixe 200 bougies, seuil FVG 0,1 ATR. **À ne pas toucher avant l'annotation founder.**

Le cas flaggé du 12 juin (OB 4185-4193 sous prix 4190-4196) est **tracé à 100 % et reproduit au centime** : le moteur a suivi sa règle ; le résultat choque parce que la règle marque la bougie précédente, pas une zone institutionnelle.

---

## 1. Constats transversaux (s'appliquent à tous les détecteurs)

### T1 — Pipeline mono-bougie : la cardinalité est structurellement ≤ 1

- Le moteur (`SmartMoneyEngine.analyze()`) calcule des **colonnes par barre** sur tout l'historique (200 bougies, `MarketReadingAssembler.DEFAULT_LOOKBACK`, `market_reading_assembler.py:142`).
- L'assembleur ne lit que la **dernière ligne** : `last_row = enriched.iloc[-1]` (`market_reading_assembler.py:118`), complétée par `realized_levels(enriched, idx=len-1)` (`market_reading_assembler.py:131`).
- Le mapper construit des listes de **longueur ≤ 1** : un OB seulement si le pattern a tiré sur la dernière barre (`market_reading_mappers.py:311-336`), un FVG idem (`:339-358`).
- Conséquences mesurées (45 lectures XAUUSD M15 en base) : OB présent dans 13 lectures, FVG 9, BOS 22 (grâce à la persistance D1-b), **CHOCH 1 seule** (durée de vie = 1 bougie).
- Le **schéma produit est déjà prêt** pour le multi-zones : `order_blocks` / `fair_value_gaps` sont des listes, avec vocabulaire de cycle de vie (`OBStatus = active|mitigated|invalidated`, `FVGStatus = active|partially_filled|filled`, `tested`, `market_reading_schema.py:41-42,89-109`). **Aucun producteur n'émet jamais autre chose que `active` / `tested=false`.**

**Ce qu'il faudrait pour suivre/surfacer plusieurs zones** (Groupe B, gated annotation) : un registre de zones balayant les 200 barres (le moteur produit déjà tous les candidats), avec règles d'invalidation (zone consommée : prix a traversé / comblée : gap rempli), critère d'importance pour le tri, et top-N par TF. Aucune modification du moteur de détection n'est requise pour la partie « collecte » : les événements sont déjà dans les colonnes.

### T2 — BUG prouvé : timestamps du provider décalés de +10 h, stockés comme UTC

- `TwelveDataProvider._fetch_dataframe` n'envoie **pas** de paramètre `timezone` à l'API (`twelve_data_provider.py:229-235`) ; TwelveData renvoie alors l'heure locale de l'exchange. `_parse_time_series` tague aveuglément ces timestamps `utc=True` (`twelve_data_provider.py:299`).
- **Preuves empiriques** (offset = +10 h 00 exactement, cohérent avec un fuseau UTC+10 type Sydney/AEST) :
  - `candles.db` contenait à ~15:00 UTC le 12 juin des bougies étiquetées jusqu'au **13 juin 00:45** ;
  - la lecture XAU M15 `candle_close_ts=2026-06-12T14:00Z` (`created_at` réel : 14:10:44 UTC) a pour dernière barre la bougie étiquetée `2026-06-13T00:00` → 14:10:44 + 10 h = 00:10:44, barre en cours 00:00-00:15 : alignement exact ;
  - **timestamps de cassure dans le futur publiés en production** : lecture EURUSD M15 du 11 juin 00:30Z → `broken_at=2026-06-11T10:00:00Z` (9 h 30 après la lecture) ; lecture XAU M15 01:30Z → `broken_at=11:00Z`. Cause : `broken_at` d'une cassure persistée vient de `BOS_BREAK_TS`, lu sur l'index des bougies décalées (`market_reading_mappers.py:175-179, 284`).
- **Régimes d'horloge mélangés dans un même payload** : `created_at`/`candle_close_ts` = vrai UTC (horloge serveur), `broken_at` persisté = UTC+10 (index bougies), news = vrai UTC (UoM publié 14:00Z ✓). Tout affichage temporel croisé est incohérent.

### T3 — BUG prouvé : analyse sur bougie en formation + bougies partielles écrasées en cache

- TwelveData renvoie la bougie **en cours** ; l'assembleur la traite comme close (la lecture étiquetée « close 14:00 » a été générée à 14:10:44 sur une barre ouverte depuis 10 min, `close_price=4194.65` ≠ close final 4192.24).
- Conséquences observées en base :
  - la **même zone OB** [4185.50-4192.87] publiée deux fois (lectures 14:00Z et 14:15Z) sous **deux ids différents** (`OB_20260612140000`, `OB_20260612141500`) — aucune identité de zone, le pattern a re-tiré sur la même paire de bougies vue à travers le lag du feed ;
  - la lecture 14:45Z (`close_price=4185.25`) était **incompatible avec le snapshot des bougies** au moment de l'audit, puis est devenue cohérente quand le cache a réécrit la bougie partielle : la barre étiquetée 00:30 est passée de `O=4194.23/L=4190.06/C=4190.71` à `O=4192.28/L=4181.48/C=4184.21` entre deux requêtes (cache `candles_cache` écrasé par upsert à chaque fetch, `market_reading_assembler.py:233`).
- **Conséquence d'audit** : sans snapshot brut du provider, une lecture n'est **pas reproductible** a posteriori (les signaux peuvent « repeindre » : une bougie partielle haussière peut finir baissière). Le sens marginal d'une bougie (corps 0,98 $ sur la barre 00:15) décide du déclenchement OB alors que nous avons observé des révisions de plusieurs dollars sur ces mêmes barres.

---

## 2. Par détecteur

### 2.1 Order Block

**A. Définition actuelle** (`strategy_features.py:756-815`) :
- OB haussier = `close[i-1] < open[i-1]` (bougie précédente baissière) ET `close[i] > open[i]` (bougie courante haussière) ET `high[i] > high[i-1]` (`:766-770`). Symétrique pour baissier (`:772-776`).
- Zone = **range complet (high/low) de la bougie i-1** (`:791-794`).
- Force = `(high[i-1]-low[i-1]) / ATR` + bonus 0,2 si un FVG a tiré sur la barre précédente (`OB_REQUIRE_FVG=False` par défaut, `OB_FVG_BONUS=0.2`, `:807-815` ; config `:483-491`).
- Côté produit : `importance = high si force ≥ 0.75, medium ≥ 0.4, low sinon` (`market_reading_mappers.py:319`) ; direction = celle du signal de confluence sinon le sens du trend BOS (`:315-318`) ; `status="active"`, `tested=False` codés en dur (`:326-336`).

**B. Cardinalité** : ≤ 1 par lecture, uniquement si le pattern tire sur la **dernière barre** (cf. T1). Durée de vie produit = 1 bougie (re-émis avec un nouvel id si re-détection, jamais suivi). Le pattern tire sur **23 % des barres** (40 fois dans la fenêtre de 200 du cas flaggé) — il n'y a ni tri d'importance entre candidats, ni mémoire, ni invalidation. Pour en surfacer plusieurs : registre + invalidation (T1).

**C. Comportement observé — cas flaggé** : voir §3 (trace complète, reproduite au centime).

**D. Écart vs SMC standard** :
- Ce n'est pas un OB ICT : pas d'ancrage à une cassure de structure, pas d'exigence d'impulsion/displacement, pas de balayage de liquidité, pas de « dernière bougie contraire avant le move ». C'est un pattern « bougie contraire suivie d'un plus haut/plus bas » — même pas un engulfing complet (aucune condition sur les corps). Connu et documenté (P0-2, `src/intelligence/smart_money/__init__.py:53`).
- « Importance » = taille de la bougie précédente relative à l'ATR (+ bonus FVG) — aucun rapport avec l'importance SMC (zone non mitigée, origine d'impulsion, confluence HTF).
- Un trader marquerait **plusieurs** OB par chart (zones d'origine des impulsions, non consommées) ; le produit en montre 0 ou 1, toujours collé au prix courant (c'est la bougie précédente).

**E. Classification** :
- Définition du pattern, zone, importance : **ÉCART DE DÉFINITION** (le code suit sa règle ; la règle est à trancher par annotation).
- Cardinalité/absence de suivi : **ÉCART structurel** (Groupe B, gated annotation).
- Détection sur bougie partielle (T3) et horodatage +10 h (T2) : **BUG**.

### 2.2 Fair Value Gap

**A. Définition** (`strategy_features.py:646-679`) : gap 3-bougies standard — haussier si `low[i] > high[i-2]`, baissier si `high[i] < low[i-2]` ; taille normalisée par ATR ; signal émis si `taille > FVG_THRESHOLD=0.1 ATR` (`SMCConfig:478-481`). Bornes réelles publiées via `realized_levels` (`market_reading_mappers.py:192-204`).

**B. Cardinalité** : ≤ 1, uniquement si la dernière barre est la 3ᵉ bougie d'un gap → **durée de vie produit = 1 bougie** (9/45 lectures XAU M15). 34 FVG détectés dans la fenêtre de 200 barres du cas flaggé. Aucun suivi de comblement : `status="active"` et `tested=False` en dur (`market_reading_mappers.py:349-358`) alors que le schéma prévoit `partially_filled`/`filled`.

**C. Comportement observé** : lecture XAU H1 du 11 juin 00:00Z : `FVG bearish [4064.99-4071.16]` — bornes = géométrie 3-bougies réelle ✓ (fix F3 opérant). Aucune erreur de placement constatée sur les bornes.

**D. Écart vs SMC** : la définition 3-bougies est **standard** (F1 0,88 à l'audit du 27 mai). Les écarts : seuil 0,1 ATR ≈ spread XAU (P1-2 connu), pas de suivi de comblement, disparition après 1 bougie (un trader garde le FVG affiché tant qu'il n'est pas comblé).

**E. Classification** : définition = **DÉFENDABLE** ; seuil = **ÉCART DE DÉFINITION** (annotation) ; cycle de vie/cardinalité = **ÉCART structurel** (Groupe B) ; T2/T3 = **BUG** transverse.

### 2.3 BOS

**A. Définition** (`strategy_features.py:32-134`) :
- Swings = fractales 2-2 (extremum local sur 5 bougies, confirmé 2 barres plus tard — causal, `:617-644`).
- Structure haute/basse = dernières fractales ; cassure = **close** au-delà de la structure (`:117, :124`), pas la mèche.
- `BOS_EVENT` = ±1 uniquement sur la barre de cassure ; `BOS_BREAK_LEVEL` = niveau structurel cassé, persisté (`:118, :125`) ; garde anti-redéclenchement `allow_bos_up/down` (`:97-98`).
- Surface produit (D1-b, `market_reading_mappers.py:252-291`) : cassure affichée si fraîche (`BOS_EVENT≠0`) OU persistée tant que la machine à retest est active (`BOS_RETEST_STATE≠0`, fenêtre bornée awaiting 20 + armed 30 barres, `SMCConfig:505-518`) et que le trend ne s'est pas inversé. Niveau réel publié (forward-fill `BOS_BREAK_LEVEL_LAST`).

**B. Cardinalité** : **1 cassure max** (la plus récente encore « vouchée »). Pas d'historique des cassures précédentes ni de niveaux structurels multiples (un trader marque plusieurs niveaux cassés/à retester). 5 événements BOS dans la fenêtre de 200 barres du cas flaggé.

**C. Comportement observé** : XAU H1 11 juin : `BOS bearish @4172.57, broken_at=2026-06-10T18:00Z` persisté sur plusieurs lectures ✓ (D1-b opérant, niveau stable). MAIS `broken_at` est en heure décalée +10 h (T2) — la vraie cassure date du 10 juin ~08:00 UTC. Et les `broken_at` **futurs** sont publiés (preuves §1-T2).

**D. Écart vs SMC** : fractale 2-2 + cassure au close = lecture SMC valide (variante conservatrice). La confirmation 2 barres retarde la mise à jour de structure — acceptable. Le niveau de cassure publié est le bon (validé 99,8 % à l'audit du 27 mai).

**E. Classification** : définition = **DÉFENDABLE** ; fenêtre de persistance (20+30) et incohérence historique `armed_window=5` engine-defaults vs 30 config (P1-1) = **ÉCART DE DÉFINITION** (annotation tranche la durée d'affichage souhaitée) ; `broken_at` décalé/futur = **BUG** (T2) ; cardinalité (1 seul niveau) = **ÉCART structurel**.

### 2.4 CHOCH

**A. Définition** (`strategy_features.py:100-115`) : cassure **en sens inverse** du trend en cours (close au-delà de la structure opposée) — `CHOCH_SIGNAL=±1` sur cette barre uniquement, avec `BOS_EVENT` simultané. Niveau publié = `BOS_BREAK_LEVEL` du même bar (`market_reading_mappers.py:293-309`).

**B. Cardinalité** : ≤ 1 et **visible 1 seule bougie** — contrairement au BOS, aucune persistance D1-b n'est appliquée au CHOCH (`choch_direction` lu sur le signal du bar courant uniquement). Résultat mesuré : **1 CHOCH sur 45 lectures** XAU M15. Un changement de caractère — l'information la plus structurante pour un trader — disparaît du produit après 15 minutes sur M15.

**C. Comportement observé** : XAU M15 11 juin 01:15Z : `CHOCH bullish @4077.67` affiché une lecture, disparu à 01:30 (remplacé par BOS persisté). Conforme au code.

**D. Écart vs SMC** : définition de la bascule = défendable (même moteur que BOS). L'écart est la **persistance nulle** côté produit.

**E. Classification** : définition = **DÉFENDABLE** ; visibilité 1-bougie = **ÉCART** (cardinalité/persistance, Groupe B — la durée d'affichage souhaitée est une décision d'annotation) ; T2/T3 = **BUG** transverse.

### 2.5 Phase / Régime

**A. Définition** (`market_reading_mappers.py:397-483`) — heuristiques pures sur les 200 bougies du TF demandé (le HMM a été retiré du client, conformément à l'audit du 27 mai) :
- `trend` : premier close vs dernier close de la fenêtre ; « ranging » si |mouvement| < 0,3 × range total (`:401-414`).
- `volatility_observed` : moyenne des true ranges des 7 dernières barres vs moyenne des ~193 précédentes ; low < 0,7, elevated > 1,3 (`:417-435`).
- `market_phase` : mapping direct — trend directionnel + vol élevée → `expansion`, sinon `trend` ; ranging → `ranging` ; neutral → `accumulation` (`:438-443`).
- `mtf_confluence` : **vide en production** — aucun `mtf_provider` câblé dans le bootstrap (`bootstrap.py:106-112`).

**B. Cardinalité** : 1 triplet par lecture (par construction — pas un problème de tracking).

**C. Comportement observé — cas flaggé** : XAU M15 14:00Z : `trend=ranging, vol=normal, phase=ranging` pendant que H1 affiche `bearish/elevated/expansion` au même instant. La fenêtre M15 (200 barres = 50 h) et la fenêtre H1 (200 barres = 8,3 jours) mesurent des choses différentes sous le même vocabulaire — l'utilisateur qui regarde un rally intraday peut lire « bearish » (hérité de la baisse du début de fenêtre).

**D. Écart vs SMC/Wyckoff** : « accumulation/distribution/expansion » évoquent Wyckoff mais ne sont **pas** détectés comme tels : `distribution` est **inatteignable** (aucune branche ne le produit, `:438-443` — vocabulaire mort du schéma) et `accumulation` ne sort que si trend=`neutral` (rare avec first-vs-last sur 200 barres). La fenêtre fixe de 200 barres n'est pas un choix de définition documenté, c'est un artefact du lookback.

**E. Classification** : heuristique trend/vol = **ÉCART DE DÉFINITION** (à trancher : fenêtre, vocabulaire, méthode) ; `distribution` inatteignable = **BUG mineur** (promesse de vocabulaire non tenue — corrigeable par documentation OU attendu d'annotation) ; bougie partielle dans la fenêtre = **BUG** transverse (T3).

### 2.6 News

**A. Définition** (`news_pipeline.py`) : flux JSON ForexFactory (this-week + next-week, `:49-52`), seuls medium/high conservés (`:56, :125-127`), devises USD/EUR (`market_reading_assembler.py:307-310`), fenêtres : upcoming 240 min, just_published 60 min. Descriptions template factuelles (niveau 1.5 strict), garde anti-tokens interdits. TTL fetch 120 s, cache SQLite.

**B. Cardinalité** : **liste de N événements — le seul détecteur avec une vraie cardinalité multiple.** ✓

**C. Comportement observé — cas flaggé** : lecture 14:00Z → `Prelim UoM Consumer Sentiment (USD), publié il y a 10 min` + `UoM Inflation Expectations`. Horaire exact (UoM = 10:00 ET = 14:00 UTC ✓), impact correct, `actual=null` honnête (pas encore dans le flux). **Correct.**

**D. Écart** : `_FF_DEFAULT_TZ = UTC-5` fixe (`:69`) pour les timestamps naïfs — faux en été (EDT = UTC-4) → décalage potentiel d'1 h, **latent** (le flux émet normalement des offsets). Pas de lien news↔instrument au-delà du filtre devise. Pour mémoire : le « blackout 30/60 vs 30/30 » documenté à l'audit du 27 mai concerne le chemin legacy `ConfluenceDetector` (hors /app).

**E. Classification** : **DÉFENDABLE** (le plus sain des six) ; DST naïf = **BUG latent** à corriger sur preuve.

---

## 3. Trace pas-à-pas du cas flaggé (XAU M15, 12 juin, OB 4185-4193 sous prix ~4190-4196)

**Lecture en base** : `XAUUSD M15 candle_close_ts=2026-06-12T14:00:00Z`, `created_at=14:10:44 UTC`, `close_price=4194.65`, structure = 1 OB `bullish [4185.50142-4192.87212] importance=high`, bos/choch/fvg null, régime ranging.

1. À 14:10:44 UTC, l'assembleur (déclenché lazy/scheduler) fetch 200 bougies M15 chez TwelveData. Le feed étiquette en heure exchange **UTC+10** : la barre en cours (vraie 14:00-14:15 UTC, ouverte depuis 10 min) arrive étiquetée `2026-06-13T00:00` (T2).
2. Dernière barre du frame = cette **bougie partielle** (T3) : `O=4186.97`, close au moment du fetch `C=4194.65` (close final : 4192.24 — le signal a été calculé sur un état intermédiaire).
3. `SmartMoneyEngine` évalue l'OB haussier sur la paire (i-1, i) :
   - i-1 = barre étiquetée 23:45 (vraie 13:45-14:00 UTC) : `O=4186.55 > C=4186.25` → baissière ✓ (corps de 0,30 $ — la condition tient à 30 cents) ;
   - i (partielle) : `C=4194.65 > O=4186.97` → haussière ✓ ;
   - `high[i] ≥ 4194.65 > high[i-1]=4192.87` ✓.
4. Zone publiée = **range complet de la bougie i-1** : `[low=4185.50142, high=4192.87212]` — correspondance exacte avec la barre 23:45 du cache, au 5ᵉ décimal.
5. Force : `(4192.87-4185.50)/ATR(11.98) = 0.615` + bonus FVG 0,2 (un FVG avait tiré sur la barre précédente) = **0.815 ≥ 0.75 → importance=high**. *Reproduit à l'identique en relançant le moteur sur les bougies finales stockées (lecture seule).*
6. À 14:15:06 (lecture suivante, tick scheduler 6 s après la borne), le feed — avec ~1 min de lag — présentait encore la barre 00:00 comme courante (`close_price=4191.44` = prix vers 14:14) : le **même pattern** re-tire sur la même paire → même zone publiée sous un **nouvel id** (`OB_20260612141500`). Pas de bug de cache : absence d'identité de zone.
7. La lecture 14:45Z (`OB [4183.40-4197.98]`, zone = H/L exacts de la barre étiquetée 00:15) confirme le mécanisme ; son `close_price=4185.25` n'était explicable qu'après réécriture de la bougie partielle dans le cache (T3) — **non reproductible sans snapshot brut du feed**.

**Pourquoi un trader conteste ce placement** : la « zone » est simplement la bougie de 13:45-14:00 — collée sous le prix courant, sans impulsion à son origine, sans statut de mitigation, pendant que les zones que l'œil humain marque (origine du selloff H1 4245→4172 du matin, blocs H4) ne sont jamais émises car le pipeline ne regarde que la dernière barre M15. Sur H1 et H4 au même moment : aucune structure affichée du tout (`trend=bearish/expansion` seuls), le pattern 2-bougies n'ayant pas tiré sur la dernière barre de ces TF.

**Verdict du cas** : aucun non-respect de la règle par le moteur (pas un bug de logique de détection) — mais le résultat traverse **deux bugs réels** (bougie partielle, fuseau) et illustre **l'écart de définition OB + la cardinalité 1**.

---

## 4. Ce que l'annotation founder doit définir (sans quoi « correct » n'est pas mesurable)

| Détecteur | Définition à trancher | Critère d'importance | Critère d'invalidation | « Bon nombre » attendu |
|---|---|---|---|---|
| **OB** | Quelle bougie est un OB : dernière bougie contraire avant impulsion ? exigence de displacement (ex. move ≥ X×ATR après) ? balayage de liquidité requis ? ancrage à un BOS ? Zone = corps ou range complet ? mèche incluse ? | Qu'est-ce qui rend un OB « affichable » : fraîcheur (jamais retesté) ? taille de l'impulsion qui en sort ? confluence HTF ? proximité du prix ? | Quand cesse-t-il d'exister : 1er retest (mitigé) ? close au-delà ? traversée à X % ? délai max ? | Combien par chart/TF (ex. 2-5) ? les zones HTF doivent-elles apparaître sur le TF courant ? |
| **FVG** | Garder la définition 3-bougies (validée F1 0,88) ; trancher le **seuil minimal** (0,1 ATR actuel ≈ spread — 0,3-0,4 ATR ?) | Taille relative ? FVG dans le sens de la structure seulement ? | Comblé à 50 % (`partially_filled`) ? 100 % (`filled`) ? close dans le gap ? | Combien de gaps ouverts affichés simultanément ? |
| **BOS** | Fractale 2-2 OK ou swing plus large ? cassure au close (actuel) ou à la mèche ? | Toutes les cassures ou seulement celles de swings majeurs ? | Quand un niveau cassé cesse-t-il d'être pertinent (actuel : timeout 20+30 barres) ? reclaim ? | Combien de niveaux structurels visibles (le dernier seul, actuel, ou les N derniers) ? |
| **CHOCH** | Même base que BOS — valider que « cassure inverse au trend » suffit (vs exigence de séquence HH/HL cassée) | — | Durée d'affichage : combien de temps un CHOCH reste-t-il l'info dominante (actuel : 1 bougie) ? | 1 par TF (le dernier) ? |
| **Phase/Régime** | Fenêtre de mesure par TF (200 barres actuelles = 50 h M15 / 8 j H1 / 33 j H4 sous le même mot) ; vocabulaire : garder accumulation/distribution (jamais émis correctement) ou réduire à trend/range/expansion ? | — | — | — |
| **News** | Fenêtres 240/60 min OK ? medium inclus ou high seul ? | Impact min affichable | — | Cap d'événements affichés ? |

---

## 5. Grille d'annotation (60 lectures)

Mode d'emploi : pour chaque lecture sauvegardée (screenshot /app + payload `market_readings.db`), juger chaque détecteur sur 3 axes binaires + le nombre attendu. Remplissage cible < 2 min/lecture. **Pré-requis : trancher d'abord les définitions du §4** (sinon les O/N ne sont pas interprétables) et **corriger T2/T3 avant la campagne** (annoter des sorties stables, pas des bougies partielles).

```
Lecture #__  | Instrument: ____ | TF: ____ | candle_close_ts (UTC): ____________
Prix au moment de la lecture : ________

                       Détecté à raison ?  Placement OK ?   Nombre OK ?   Nombre attendu   Note libre
                       (O/N/Manqué*)       (O/N/n.a.)       (O/N)         (entier)
OB                     ____                ____             ____          ____             ______________
FVG                    ____                ____             ____          ____             ______________
BOS (niveau affiché)   ____                ____             ____          ____             ______________
CHOCH                  ____                ____             ____          ____             ______________
Phase/Régime           ____                n.a.             n.a.          n.a.             ______________
News                   ____                n.a.             ____          ____             ______________

* « Manqué » = le détecteur n'a rien émis alors qu'une zone/un événement évident aurait dû l'être
  (faux négatif). « N » sur Détecté = émis à tort (faux positif).
Pour chaque OB attendu mais absent : niveau_high ____ / niveau_low ____ / TF d'origine ____
Importance affichée (OB) : low/medium/high — d'accord ? O/N ; importance attendue : ____
```

Colonnes CSV équivalentes (pour `scripts/audit/` futur, format compatible `SCORING_TEMPLATE_2026_06_06.csv`) :
`reading_id, instrument, tf, close_ts, detector, detected_ok, placement_ok, count_ok, expected_count, expected_zones, importance_ok, note`

Métriques dérivées après campagne : précision/rappel par détecteur, taux de placement correct, écart de cardinalité (nombre émis vs attendu), matrice par TF.

---

## 6. Synthèse et plan stagé

### 6.1 Tableau de synthèse

| Détecteur | Cardinalité actuelle (produit) | Classification dominante | Corrigeable maintenant (bug) | Gated annotation |
|---|---|---|---|---|
| Order Block | 0-1, vie 1 bougie, jamais invalidé | ÉCART DE DÉFINITION (pattern + importance) + ÉCART structurel (cardinalité) | T2, T3 | définition ICT, importance, invalidation, top-N |
| FVG | 0-1, vie 1 bougie, jamais « filled » | DÉFENDABLE (définition) + ÉCART (seuil, cycle de vie) | T2, T3 | seuil, règle de comblement, N gaps ouverts |
| BOS | 1 (dernier, persisté 20+30 barres) | DÉFENDABLE (définition, niveau réel) | T2 (broken_at futur), T3 | fenêtre de persistance, N niveaux |
| CHOCH | 0-1, vie 1 bougie | DÉFENDABLE (définition) + ÉCART (persistance nulle) | T2, T3 | durée d'affichage |
| Phase/Régime | 1 triplet | ÉCART DE DÉFINITION (heuristique, fenêtre fixe) + BUG mineur (`distribution` mort) | T3 ; vocabulaire mort (doc ou code) | fenêtre/TF, vocabulaire, méthode |
| News | **N** ✓ | DÉFENDABLE | DST naïf (latent) | fenêtres, impact min |

### 6.2 Plan stagé vers « OB multiples corrects + bon nombre + FVG/BOS/CHOCH justes »

**Étape 1 — Bugs, maintenant, sur preuve (aucun seuil de détection touché)**
1. **T2** : requêter TwelveData avec `timezone=UTC` explicite (ou parser le fuseau retourné) + test d'alignement « dernière bougie ≤ horloge UTC » ; purger/re-fetcher `candles_cache`.
2. **T3** : exclure la bougie en formation de l'analyse (dropper la dernière barre si son close théorique n'est pas atteint) — correction de contrat de données (`candle_close_ts` promet une bougie close), pas une heuristique.
3. Unifier le domaine d'horloge de `broken_at`/`created_at` (suit mécaniquement T2) + test « broken_at ≤ candle_close_ts ».
4. News : remplacer `_FF_DEFAULT_TZ` fixe UTC-5 par le fuseau America/New_York DST-aware (latent, sur preuve du flux).
5. Observabilité : snapshotter la réponse brute du provider par lecture (reproductibilité d'audit — la lecture 14:45 était inexplicable sans).

**Étape 2 — Annotation founder = définition du « correct »**
Trancher le §4, puis campagne de 60 lectures avec la grille §5 sur les sorties post-Étape 1. Livrable : définitions signées + dataset annoté (vérité terrain).

**Étape 3 — Calibration + multi-zones/invalidation (Groupe B), validées CONTRE l'annotation**
Registre de zones (scan des 200 barres — les candidats existent déjà dans les colonnes du moteur), cycle de vie (`mitigated`/`invalidated`/`filled`, `tested`), critère d'importance annoté, top-N par TF, persistance CHOCH, recalibrage des seuils (FVG 0,1 ATR, fenêtres régime) — chaque choix mesuré contre le dataset de l'Étape 2, pas à l'intuition.

**Étape 4 — Mesure avant/après**
Re-passer les 60 lectures (et un échantillon frais) dans le pipeline post-Étape 3 : précision/rappel par détecteur, taux de placement, écart de cardinalité — comparés à la baseline mesurable dès l'Étape 2. Critère de sortie chiffré à fixer avec le founder (ex. ≥ 80 % « détecté + placé + nombre OK » sur OB/FVG).

### 6.3 Verdict

**Mélange, avec des proportions claires.** Le « mauvais placement » du 12 juin n'est pas une violation de règle : c'est une **règle pauvre (écart de définition OB)** rendue encore moins fiable par **2 bugs transverses prouvés** (fuseau +10 h → horodatages incohérents/futurs ; analyse sur bougie partielle → signaux non finaux, doublons, non-reproductibilité). Le « manque d'analyse » (un seul OB, FVG fugaces, CHOCH invisible, rien sur H1/H4) est à ~80 % un **manque structurel de cardinalité** : le moteur calcule déjà tous les candidats (40 OB, 34 FVG sur la fenêtre du cas), le produit n'en surface qu'un, choisi par récence. Décompte : **3 bugs prouvés + 1 latent + 1 mineur** (corrigeables maintenant) ; **~7 écarts de définition** (OB pattern, OB importance, seuil FVG, persistances, fenêtre régime, vocabulaire phase) gated annotation ; **1 chantier structurel** (registre multi-zones + invalidation) gated annotation. La bonne nouvelle : le schéma produit (listes + statuts de cycle de vie) et le moteur (colonnes par barre) sont déjà compatibles avec la cible multi-zones — le chaînon manquant est le registre intermédiaire et les définitions du founder.

---

*Audit lecture seule — aucun code, seuil ou test modifié. Reproductions exécutées en mémoire sur les données live (`market_readings.db`, `candles.db`) du 12 juin 2026.*
