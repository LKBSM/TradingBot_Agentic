# AUDIT LP-2 — Page d'accueil : diagnostic (Section 1, lecture seule)

> Branche `feat/lp-2-accueil` (worktree `C:/MyPythonProjects/wt-lp-2`), base `main` @ `6cfe8cf`.
> Référence de structure : `docs/design/reference-accueil.html` (maquette v3, poussée sur `main`).
> **Aucun code de page écrit.** Ce document est le livrable de la Section 1 : captures, inventaire, écarts, chiffres du bandeau.

---

## A) Captures du produit RÉEL

Produit lancé en local, hors ligne : backend `python -m src.intelligence.main` (`DATA_SOURCE=csv`, lecture *read-through* de `candles.db`, `SENTINEL_TESTING_MODE=1`, port 8000) + frontend `next dev` (port 3400, proxy `/api → :8000`). Playwright a capturé le produit tel qu'il tourne aujourd'hui — **structure de détection réelle** (vrais BOS/CHOCH, niveaux, horodatages, `bars_ago`). Données de marché figées au 03/07/2026 (cache) → le produit affiche donc son badge « en retard », comportement réel.

Limite honnête : **M.I.A ne répond pas** (crédits Anthropic épuisés → narration en repli sur le moteur de gabarits, chat désactivé). Les captures M.I.A montrent donc l'**interface** réelle (en-tête, invite, questions suggérées), pas une réponse générée live.

Emplacement : `docs/design/captures/` (14 PNG).

| Fichier | Surface / état |
|---|---|
| `desktop-app-xauusd-m15-full.png` | /app Or M15 — page entière (chart + lecture narrée + régime + structure + M.I.A) |
| `desktop-app-xauusd-m15-viewport.png` | /app Or M15 — au-dessus de la ligne de flottaison |
| `desktop-app-layers-before.png` / `desktop-app-layers-after-fvg.png` | /app — avant / après masquage de la couche FVG |
| `desktop-app-xauusd-h1-full.png` | /app Or **H1** — changement d'unité de temps |
| `desktop-app-mia-panel.png` | /app — panneau M.I.A ouvert (questions suggérées réelles) |
| `desktop-scanner-full.png` / `desktop-scanner-viewport.png` | /scanner — construction des conditions |
| `desktop-zones-full.png` | /zones — fiches de cycle de vie (24 zones) |
| `desktop-actualites-calendar-full.png` | /actualites — calendrier mensuel + filtres + honnêteté |
| `mobile-app-full.png` | /app 390 px — shell 3 onglets (Marchés / Lecture / Chat) |
| `mobile-scanner-full.png` / `mobile-zones-full.png` / `mobile-actualites-full.png` | surfaces mobiles 390 px |

Reste à capturer après GO si tu veux (nécessite crédits Anthropic pour du live M.I.A, et un onglet « Lecture » mobile déplié) : une réponse M.I.A réelle, une carte de résultat scanner détaillée, une page de publication /actualites/{id}, une fiche de zone dépliée « Détails ».

---

## B) Inventaire des capacités réelles (source : code)

### /app — l'espace de lecture
- **Chart à couches** (5 bascules, `DesktopReading.tsx` `LayerPills`) : `OB` (Order Blocks), `FVG` (Fair Value Gaps), `BOS/CHOCH` (cassures + retest + marqueurs), `Liquidité` (poches BSL/SSL), `Mitigées` (filtre : masque les zones déjà testées). Toute bascule est **affichage seul**, jamais la détection.
- **Lecture narrée** (`NarratedPanel`) : paragraphe(s) composés depuis `reading.conditions.description`, régénéré côté backend à chaque lecture — **il se réécrit quand une couche change**. En-tête « Lecture narrée », badge « Ancrée au moteur », pied « Chaque niveau cité correspond à une sortie réelle ».
- **Panneau Régime** (`RegimeCard.tsx`, **10 tuiles**, chacune à 2 onglets Donnée/Concept) : `Tendance`, `Volatilité`, `Position dans le range`, `Alignement` (confluence multi-unités), `Maturité` (bougies depuis l'ancre), `Dernier évén.`, `Densité` (N OB · M FVG actifs), `Session`, `Niveaux de référence`. La tuile « Phase de marché » a été **retirée (RG-1b)** : le moteur n'expose pas de phase réelle.
- **Journal des événements** : tous les BOS/CHOCH de la fenêtre d'analyse (plus de troncature), par ligne : `BOS ↑` / `CHOCH ↓`, niveau cliquable (cadre la barre de confirmation), horodatage. Fenêtre nommée (« ≈ 1 mois en H1 »).
- **M.I.A** (`chatbot.py`) : actions d'affichage **whitelistées** — `set_layer_visibility`, `filter_zones`, `focus_zone`, `highlight_zone`, `hide_zones`, `isolate_zones`, `show_zones`, `focus_price`, `fit_chart`, `reset_view`, `set_instrument_timeframe`. **Ancrage dur** : toute zone référencée doit avoir un id réellement émis (id-lock) ; toute clé de géométrie dans les params est rejetée ; **refus** de prédiction/conseil/signal via 3 couches (filtre adverse d'entrée, prompt système, filtre de sortie). Outils factuels : `get_market_reading`, `get_signal_summary`, `get_ob_diagnostic` (pourquoi telle bougie est/n'est PAS un OB).
- **Unités de temps** : M5, M15 (défaut), H1, H4, D1 affichées (M1 derrière un flag). Changer d'unité refait la lecture + recalcule l'alignement multi-unités.
- **Interactions chart** (`focusController.ts`) : clic zone → cadrage ; clic événement → cadre la confirmation ; clic niveau → cadrage horizontal ; Échap → efface ; caméra animée ~400 ms.

### /scanner — « l'angle est la stratégie »
- **22 conditions, 4 familles repliables** : Structure (6), Zones (8), Liquidité (4), Contexte (4). Contrôles segmentés (direction, fenêtre en bougies, etc.). Bloc « ce qui va à l'encontre » jamais masquable. Compte de combos **live** (construction) + **par lecture**. 5 états de résultat (correspond / presque / non-correspondant / non-évaluable / sur données plus anciennes). Export/import + stratégies enregistrées (localStorage). Vocabulaire **« stratégie »** confirmé.

### /zones
- Fiches de **cycle de vie** par zone détectée : type + direction (`OB ↑/↓`, `FVG ↑/↓`), fourchette de prix, badge d'état **`ACTIF` / `Mitigé` / `Comblé X %`**, frise (Formé → touches → Maintenant), narration (âge en bougies + durée, distance au prix, testée/comblée), actions « Analyser la zone » / « Masquer du graphique ». Filtres Toutes/Actives/Mitigées ; tri Proximité/Fraîcheur/État.

### /actualites
- Calendrier mensuel, filtres factuels **Organisme** (BLS, BEA, Census, Réserve fédérale, Eurostat, BCE), **Marché** (Or, EUR/USD), **Périodicité** (Mensuel, Trimestriel, 8×/an) ; 3 états de chargement ; bloc « CE QUE CE CALENDRIER NE DIT PAS ». Page de publication : courbe des valeurs publiées, questions/mesures (rejeu sans look-ahead), états de valeur (publiée/à venir/non récupérée/indisponible), liens source nommés, révisions. **Sources officielles uniquement** (ForexFactory exclu).

---

## C) Écarts maquette v3 ⟷ produit réel (corrigés vers la réalité)

| # | La maquette montre / omet | Réalité produit | Correction prévue |
|---|---|---|---|
| C1 | **Espace de lecture = un graphique annoté** (une seule vignette dans « Outils ») | 4 panneaux : chart à couches **+ lecture narrée + régime (10 tuiles) + journal + M.I.A** | Carrousel ≥ 5 volets (§3), volet chart réellement manipulable si possible |
| C2 | Régime cité comme « tendance, phase, volatilité, accord » | **10 tuiles réelles**, et **« Phase » retirée** du produit (RG-1b) | Montrer les vraies tuiles ; ne pas promettre « Phase » |
| C3 | Carrousel « **20 zones suivies** » | **24 zones** sur Or M15 aujourd'hui (variable) | Formuler sans nombre figé, ou lire un compte réel |
| C4 | Bandeau : 4e chiffre **« 7 structures détectées »** | Aucune source unique « 7 structures » dans la config (`stats.ts` = markets/timeframes/combinations/conditions) | Voir §D — soit dériver « types de structure » d'une source réelle, soit remplacer par « 12 combinaisons » (déjà source unique) |
| C5 | Barre de nav visiteur affiche… (maquette : M.I.A/Démo/Outils/Tarifs/FAQ) | **Implémentation actuelle affiche App/Zones/Scanner à TOUT visiteur** (`Nav.tsx`, sans gate) | Masquer App/Zones/Scanner pour non-connecté (exigence + test) |
| C6 | Démo « couches » = SVG figé + narration statique | Le vrai produit **réécrit** la narration au masquage d'une couche | Rendre la démo « lecture » réellement recomposable (prouve le principe) |
| C7 | M.I.A : chat scénarisé, sans portée réelle | M.I.A **pilote l'affichage**, refuse la prédiction, ancre chaque référence, `get_ob_diagnostic` | Étoffer la section M.I.A (que fait-elle / où / pourquoi) avec vraies capacités |
| C8 | Scanner présenté surtout comme « conditions » | Angle réel : **décris ta stratégie une fois → vérifiée en continu sur tous tes marchés** | Employer « stratégie », montrer la carte de résultat + bloc « à l'encontre » |
| C9 | « Comment ça marche » sans compteurs | OK côté maquette | Garder sans nombres internes |
| C10 | Vitrine actualités : « Rapport sur l'emploi · 7 août » | Le vrai calendrier a bien un événement le 7 (08:30 Rapport…) | Fidèle — s'appuyer sur une capture réelle |

### Écart transverse — VOCABULAIRE « moteur » / « engine »
L'implémentation actuelle de l'accueil (`messages/fr.json`, `messages/en.json`, namespace `home`) contient **~11 occurrences « moteur » (fr)** et **~10 « engine » (en)** (eyebrow « Cinq outils, un même moteur », « ce que le moteur a détecté », « Ancrée au moteur », etc.). **La règle LP-2 interdit ces mots.** La maquette v3, elle, dit déjà « Cinq outils, un même **produit** » et « Ancrée aux données réelles ». → Réécriture complète du namespace `home` (fr+en, puis 7 locales) + test d'absence de « moteur »/« engine ». Note : la tuile produit « Ancrée au moteur » vit dans le namespace **app** (hors périmètre) — ne pas y toucher.

### Écart transverse — MOTS INTERDITS vs FAQ
Le test LP-2 doit interdire *signal, setup, opportunité, gagnant, probabilité, cible, biais* (et équivalents EN) dans les chaînes visibles de l'accueil. Or la FAQ de la maquette pose « Est-ce que MIA donne des **signaux** d'entrée ? ». Il faudra reformuler la question sans le mot interdit tout en gardant le sens (ex. « MIA me dit-elle quand acheter ou vendre ? »). **À trancher en Section 4** (ne rien casser silencieusement).

---

## D) Les chiffres du bandeau — d'où viennent-ils, disent-ils vrai ?

Source unique actuelle : `webapp/lib/landing/stats.ts` (dérivée de `perimeter.ts`, `timeframes.json`, palette scanner).

| Chiffre | Valeur réelle | Source | Maquette v3 | Verdict |
|---|---|---|---|---|
| Marchés suivis | **2** (XAUUSD, EURUSD) | `perimeter.ts` `SUPPORTED_INSTRUMENTS` | 2 | ✅ vrai |
| Unités de temps | **6** (M1, M5, M15, H1, H4, D1) | `timeframes.json` (`perimeter:true`) | 6 | ✅ vrai |
| Combinaisons | **12** (2 × 6) | `stats.ts` | *(absente en v3)* | ✅ vrai |
| Conditions de recherche | **22** | palette scanner | 22 | ✅ vrai |
| **« 7 structures détectées »** | **pas de source unique** | — | **présente en v3 (4e chiffre)** | ⚠️ **à sourcer ou remplacer** |

Décision à prendre (§C4) : les « types de structure » réels que le produit détecte sont : Order Block, Fair Value Gap, BOS, CHOCH, poche BSL, poche SSL, niveaux égaux (EQH/EQL) → **7** est défendable, mais **il n'existe aucune constante unique** aujourd'hui. Deux options honnêtes :
1. Créer une source unique `STRUCTURE_TYPES` (7) lue par le bandeau + test de garde.
2. Remplacer le 4e chiffre par **« 12 combinaisons »** (déjà source unique, déjà testé).

**Ligne d'ambition** (sous le bandeau) — la maquette la porte déjà : « Accès anticipé · **2 marchés aujourd'hui, 50 à 80 prévus au lancement** ». À conserver.

---

## Structure de la maquette v3 (ordre confirmé)
1. En-tête (nav : M.I.A · Démo · Outils · Tarifs · FAQ · Se connecter · Essayer gratuitement · sélecteur FR)
2. Hero + **bandeau de chiffres** + ligne d'ambition
3. **M.I.A Agent** (remontée : que fait-elle / où / pourquoi + 4 capacités + chat exemple avec refus)
4. **Démonstrations interactives** (4 onglets : Lire une structure / Définir une stratégie / Suivre une zone / Parler à M.I.A) + « Données d'illustration »
5. **Outils** (5 surfaces) — dont **carrousel « espace de lecture »** (Graphique / Lecture narrée / Régime / Journal / Unités)
6-8. Scanner · Zones · Actualités (dans « Outils »)
9. Comment ça marche · 10. Pour qui · 11. Ce qui nous distingue · 12. Tarif · 13. FAQ · 14. CTA · 15. Footer

## FAQ de la maquette — vérification vs produit (à finaliser en §4)
- « MIA donne des signaux d'entrée ? » → **exact** (refus, décision au lecteur) — mais **reformuler sans « signaux »** (mot interdit).
- « À quoi sert M.I.A ? » → exact (ne cite que du détecté).
- « Faut-il connaître le SMC ? » → exact (M.I.A enseigne avec l'exemple).
- « Remplace TradingView ? » → exact (complémentaires).
- « Quels marchés ? » → **exact : l'or et l'euro/dollar (2)**.
- « Fréquence des données ? » → exact (chaque écran affiche l'âge de sa lecture ; badge de retard réel).
- « Annuler quand je veux ? » → exact (palier gratuit permanent, mensuel annulable).
- « D'où viennent les données ? » → **à nuancer** : pour le **calendrier**, sources officielles seules (BLS/BEA/Census/Fed/Eurostat/BCE, pas d'agrégateur). Pour les **prix/bougies**, le fournisseur est **Twelve Data** (vendeur de données de marché). Ne pas laisser croire « aucune source tierce » pour les prix.

---

## Tarif (rappel PRIX-1, source unique `config/pricing.json` → `pricing.generated.ts`)
39 $ US/mois · 348 $ US/an (= 29 $ US/mois) · palier gratuit permanent sans carte · devise explicite partout · mentions : annulable, USD, outil d'information/éducation sans conseil/signal, risque de perte, 18 ans+.
