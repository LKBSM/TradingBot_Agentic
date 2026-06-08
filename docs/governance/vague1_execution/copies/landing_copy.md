# Copy — Landing page

**Statut** : ✅ Révisé 2026-05-27 post-audit algo. Tous claims de performance retirés. Positionnement "outil de compréhension augmentée".

---

## Tagline H1

> **Comprenez les marchés Or et FX.**
> **Décidez en autonomie.**

(Optionnel : « **comprenez** » et « **décidez** » accentués en or `var(--gold)`.)

## Sub-tagline

> M.I.A. Markets est un outil pédagogique d'analyse algorithmique des marchés Or et FX. Il décompose la lecture du marché en couches, vous laisse poser toutes vos questions via le chatbot Sentinel, et refuse pédagogiquement de vous donner des ordres. Vous restez seul décideur.

## Sous-tagline factuelle (en-dessous des stats hero)

> Pipeline algorithmique transparent · Méthodologie publique (López de Prado, Corsi, Gibbs & Candès, etc.) · Validation OOS en cours · Posture éducative assumée.

## 4 stats hero — Révisées (zéro claim de performance non substancié)

| Chiffre | Label |
|---|---|
| **8** | facteurs analysés |
| **12** | papers académiques sources |
| **7 ans** | de données historiques |
| **2 actifs** | XAU + EUR/USD |

> ❌ Ce qu'on n'affiche PLUS (retiré 2026-05-27) :
> ~~329 setups analysés~~ · ~~Profit factor 1.30~~ · ~~IC 95 %~~ · ~~Walk-forward validated~~
>
> Tous ces chiffres seront ré-introduits **uniquement après validation OOS** (Brier > +2 % AND DSR > 1.0 AND PBO < 0.5).

## CTAs principaux

| Bouton primary | "Découvrir gratuitement" → `/signup?tier=free` |
| Bouton secondary | "Voir la méthodologie" → `/methodologie` |

## Section "Démo live" — eyebrow + titre

> **DÉMO LIVE**
>
> Tout est là. Vous décidez ce que vous voulez voir.
>
> Une seule lecture, structurée en couches. Le hero card reste visible en permanence. Le détail s'ouvre quand vous le voulez. Pas de choix imposé avant la valeur.

## Section "Différenciateurs" — eyebrow + 3 cards

> **TROIS DIFFÉRENCIATEURS RÉELS, PAS DE PROMESSES**

### Card 1 — Méthodologie publique transparente
**Icône** : 📊
**Titre** : Méthodologie publique, sources académiques citées
**Body** : Pipeline algorithmique 5 briques (HAR-RV Corsi 2009, HMM 3-état, BOCPD Adams & MacKay, ACI Gibbs & Candès, Bipower Barndorff-Nielsen). Code open-source. Notebooks de backtest disponibles sous accord. Pas de boîte noire.

### Card 2 — Chatbot Sentinel comme moyen principal
**Icône** : 💬
**Titre** : Sentinel répond à toutes vos questions
**Body** : Sentinel ne livre pas un dashboard à lire — il dialogue avec vous. Il définit le jargon technique, décompose la lecture algorithmique, et **refuse pédagogiquement de vous donner un ordre**. C'est ce qui sépare un outil d'analyse d'un faux signal de trading.

### Card 3 — Honest confidence assumée
**Icône** : 🛡️
**Titre** : « Nous ne disons pas ce qui va arriver »
**Body** : *« Nous ne vous disons pas quoi faire — nous vous donnons les meilleurs outils pour comprendre. »*<br>Posture éducative. Compliance UE 2024/2811 par construction. `edge_claim=False` tant que validé OOS. Phase d'accès anticipé en cours.

## Footer compliance permanent

> **Outil pédagogique d'analyse algorithmique** · Phase d'accès anticipé · Ne constitue ni un signal de trading, ni un conseil en investissement, ni une recommandation.
>
> Données historiques 7 ans (Dukascopy) · Sources académiques publiques · Validation statistique OOS en cours · Conformité UE 2024/2811 par construction.
>
> © 2026 M.I.A. Markets · [CGU](/cgu) · [Privacy](/privacy) · [Méthodologie](/methodologie) · [Contact](mailto:contact@mia.markets)

---

## EN version (secondaire)

### Tagline H1
> **Understand the gold and FX markets.**
> **Decide on your own.**

### Sub-tagline
> M.I.A. Markets is an educational algorithmic analysis tool for gold and FX markets. It breaks down market reading into layers, lets you ask any question via the Sentinel chatbot, and pedagogically refuses to give you orders. You remain the sole decision-maker.

### Factual sub-tagline
> Transparent algorithmic pipeline · Public methodology (López de Prado, Corsi, Gibbs & Candès, etc.) · OOS validation in progress · Educational posture assumed.

### 4 stats hero (EN)
| Number | Label |
|---|---|
| **8** | analyzed factors |
| **12** | academic source papers |
| **7 years** | historical data |
| **2 assets** | XAU + EUR/USD |

### CTAs
- Primary: "Discover for free" → /signup?tier=free
- Secondary: "See methodology" → /methodologie

### Footer compliance
> **Educational algorithmic analysis tool** · Early Access phase · Does not constitute a trading signal, investment advice, or recommendation.

---

## ⚠️ Justification du retrait des claims chiffrés

Suite à l'audit algo `AUDIT_ALGO_2026_05_27.md` :
- Scoring rule-based actuel : **Pearson −0.023** (zéro pouvoir prédictif)
- Backtest 7 ans : **PF 0.786, return −62 %**, sous-perf −318 pp vs Buy & Hold
- PREMIUM tier : **1 trade en 7 ans** (cosmétique)

→ Tout claim "PF 1.30 / 329 setups / IC 95 % / Win rate 31.9 %" est **non substancié empiriquement**.
→ Les chiffres retirés. Repositionnement "outil de compréhension augmentée" assumé.
→ Ré-introduction conditionnelle des claims **après validation OOS** (Brier > +2 % AND DSR > 1.0 AND PBO < 0.5).

Référence : `docs/governance/decisions/2026-05-27_pivot_positioning_audit.md`.
