# AUDIT VZ-2 — Page Zones : mise en page, densité, hiérarchie

Branche : `fix/vz-2-zones-mise-en-page` (worktree dédié `wt-vz-2`, partie de
`origin/main` à jour `e506735` — le checkout primaire était **22 commits en
retard** en detached HEAD, non touché).

Cible visuelle : `docs/design/reference-zones.html` (présente ✅, suivie pour le
style de carte, la hiérarchie, la jauge et le traitement des blocs).

---

## 1. Diagnostic (lecture seule)

### A) La contrainte de hauteur — cause exacte

`webapp/components/app/ui2c.css`, règle `.zlist` :

```css
.app-shell .zlist {
  max-height: 210px;   /* ← le plafond */
  overflow-y: auto;    /* ← le scroll interne (curseur minuscule) */
  border: 1px solid var(--line);
  border-radius: var(--r-s);
  background: var(--panel-2);
  margin-top: 9px;
}
```

Règle **périmée**, écrite pour une ancienne liste compacte (« Listes zones /
liquidité (scroll à hauteur fixe) », à base de `.zrow`). `ZonesWorkspace` rendait
la colonne de cartes avec `className="zlist"` → **collision de nom de classe** :
la colonne VZ-1 héritait du plafond 210 px, du scroll interne, de la bordure et du
fond `panel-2`. C'est la « boîte trop courte » (défaut 1) **et** la carte tranchée
(défaut 2 : la carte est coupée par le bas du conteneur de défilement à hauteur
fixe, PAS par un `overflow:hidden` sur la carte).

**`.zlist` est partagée** avec `/app` (`LiquidityCard`, `StructureCard`), qui en
dépendent légitimement → **elle n'a PAS été modifiée**. La colonne /zones a reçu
sa propre classe `.zcol` sans plafond ni boîte.

### B) Nombre réel de zones

Backend non joignable en local (aucune lecture persistée) → le nombre live exact
n'est pas mesurable ici. Fixture représentative XAU M15 = **4 zones vivantes** +
groupe « Comblées ». La capture d'origine (curseur de scroll minuscule) confirme
un ensemble live nettement plus grand, dominé par le groupe « Comblées ». La mise
en page livrée (hauteur pleine + scroll unique + 2-up) tient pour 4 comme pour 25.

### C) La coupure

`.zone` n'a aucun `overflow:hidden`. La carte était tranchée par le bas du
conteneur à hauteur fixe (`max-height:210px`) — donc coupable **même avec 350 px
de vide en dessous**. Sans hauteur fixe, plus aucune coupure possible.

### D) Coût de la confluence — **ZÉRO requête par carte**

`useSiblingZones(instrument, timeframe)` est appelé **une seule fois** au niveau du
workspace (≈5 lectures d'unités sœurs servies par cache, pour toute la page) et
passe `siblings` en prop ; chaque carte calcule `buildConfluence(...)` dans un
`useMemo` **pur, côté client**. Passer de 1 à N cartes n'ajoute **aucun** appel
réseau → densifier était sûr, aucun pré-travail requis.

### E) Grille d'origine

`.zlayout` = grid `1fr 340px`, gap 18. Colonne 1 = `.zlist` (cartes empilées 1/rangée,
plafond 210 px). Colonne 2 = `.zmia-col` sticky. Barre filtres/tri en flex au-dessus.
Repli mobile `@media(max-width:1000px)`. `.pagewrap` max-width 1020.

---

## 2. Corrections livrées (présentation uniquement — 0 règle métier, 0 calcul, 0 vocabulaire)

**Mise en page**
- Colonne cartes = nouvelle classe `.zcol` : **plus de `max-height`, plus
  d'`overflow`, plus de boîte**. Le défilement appartient à `.center` (colonne
  centrale) — **un seul conteneur de défilement** (vérifié : seul `.center`
  possède une barre).
- Chaque groupe (`.zgroup`) est un contexte de conteneur ; **2 cartes par rangée**
  via `@container (min-width:600px)`, activé seulement ≥1081 px (là où la colonne
  M.I.A est présente) → **1 carte** en dessous. En pratique : 2-up dès ~1240 px de
  fenêtre (colonne cartes ≥600 px), 1-up entre 1081 et ~1240 et sous 1081.
- En-têtes de groupe `.zsep` **collés** (`position:sticky; top:0`) — chaque en-tête
  reste épinglé pendant le défilement de sa section.
- Aucune carte ne peut être tronquée (plus de boîte à hauteur fixe).
- `.zones-wrap` élargit /zones à **1240 px** (référence).
- Repli M.I.A aligné sur la référence à **1080 px** (bottom-sheet).

**Hiérarchie de carte (3 niveaux)**
- N1 : type + intervalle de prix (la bande prix est le **seul** élément agrandi,
  15,5 px).
- N2 : distance / position + état des contacts (badge). La hauteur & le pourcentage
  passent **sous la bande prix** comme légende (`.zttl`), fini la ligne isolée.
- N3 (« lu si on s'y intéresse ») : confluence, ledger des contacts, frise,
  comblement, origine et clés brutes → **repliés dans « Détails »**. Rien n'est
  amputé : tout est à un clic et repris dans le panneau M.I.A. L'état d'absence de
  confluence (« Rien d'autre détecté ») est montré là aussi — jamais d'implication
  silencieuse de présence.

**Densité**
- **Deux niveaux de boîtes** : la suppression de la boîte `.zlist` (fond + bordure)
  retire le 3ᵉ niveau d'emboîtement ; il reste carte → bloc.
- Étiquette confluence conservée dans le style discret de la référence (mono, petit).
- **≥ 4 cartes visibles sans défiler à 1280 × 800** (voir §4).

**Jauge de proximité (défaut 7)**
- Porte désormais son échelle : les **deux prix d'extrémité** aux bords, la bande de
  la zone surlignée, le prix marqué par la ligne vive. `aria-hidden` — la ligne de
  distance au-dessus énonce les mêmes faits en toutes lettres.

**Non négociables préservés** : aucun jugement (garde `chevauche/respect/valide/
solide/fiable/qualité/meilleur` verte, fr+en), aucun classement/score/tri qualité,
aucune prédiction, ligne absente si donnée absente (pas de tiret de remplissage),
aucun compte avant chargement, filtre vide → message explicite, jargon expliqué.

---

## 3. Cause de la contrainte & confluence (rappel)

- Contrainte de hauteur : `.zlist { max-height:210px; overflow-y:auto }`
  (`ui2c.css`), héritée par collision de nom de classe. **Non modifiée** (partagée
  avec /app) ; la colonne /zones utilise `.zcol` sans plafond.
- Confluence : **0 requête par carte** (une seule lecture des unités sœurs pour
  toute la page, calcul client pur). Densifier n'a multiplié aucun appel.

---

## 4. Cartes visibles — avant / après (1280 × 800, fixture 19 zones)

| | Avant | Après |
|---|---|---|
| Conteneurs de défilement dans la page | **2** (`.center` + boîte `.zlist`) | **1** (`.center` seul) |
| Hauteur de la boîte liste | ~210 px (plafond) + vide mort ~350 px | pleine hauteur, 0 vide |
| Cartes entièrement visibles sans défiler | **1** | **4** |
| Cartes par rangée ≥1240 px | 1 | 2 |
| Carte tranchée possible | oui | non |

Mesures Playwright (fixture 19 zones, 1280×800) : `fullyVisible: 4`,
`scrollers: ["center"]`, `.zcol/.zcards/.zgroup` non-scrollables, `columns` = 2
pistes. Test permanent : `tests/e2e/vz-2-measure.spec.ts`.

**Réserve honnête (à trancher en revue live) :** la référence montre des cartes
riches en colonne unique ; l'objectif chiffré « ≥ 4 cartes visibles » impose des
cartes compactes en 2-up. J'ai suivi la **hiérarchie de la mission** (N3 « lu si on
s'y intéresse ») en repliant confluence + ledger + frise + origine dans « Détails ».
C'est le seul moyen d'atteindre 4 cartes sans amputer : tout reste accessible en un
clic + dans M.I.A. Si tu préfères la confluence visible par défaut (moins de cartes
au pli), c'est un petit retour arrière.

---

## 5. Captures

`docs/audits/vz-2/` — `before-*` et `after-*`, fr + en, 1280×800 et 390×844
(full-page + `-fold` pour le pli).

---

## 6. Vérifications

- `tsc --noEmit` : **vert**.
- `next build` : **vert**.
- `vitest run --pool=threads` : **913 passés** (97 fichiers).
- Playwright vz-1 (fr+en, 1280 + 390) : 14/15 verts ; 1 flake `page.goto` (timeout
  navigation sous charge, pas d'assertion) reconfirmé vert en isolation.
- Playwright vz-2 (mesure + captures) : verts.

Aucune modification du moteur, des mappers, ni des règles de détection.
