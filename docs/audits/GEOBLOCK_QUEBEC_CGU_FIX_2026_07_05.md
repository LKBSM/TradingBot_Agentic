# Déblocage du Québec (geo-block) + alignement des CGU — 2026-07-05

Décision fondateur (suite du nettoyage claims `FOOTER_CLAIMS_CLEANUP_2026_07_04.md`,
section « Découverte majeure ») : le Québec est la **juridiction de rattachement**
de l'entreprise (stratégie légale Loi 25 + LPC québécoises). Son blocage venait
d'un boilerplate généré et contredisait la réalité.

## Changements

| Surface | Avant | Après |
|---|---|---|
| `src/api/middleware/geo_block.py` | `BLOCKED_REGIONS = {"CA-QC"}` → HTTP 451 pour tout client québécois (headers CDN) | `BLOCKED_REGIONS = set()` — le Québec est servi ; mécanique région conservée. US/GB/OFAC inchangés |
| `docs/legal/conditions-utilisation.md` §4 (rendu par `/conditions` + `/api/v1/legal/conditions`) | « indisponible aux personnes résidant aux États-Unis, au Québec (Canada), au Royaume-Uni… » | Québec retiré + mention explicite « exploité depuis le Québec (Canada) et y est pleinement disponible ». **Version bumpée 2026-04-28 → 2026-07-05** (horodatage de consentement Loi 25/RGPD) |
| `src/api/routes/legal.py` `_TERMS` (en/fr/de/es) | Québec listé comme juridiction exclue (4 langues) | Québec retiré + disponibilité affirmée (4 langues). `LAST_UPDATED` → 2026-07-05 (source unique des versions terms/privacy/conditions et du stamp de consentement via `accounts.py`) |
| Tests | `test_quebec_in_blocked_regions`, `TestQuebecBlocking` (451 attendu), version « 2026-04-28 » en dur ×3 | `test_quebec_not_in_blocked_regions` + `TestQuebecServed` (200 attendu — verrou anti-régression), versions testées via `CONDITIONS_VERSION` importé + test de lockstep en-tête markdown ↔ code |

## Cohérence claims ↔ comportement

Après ce changement, la divulgation légale décrit exactement ce que le
middleware fait : US + GB + pays OFAC refusés (HTTP 451 quand les headers CDN
sont présents), Québec et reste du monde servis. Aucun périmètre marketing
n'est annoncé (cf. nettoyage claims du 2026-07-04).

## Impact consentement

Le bump de version fait que toute nouvelle inscription horodate son
consentement sur « 2026-07-05 ». Phase de test personnel : aucun utilisateur
existant à re-consentir.

## Question laissée ouverte (décision produit/légal future)

**Le Royaume-Uni reste bloqué** (FCA financial promotion regime). Certains
documents de stratégie l'incluent en Phase 1 (`dev_focus_pivot_2026_05_27`),
d'autres non (`legal_bootstrap_strategy` : geo-restrict FR/BE/CH/LU). Statu quo
conservé (bloqué = prudent) ; à trancher explicitement avant toute ouverture UK.

## Validation

- `tests/test_geo_block.py` + `test_legal_endpoints.py` + `test_account_auth.py` : 63 verts.
- `test_disclaimers.py` + `test_subscription_gate_freemium.py` + `test_billing.py` + `test_account_billing.py` : 68 verts.
- Suite complète exécutée par la CI (algo-tests) — désormais réellement bloquante (CI 6/6 verte depuis PR #31).
