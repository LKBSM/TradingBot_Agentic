# AUDIT LEG-1 — Cadre légal minimum avant encaissement

> ## ⚠️ Textes non validés par un juriste à la date du merge.
> ## Révision professionnelle à faire avant montée en volume.
>
> Les conditions d'utilisation et la politique de confidentialité livrées ici
> ont été rédigées en interne. Elles sont volontairement **prudentes** : elles
> n'affirment que ce qui est vérifiable dans le produit et n'excluent que ce qui
> peut l'être au Québec. Elles ne remplacent pas l'avis d'un avocat.

**Mission** : LEG-1 · **Branche** : `feat/leg-1-cadre-legal` · **Base** :
`origin/main` @ `9de5138` (10 septembre 2026) · **Dates** : 13 septembre 2026
(cadre légal), 14 septembre 2026 (ouverture États-Unis + remboursement automatique)
**Version des documents livrés** : `2026-09-14`

---

## 1. Pourquoi cette mission

Avant tout encaissement, un client doit pouvoir **lire** les conditions et la
politique de confidentialité, et les **accepter explicitement** avant d'arriver
sur Stripe Checkout.

Le diagnostic a montré que le point de départ n'était pas « un brouillon dans
`docs/legal/` » mais **quatre sources de texte légal qui se contredisaient** :

| Source | Langues | Servait | Problème |
|---|---|---|---|
| `docs/legal/conditions-utilisation.md` | FR | page `/conditions` | 8 clauses, cadre AMF/ACPR français |
| `src/api/routes/legal.py` (dicts `_TERMS`/`_PRIVACY`) | en, fr, de, es | `/api/v1/terms`, `/api/v1/privacy` | cadre RGPD/CNIL/AEPD, sous-traitants faux |
| bundles i18n `legal.privacy.sections.*` | 9 locales | page `/confidentialite` | placeholder assumé |
| `reports/legal/*_phase2b.md` | FR | rien | marque et offre périmées |

Un client pouvait donc lire, sur la même marque, trois descriptions
incompatibles de ses droits.

---

## 2. Ce qui existait déjà et n'a pas été refait

- Routes `/conditions` et `/confidentialite` dans les 9 locales, **lisibles sans
  compte**.
- Pied de page pointant déjà `/conditions`, `/confidentialite` et
  `contact@mia.markets`.
- **Infrastructure de consentement** : table `account_consents (doc, version,
  accepted_at)`, deux consentements obligatoires côté serveur à l'inscription,
  version tirée d'une source unique, déclaration 18+.
- **Préavis de renouvellement annuel 30 jours** (`src/billing/renewal_notices.py`).
- PAY-3 mergée : l'écran qui précède Checkout est `/abonnement`.

La mission a donc porté sur le **contenu** des textes, leur **unification**, et
le **consentement au bon endroit du tunnel** — pas sur la plomberie.

---

## 3. Décisions prises

Cinq points de la commande ne pouvaient pas être exécutés à la lettre sans
écrire quelque chose de faux. Ils ont été remontés au diagnostic ; le point 1 a
été tranché par le fondateur le 2026-09-14, les quatre autres avant
l'implémentation.

| # | Demande initiale | Décision | Motif |
|---|---|---|---|
| 1 | Territoire = **Canada et États-Unis** | **Canada et États-Unis** ✅ | Signalé au diagnostic parce que le code bloquait les US (`BLOCKED_COUNTRIES`, HTTP 451, motif *SEC Investment Advisers Act §202(a)(11)*). **Décision fondateur du 2026-09-14 : on commercialise aux deux.** Le texte, le `GeoBlockMiddleware` et `insight_v2/contract.py` ont été alignés ensemble (cf. §4.5). |
| 2 | Prix **39,99 $/mois** | **39 $/mois** | C'est le montant réellement facturé (`config/pricing.json`, source unique du front et de Stripe). Écrire 39,99 $ aurait mis un prix faux dans un document contractuel. |
| 3 | Auth = **Clerk** | **Authentification maison** | Clerk a été explicitement écarté (`docs/audits/AUDIT-pay-1.md` : « *US-only sans choix de région* »). Le nommer aurait été une fausse déclaration de sous-traitant. |
| 4 | Hébergement à préciser | **États-Unis** | `render.yaml` : « *the host runs in a US region* ». ⚠️ **À reconfirmer avant chaque déploiement** (cf. §7). |
| 5 | EFVP « réalisée » | **« en cours »** | Aucune évaluation des facteurs relatifs à la vie privée n'existe dans le dépôt. Le texte ne l'affirme donc pas, et un test l'interdit. |

Deux décisions d'architecture s'y ajoutent :

- **Documents markdown par locale** plutôt que ~80 clés juridiques dans les
  bundles i18n. La parité i18n est **stricte sur 9 locales** : mettre le texte
  légal dans les bundles aurait forcé une rédaction juridique dans 6 langues que
  personne ne peut relire. Un avocat relit un document, pas du JSON.
- **Tutoiement** dans les deux documents, pour rester cohérent avec la clause 8
  dont la formulation était imposée au tutoiement.

---

## 4. Ce qui a été livré

### 4.1 Une seule source de texte

```
docs/legal/conditions-utilisation.{fr,en,es}.md
docs/legal/politique-confidentialite.{fr,en,es}.md
```

Le **français est canonique** ; l'anglais et l'espagnol le disent dans leur
propre en-tête. Les documents sont servis **tels quels** : rien dans le code ne
les réécrit, ne les résume ni ne les paraphrase.

`src/api/routes/legal.py` a été réécrit : les dicts `_TERMS`/`_PRIVACY` ont
disparu, les endpoints ne font plus que **choisir la langue et estampiller la
version**. Les URLs publiques `/api/v1/terms` et `/api/v1/privacy` sont
conservées (elles peuvent être déclarées chez Stripe) et servent désormais
**les mêmes octets** que `/api/v1/legal/conditions` et `/api/v1/legal/privacy`.

Langues publiées : **fr, en, es**. Une locale de l'interface sans texte légal
(de, it, pt, nl, pl, ar) reçoit l'**anglais** — et le document le dit.

### 4.2 Les 12 clauses

| # | Clause | Ce qui a changé |
|---|---|---|
| 1 | Nature du service | « pas un conseil **personnalisé** » → **aucun conseil, aucun signal, aucune recommandation, aucune indication d'intervention**. Le qualificatif laissait entendre qu'il existait un conseil non personnalisé. + « la décision t'appartient entièrement ». |
| 2 | Risque | Statistique **« 74 % à 89 % des comptes particuliers perdent de l'argent » supprimée** (chiffre ESMA, non sourcé, inapplicable ici). Remplacée par le risque de perte, l'effet de levier, et « les mesures passées ne s'appliquent pas aux situations à venir ». |
| 3 | Âge | 18+ **ou l'âge de majorité local s'il est supérieur** (plusieurs provinces et États sont à 19 ou 21). |
| 4 | Territoire | Exclusion US/UK/OFAC → **offert au Canada et aux États-Unis**. Le texte, le geo-block et le contrat `insight_v2` ont été alignés ensemble (cf. §4.5). |
| 5 | Compte | **Nouveau** : un compte par personne, un courriel par compte, pas de partage, suspension motivée en cas d'abus. |
| 6 | Données de marché | **Nouveau et obligatoire** : usage personnel seulement, interdiction de redistribuer / revendre / republier / extraire automatiquement. L'ancien texte ne couvrait que « les Analyses », pas les données — or c'est la licence de données qui l'impose. |
| 7 | Prix et facturation | **Nouveau** : 39 $/mois ou 348 $/an, USD explicite, Stripe, aucune carte conservée, renouvellement automatique, préavis 30 jours à l'annuel, préavis en cas de changement de prix. |
| 8 | Résiliation et remboursement | **Nouveau**, formulation imposée **reproduite mot pour mot**, sans durcissement. Garantie 14 jours à l'annuel ; primauté de la *Loi sur la protection du consommateur* au Québec. |
| 9 | Disponibilité | **L'exclusion totale de responsabilité a été retirée** (inopposable au Québec, art. 10 LPC). Remplacée par : « tel quel », données tierces, interruptions possibles, efforts raisonnables, responsabilité pour faute selon la loi, primauté des dispositions d'ordre public. Ni arbitrage obligatoire, ni renonciation à l'action collective. |
| 10 | Modifications | **Nouveau** : préavis raisonnable par courriel, version et date affichées. |
| 11 | Droit applicable | **Québec, Canada** (l'ancien texte disait seulement « exploité depuis le Québec », ce qui n'est pas une clause de droit applicable, et invoquait l'AMF française). |
| 12 | Contact | `loukmanebessam@gmail.com` → **`contact@mia.markets`**, cohérent avec le pied de page. |

### 4.3 Politique de confidentialité (Loi 25)

Le placeholder de `/confidentialite` est remplacé par un document réel en 12
sections : responsable nommé, liste **fermée** de ce qui est collecté, finalités,
hébergement aux États-Unis, durées, droits (dont **retrait du consentement** et
**suppression**), sous-traitants réellement utilisés, dictée vocale, **registre
des incidents** et procédure de notification, témoins, modifications, contact.

Corrections de fond par rapport au texte précédent :

- sous-traitants : **Stripe, Anthropic, TwelveData, l'hébergeur**. **Ni Clerk, ni
  Telegram** (ni l'un ni l'autre ne traite les données du produit vendu) ;
- Anthropic : il est dit explicitement que **le contenu des questions posées à
  l'agent lui est transmis** pour produire la réponse — l'ancien texte affirmait
  « payload signal, pas de données personnelles », ce qui n'est plus vrai depuis
  M.I.A Agent ;
- cadre **Loi 25 / Commission d'accès à l'information du Québec** au lieu de
  RGPD / CNIL / AEPD / BfDI ;
- la section **dictée vocale** de SC-2, déjà honnête, a été conservée.

### 4.4 Consentement avant paiement

Sur `/abonnement`, juste avant les deux boutons qui mènent à Stripe :

- une case **non pré-cochée**, jamais restaurée d'un stockage — elle est toujours
  un acte du client sur la version affichée à cet instant ;
- les **deux liens** vers les documents complets, ouverts dans un nouvel onglet
  pour que cocher ne soit pas perdu ;
- **les deux CTA sont inactifs** tant que la case n'est pas cochée, et la raison
  est **écrite** (`aria-describedby` la relie aux boutons) — un bouton grisé qui
  n'explique rien n'est pas une explication ;
- au clic, le consentement est enregistré **avant** de partir vers Stripe. Si
  l'enregistrement échoue, **le checkout ne démarre pas** : on ne prend pas
  d'argent sans trace de ce qui a été accepté ;
- une garde côté code (`if (!consented) return`) double l'attribut `disabled`.

Côté serveur, `POST /api/auth/consents` : le client envoie seulement « j'accepte »,
et **la version est décidée par le serveur** à partir de la source unique. Un
client ne peut donc pas estampiller un consentement sur une version qu'il aurait
inventée ou gardée en cache. Seuls `(doc, version, accepted_at)` sont écrits —
rien d'autre. L'écriture est **idempotente** par version et **append-only** :
accepter une nouvelle version ajoute une ligne sans effacer l'historique.

### 4.5 Ouverture aux États-Unis (décision fondateur, 2026-09-14)

Le diagnostic avait mis ce point de côté parce que **le texte et le code se
contredisaient** : les quatre textes annonçaient l'exclusion des États-Unis et
`BLOCKED_COUNTRIES` la faisait respecter par un HTTP 451 — alors qu'en
production `GEO_BLOCK_DISABLED=1` (l'hôte est lui-même en région US) rendait ce
blocage inopérant. Autrement dit, on **annonçait une exclusion qu'on
n'appliquait pas**.

La décision lève la contradiction dans le bon sens. Trois endroits bougent
**ensemble**, parce que les laisser diverger est exactement ce qui a créé le
problème :

1. **clause 4** des conditions, dans les trois langues : « Canada **et
   États-Unis** » ;
2. **`BLOCKED_COUNTRIES`** : `"US"` retiré (le Royaume-Uni et les pays OFAC
   restent) ;
3. **`insight_v2/contract.py`** : `jurisdiction_blocked` passe de
   `("US","QC","UK","OFAC")` à `("UK","OFAC")`.

Un test refuse désormais que le texte annonce un territoire que le code bloque —
la divergence ne peut plus réapparaître silencieusement.

La posture qui rend cette ouverture défendable est celle que le produit tient
déjà et que la clause 1 énonce : **publication impersonnelle**. Aucun conseil,
aucune recommandation, aucun signal, aucune prise en compte de la situation
d'un utilisateur. Cette posture est la substance de la décision — elle reste à
valider par un juriste, comme le reste de ces textes.

### 4.6 La garantie de 14 jours, honorée automatiquement

La clause 8 promet une garantie de 14 jours à l'annuel. Elle est désormais
**exécutée par le produit**, plus par un courriel au fondateur.

**La règle est pure et isolée** (`src/billing/refund_guarantee.py`) : cadence,
date de paiement, maintenant → couvert ou non. Testable sans Stripe, sans base
et sans horloge. Deux choix y sont explicites :

- **la fenêtre part du PAIEMENT**, pas de l'inscription ni du début de période —
  c'est la lecture littérale de la clause, et un renouvellement rouvre donc 14
  jours ;
- **un paiement daté dans le futur** (décalage d'horloge) est traité comme
  « à l'instant » : une dérive d'horloge ne doit jamais coûter sa garantie à
  quelqu'un.

**Deux endpoints** : `GET /api/billing/refund-eligibility` (lecture seule, sert
à décider si on **propose** le remboursement — personne ne voit un bouton qui
va le refuser) et `POST /api/billing/refund`.

L'ordre du remboursement compte : on rembourse **intégralement**, on annule
l'abonnement, puis on écrit `suspended` **localement** — l'accès est révoqué
tout de suite au lieu d'attendre le webhook `charge.refunded` (qui arrive quand
même, et écrit la même chose : il est idempotent). Si le remboursement réussit
mais que l'annulation échoue, **on ne signale pas d'échec** : l'argent est
revenu, c'est ce que le client demandait ; l'abonnement orphelin est journalisé
pour l'exploitant.

**Un refus n'est jamais un cul-de-sac.** Chaque motif porte un message qui dit
ce que le client *peut* faire : au mensuel, résilier et garder l'accès jusqu'à
la fin de la période payée ; hors fenêtre, idem. Et la clause 8 rappelle que la
*Loi sur la protection du consommateur* prime sur cette garantie commerciale.

Côté interface, le bloc n'apparaît que si la fenêtre est ouverte, affiche une
**date limite** plutôt qu'un nombre de jours (aucune règle de pluriel à rater en
neuf langues), et **demande confirmation** avant d'agir.

### 4.7 Trois correctifs de fond trouvés en chemin

**1. Le geo-block 451-ait la page des conditions.**
`/api/v1/legal/conditions` **n'était pas dans la liste d'exemption du geo-block** :
un visiteur d'un pays bloqué recevait un **HTTP 451 sur la page des conditions
elle-même** — c'est-à-dire précisément sur le document qui lui explique pourquoi
il est bloqué. Les quatre chemins légaux ont été ajoutés à `ALLOWED_PATHS`.

**2. Le rendu markdown cassait le gras et les puces.**
Repéré sur une capture, pas par un test : le rendu traitait chaque **ligne
source** comme une ligne rendue. Or les documents sont coupés à ~80 colonnes pour
la relecture, donc une phrase enjambe un retour à la ligne en permanence. Deux
conséquences visibles sur les pages publiques :

- un `**gras**` à cheval sur une coupure affichait ses astérisques en clair —
  « *est \*\*responsable de la protection* » était lisible tel quel ;
- la continuation d'une puce **sortait de la liste** et devenait un paragraphe
  orphelin dessous.

`render-markdown.tsx` joint désormais les lignes d'un paragraphe et d'une
citation (comme le fait markdown) et rattache une ligne indentée à la puce
qu'elle continue. 9 tests verrouillent le comportement, dont deux qui rendent les
vrais documents et vérifient qu'il ne reste **aucun `**` visible**.

**3. `/abonnement` était capturé en page d'erreur dans la couverture DS-1.**
Antérieur à cette mission (`ds-mock` ne servait pas `/api/billing/*`), mais c'est
l'écran qui porte le consentement : il devait devenir relisible. Les endpoints
légaux et de facturation ont été ajoutés au harnais, et les **24 captures** des
trois pages touchées (`conditions`, `confidentialite`, `abonnement` × 4 thèmes ×
2 viewports) ont été régénérées.

---

## 5. Tests

| Suite | Résultat |
|---|---|
| `tests/test_legal_endpoints.py` (réécrit) | **86 passés** |
| `tests/test_leg1_consent.py` (nouveau) | **13 passés** |
| `tests/test_leg1_refund.py` (nouveau — garantie 14 jours) | **21 passés** |
| Bloc légal + paiement (`test_account_auth`, `test_pay3_payment_journey`, `test_billing`, `test_account_billing`, `test_geo_block`, `test_subscription_gate_paid_only` + les 3 ci-dessus) | **241 passés** |
| Sélection `-k "insight or contract or compliance or disclaimer or claims"` (consommateurs du contrat `insight_v2`) | **171 passés** |
| `webapp/tests/leg1-legal-copy.test.ts` (nouveau) | **47 passés** |
| `webapp/components/billing/__tests__/leg1-consent-gate.test.tsx` (nouveau) | **18 passés** |
| `webapp/components/billing/__tests__/leg1-refund-ui.test.tsx` (nouveau) | **9 passés** |
| `webapp/lib/legal/__tests__/render-markdown.test.tsx` (nouveau) | **9 passés** |
| Garde-fous existants (claims-cleanup, no-free-tier, locale-parity, txt1-copy, cln1-copy, ui2-copy-honesty, pricing-prix-1) | **111 passés** |
| **Suite vitest complète** | **131 fichiers, 1259 tests, 0 échec** |
| `tsc --noEmit` | **0 erreur** |
| Playwright `leg1-legal.spec.ts` (1280×800 + 390×844, consentement **et** remboursement) | **20 passés** |
| Playwright couverture DS-1 des 3 pages touchées | **24 passés** |

### Ce que les tests verrouillent

- **le bouton de paiement est inactif tant que la case n'est pas cochée** (aux
  deux CTA, dans les 3 langues) ;
- **version et horodatage enregistrés**, et rien d'autre ; le client ne choisit
  pas la version ; la version enregistrée est celle du document servi ;
- **les 3 locales portent les mêmes sections, dans le même ordre** — une clause
  manquante en espagnol est l'échec qui compte ;
- **aucune occurrence** de « vente finale », « non remboursable », « final sale »,
  « no refund », « venta final », « sin reembolso » (+ variantes) — **dans les 6
  documents et dans les 9 bundles i18n** ;
- **aucun futur prédictif** sur les marchés ;
- **aucune exclusion totale de responsabilité** ne peut revenir ;
- la statistique 74 %–89 % ne peut pas revenir ;
- ni Clerk ni Telegram ne peuvent réapparaître comme sous-traitants ;
- la politique ne peut pas affirmer qu'une EFVP **a été** réalisée ;
- le prix écrit dans les conditions est comparé à `pricing.generated.ts` — si le
  tarif change sans que les conditions suivent, le test casse ;
- un document légal reste joignable depuis un pays géo-bloqué ;
- **le texte et le geo-block s'accordent sur le territoire** : un retour de
  `"US"` dans `BLOCKED_COUNTRIES` casse le test, puisque les conditions
  annoncent les États-Unis ;
- **la garantie de 14 jours** : annuel seulement, comptée depuis le paiement,
  jour 14 dedans / jour 15 dehors, un décalage d'horloge ne la fait pas perdre,
  l'argent ne part qu'une fois, l'accès est révoqué sans attendre le webhook, un
  refus dit toujours ce qu'on peut faire à la place, et l'annulation qui échoue
  après un remboursement réussi ne se signale pas comme un échec.

### Playwright

`webapp/tests/e2e/leg1-legal.spec.ts`, aux deux viewports **1280×800** et
**390×844** : les deux pages rendent le vrai document avec sa version et sans
compte, le placeholder a disparu, **un seul bloc d'avertissement par page**, et
l'écran de consentement se comporte comme décrit au §4.4.

---

## 6. Fichiers touchés

**Ajoutés** — 6 documents `docs/legal/*.{fr,en,es}.md`, `docs/legal/README.md`
(procédure de modification), `src/billing/refund_guarantee.py` (la règle des 14
jours, pure), `webapp/components/legal/LegalDocument.tsx`,
`tests/test_leg1_consent.py`, `tests/test_leg1_refund.py`,
`webapp/tests/leg1-legal-copy.test.ts`,
`webapp/components/billing/__tests__/leg1-consent-gate.test.tsx`,
`webapp/components/billing/__tests__/leg1-refund-ui.test.tsx`,
`webapp/lib/legal/__tests__/render-markdown.test.tsx`,
`webapp/tests/e2e/leg1-legal.spec.ts`, ce rapport.

**Supprimés** — `docs/legal/conditions-utilisation.md`,
`webapp/components/legal/ConditionsDocument.tsx`.

**Modifiés** — `src/api/routes/legal.py` (réécrit), `src/api/routes/accounts.py`,
`src/api/routes/account_billing.py`, `src/api/account_store.py`,
`src/api/middleware/geo_block.py`, `src/billing/stripe_client.py`,
`src/intelligence/insight_v2/contract.py`,
`src/intelligence/chatbot/demo_agent.py`, `webapp/lib/billing/api-client.ts`,
`tests/test_geo_block.py`,
`webapp/app/[locale]/(site)/{conditions,confidentialite}/page.tsx`,
`webapp/components/billing/SubscriptionPanel.tsx`,
`webapp/lib/auth/api-client.ts`, `webapp/lib/legal/render-markdown.tsx`,
`webapp/messages/*.json` (9 locales), `webapp/tests/e2e/ds-mock.ts`,
`docs/audits/ds-1/coverage/*.png` (24 captures régénérées),
`tests/test_account_auth.py`, `tests/test_legal_endpoints.py`.

---

## 7. Ce qui reste ouvert

1. **Relecture juridique.** C'est la seule action qui lève l'avertissement en
   tête de ce rapport. Elle porte aussi, désormais, sur la **posture de
   publication impersonnelle** qui fonde l'ouverture aux États-Unis (§4.5).
2. **Confirmer le pays d'hébergement avant chaque déploiement.** La politique dit
   « États-Unis » sur la foi de `render.yaml`. Un changement d'hébergeur change
   le texte et l'EFVP. Ce n'est pas un risque, seulement une phrase à garder vraie.
3. **EFVP à réaliser.** Le texte dit « en cours » — c'est vrai aujourd'hui et ça
   ne le restera pas indéfiniment. Une fois faite, remplacer par « réalisée le
   {date} » ; le test qui interdit l'affirmation devra être ajusté avec elle.
4. **Réactiver le geo-block en production.** `GEO_BLOCK_DISABLED=1` dans
   `render.yaml` désactive le blocage **en entier** — le Royaume-Uni et les pays
   OFAC compris — parce que l'hôte se 451-ait lui-même en région US. Maintenant
   que les États-Unis sont ouverts, cette raison a disparu : le blocage peut être
   réactivé pour couvrir les juridictions qui restent exclues.
5. **Remboursement : Stripe doit pouvoir rembourser.** L'automatisation appelle
   `Refund.create` avec la clé secrète : la clé doit en avoir le droit, et un
   paiement déjà remboursé à la main dans le tableau de bord fera échouer
   l'appel (message explicite côté client). Le webhook `charge.refunded` reste
   câblé et écrit le même état — il est idempotent.
6. **Consentement à l'inscription conservé.** Il y a désormais deux points de
   recueil (inscription et avant paiement). C'est volontaire : ils enregistrent
   la même version et la table est append-only.
7. **`reports/legal/*_phase2b.md`** restent dans le dépôt comme archive. Ils
   décrivent une offre qui n'existe plus et **ne doivent pas servir de référence**.
