# Mentions Légales — Template auto-entrepreneur

**Date** : 2026-05-26
**Usage** : remplacer les placeholders `[XXX]` par les valeurs réelles, publier sur `/mentions-legales`, lien obligatoire dans footer.

---

## Mentions Légales — M.I.A. Markets

**Dernière mise à jour** : 2026-05-XX

### 1. Éditeur du site

- **Nom commercial** : M.I.A. Markets
- **Éditeur** : [PRÉNOM NOM]
- **Statut juridique** : Micro-entrepreneur (Auto-entrepreneur, régime micro-social simplifié)
- **SIRET** : [XXX XXX XXX XXXXX]
- **SIREN** : [XXX XXX XXX]
- **Code APE / NAF** : [62.01Z — Programmation informatique] (ou 63.11Z selon activité déclarée)
- **TVA intracommunautaire** : [FRXXXXXXXXXXX] (si dépassement seuil franchise en base, sinon mention "Franchise en base de TVA — TVA non applicable, art. 293 B du CGI")
- **Adresse du siège social** : [Adresse complète]
- **Téléphone** : [+33 X XX XX XX XX] (optionnel pour AE — mention "Disponible sur demande" acceptée)
- **Email** : `contact@mia.markets`

### 2. Directeur de la publication

[PRÉNOM NOM], en qualité d'éditeur.

### 3. Hébergement

- **Hébergeur principal** : Fly.io Inc.
- **Adresse** : 2261 Market Street #4990, San Francisco, CA 94114, USA
- **Site web** : https://fly.io
- **Région d'hébergement** : Paris (cdg) avec failover EU

- **Hébergeur frontend** : Vercel Inc.
- **Adresse** : 340 S Lemon Ave #4133, Walnut, CA 91789, USA
- **Site web** : https://vercel.com

### 4. Propriété intellectuelle

L'ensemble des contenus présents sur le site M.I.A. Markets (textes, graphiques, logos, icônes, sons, vidéos, code source) sont la propriété exclusive de l'Éditeur, à l'exception :

- des **données de marché historiques** (XAUUSD, EURUSD) publiques utilisées à des fins d'analyse
- des **données du calendrier économique** fournies par Trading Economics dans le cadre d'une licence commerciale
- des **modèles statistiques open-source** mentionnés dans la section Méthodologie (HMM, HAR-RV, BOCPD, ACI conformal, etc.) sous licences respectives de leurs auteurs
- des **technologies tierces** utilisées (Next.js, Tailwind, Anthropic API, Stripe, etc.) sous leurs licences respectives

Toute reproduction, représentation, modification, publication, transmission ou exploitation totale ou partielle du contenu du site sans autorisation écrite préalable de l'Éditeur est interdite et constitue une contrefaçon sanctionnée par les articles L. 335-2 et suivants du Code de la propriété intellectuelle.

### 5. Marques

Le nom et logo "M.I.A. Markets" sont des marques d'usage de l'Éditeur. Un dépôt INPI sera effectué à M3-M6 (post-traction commerciale validée).

### 6. Liens hypertextes

Les liens hypertextes mis en place dans le cadre du site en direction d'autres ressources présentes sur le réseau Internet ne sauraient engager la responsabilité de l'Éditeur.

### 7. Données personnelles

Le traitement des données personnelles est régi par notre **Politique de Confidentialité** accessible à `/privacy`.

Conformément au Règlement (UE) 2016/679 (RGPD) et à la loi "Informatique et Libertés" du 6 janvier 1978 modifiée, l'Utilisateur dispose d'un droit d'accès, de rectification, de suppression, de limitation, de portabilité et d'opposition concernant ses données personnelles.

**Pour exercer ces droits** : `privacy@mia.markets`

**Autorité de contrôle** : Commission Nationale de l'Informatique et des Libertés (CNIL), 3 Place de Fontenoy, TSA 80715, 75334 PARIS CEDEX 07. Site web : www.cnil.fr

### 8. Conditions Générales d'Utilisation et de Vente

L'utilisation du Service est soumise aux Conditions Générales d'Utilisation et de Vente (CGU/CGV) accessibles à `/cgu`.

### 9. Médiation de la consommation

Conformément à l'article L. 612-1 du Code de la consommation, en cas de litige, l'Utilisateur peut recourir gratuitement au service de médiation [Nom plateforme — à compléter dès adhésion M2 : CM2C ou MEDICYS] :

- **Site web** : [URL]
- **Adresse** : [Adresse]
- **Email** : [Email]

### 10. Crédits

- **Design** : inspirations Bloomberg Terminal, Linear, Stripe, Pitchbook
- **Méthodologie algorithmique** : open-source academic sources (cf. page Méthodologie)
- **Polices** : Inter (Google Fonts, SIL Open Font License)

---

## ⚙️ Checklist pré-publication

- [ ] Création auto-entreprise sur `autoentrepreneur.urssaf.fr` → récupération SIRET sous 7-14 jours
- [ ] Choix code APE : 62.01Z (programmation) recommandé pour SaaS
- [ ] Souscription compte bancaire pro (Shine / Qonto) si CA annuel anticipé > 10 k€
- [ ] Création emails `contact@`, `privacy@`, `security@`, `support@` (ProtonMail Custom Domain $5/mo ou Fastmail $5/mo)
- [ ] Achat nom de domaine `mia.markets` (~15 €/an chez Namecheap ou OVH)
- [ ] Adhésion médiateur conso (à M2 — économie initiale de 150 €)
- [ ] Remplacer tous les `[XXX]` par les valeurs réelles
- [ ] Publication sur `/mentions-legales`
- [ ] Lien dans footer + page CGU

---

## Notes pratiques

### Adresse du siège social
Pour un auto-entrepreneur, l'adresse personnelle est acceptée. Mais elle apparaîtra **publiquement** sur les mentions légales. Alternatives :

1. **Domiciliation commerciale** : ~30-50 €/mois via Sedomicilier, Société.com, etc. Une adresse pro sans louer de bureau.
2. **Boîte postale numérique** : Lostik, Smiile, Doomicile (~10-15 €/mois)
3. **Adresse perso** : 0 € mais publique → à considérer selon ta préférence vie privée

Recommandé bootstrap : **adresse perso** (économie ~360 €/an), à migrer en domiciliation commerciale à M3-M6 si tu veux séparer vie pro/perso.

### Numéro de téléphone
Non obligatoire pour AE en B2C en ligne. Recommandation : ne **pas** afficher de numéro public (anti-spam). Mentionner "Disponible sur demande pour litiges nécessitant un échange téléphonique" suffit.

### TVA
**Tant que tu es en franchise en base** (CA HT < 36 800 €/an services), tu écris sur tes factures : *"TVA non applicable, art. 293 B du CGI"*. Tu ne factures pas la TVA et tu n'as pas de numéro de TVA intracommunautaire.

**Quand tu dépasses le seuil** (notification URSSAF), tu obtiens un numéro de TVA intracommunautaire et tu commences à facturer 20 %. C'est généralement à M6-M12 si traction.

### Dépôt INPI marque
Optionnel en V0. Coût ~250 € pour une marque française. À faire quand tu as validé que "M.I.A. Markets" est ta marque définitive (post-PMF, M6-M9).
