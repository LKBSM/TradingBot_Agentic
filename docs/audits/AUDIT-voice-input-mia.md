# AUDIT — Dictée vocale (micro) M.I.A sur toutes les surfaces

**Branche** : `feat/voice-input-mia-all-surfaces` (depuis `origin/main` @ `acc72d8`)
**Date** : 2026-08-20
**Statut** : implémenté, tests verts — **en attente de confirmation visuelle live du fondateur avant merge**

---

## 1. Diagnostic (le micro n'était pas absent du code)

Le symptôme de la capture — *« Micro refusé. Tu peux l'autoriser dans ton navigateur, ou
continuer au clavier »* — n'était **pas** un code manquant : c'est la chaîne i18n
`scannerChat.dictation.errors.not-allowed`, déclenchée par un **refus de permission
navigateur** sur une fonctionnalité déjà présente.

Le micro existait **uniquement** sur « Décris ta stratégie » (`DescribePanel`), via le hook
`lib/scanner-chat/use-speech-dictation.ts` (**API Web Speech du navigateur**). Ce hook était
déjà robuste : feature-detection au montage (bouton caché si non supporté → pas de bouton
mort), timeout d'écoute, transcript live jamais masqué, gestion `not-allowed / no-speech /
audio-capture / network / timeout / unknown`.

### Cartographie des surfaces M.I.A

| Surface | Route | Input | Micro avant | Micro après |
|---|---|---|---|---|
| DescribePanel | `/scanner/decrire` | `<textarea>` | ✅ | ✅ (amélioré) |
| ChatInput (**partagé**) → M.I.A Agent + ChatPanel | `/app` + toutes pages produit + landing | `<textarea>` | ❌ | ✅ |
| ZoneMiaPanel | `/zones` | `<input>` | ❌ | ✅ |
| MiaBlock (CalendarEventDetail) | `/actualites/:id` | `<input>` | ❌ | ✅ |

> Note d'architecture : `ProductShell` docke le chat `/app` (M.I.A Agent) sur **toutes** les
> pages produit — le micro y est donc présent partout via le composant partagé `ChatInput`.

---

## 2. Ligne de confidentialité (règle n°6 — aucune affirmation sans preuve)

L'API Web Speech **n'est pas garantie « purement locale »** : sur Chrome/Edge l'audio est
envoyé aux serveurs de l'éditeur du navigateur pour transcription. **Toutes les surfaces
réutilisent ce même mécanisme**, donc la même note honnête s'applique partout. La note
affichée dit (fr) :

> « La transcription est effectuée par ton navigateur **et peut transiter par ses serveurs**.
> Aucun enregistrement audio n'est conservé. »

Un test de garde (`dictation-copy-honesty.test.ts`) vérifie, **dans les 9 locales**, que la
note nomme le navigateur ET reconnaît le transit serveur, et qu'elle **ne prétend jamais**
une transcription « purement locale / sur l'appareil ».

---

## 3. Implémentation

### Étape 0 — factorisation (0 duplication de mécanisme)
- `components/dictation/MicButton.tsx` — le **bouton micro unique** partagé, 3 états visuels
  reflétant l'état réel : *inactif* (contour neutre) · *écoute* (rempli accent, `aria-pressed`)
  · **refusé** (contour ambre « alerte » — clic = réessai, le clavier reste utilisable).
- `lib/scanner-chat/use-voice-input.ts` — adaptateur partagé qui n'**ajoute** (append) que du
  texte au champ contrôlé par l'utilisateur : ce que reçoit M.I.A = exactement le texte
  visible, sans transformation cachée.
- `lib/scanner-chat/use-dictation-copy.ts` — source unique des libellés + note (aucun autre
  composant ne référence le namespace i18n).
- i18n : copie d'erreur **neutralisée** dans les 9 locales (« …au clavier » au lieu de « …écris
  ta stratégie ») pour être vraie sur chat/zones/actualités — édition **chirurgicale**
  (remplacement de sous-chaînes, CRLF/indentation/accents préservés, 4 lignes/fichier).

### Étape A — état « refusé » amélioré sur DescribePanel
Bouton micro qui **reflète l'état réel** (contour ambre quand refusé) + message existant qui
**contient déjà le chemin de ré-autorisation** (« Tu peux l'autoriser dans ton navigateur… »).
Migration vers `MicButton`/`useVoiceInput` sans changer le comportement (couverture sc2 verte).

### Étape B — branchement sur les 3 autres surfaces
`ChatInput` (→ `/app` + landing), `ZoneMiaPanel`, `MiaBlock` : même hook, même bouton, mêmes
états. CSS scoppé (`.zmia-mic`, `.pub-mia-mic`) pour que le micro reste un contour **distinct**
du bouton d'envoi rempli, sur des pages pilotées par variables CSS. Fallback clavier propre
partout (micro caché si non supporté, champ toujours utilisable si refusé).

---

## 4. Tests d'honnêteté

**Unitaires (vitest, 43 tests dédiés ; suite complète 998/998)**
- `use-voice-input.test.ts` — le transcript est **exactement** ce qui remplit le champ
  (append espace unique, respect de `maxLength`) ; refus → `denied` sans écrire dans le champ ;
  non supporté → `supported === false`.
- `MicButton.test.tsx` — `data-state` idle/listening/denied ; refusé reste cliquable ; toggle.
- `dictation-copy-honesty.test.ts` — 9 locales : note décrit le vrai mécanisme (navigateur +
  transit), jamais « purement local » ; erreurs **surface-neutres** (aucune « stratégie »).

**End-to-end (Playwright, `voice-input-mia.spec.ts` — 1280×800 et 390×844)**
- Par surface (`/app`, `/zones`, `/actualites`) : **accordée** (micro présent + dictée qui
  remplit le champ à l'identique ; `/actualites` vérifie que le backend reçoit **verbatim** le
  transcript ; `/zones` vérifie le tour utilisateur == transcript), **refusée** (message +
  champ toujours utilisable), **non supportée** (micro absent, pas de bouton mort).
- Mobile 390×844 : `/actualites` (micro présent + dictée exacte, absent si non supporté).
- Le scanner « Décris ta stratégie » reste couvert aux **deux** viewports par
  `sc2-scanner-conversationnel.spec.ts` (10/10, non régressé).

**Résultats** : tsc **0** · build **OK** · vitest **998/998** · Playwright voice **11/11** +
sc2 **10/10** + ui2-audit **7/7**.

---

## 5. Reste à faire
- **Confirmation visuelle live du fondateur** (les 4 surfaces, micro accordé/refusé) **avant
  merge sur `main`**.
- Le vrai flux Web Speech (permission OS + transcription réelle) se valide en navigateur réel ;
  les tests injectent une `SpeechRecognition` scriptée.
