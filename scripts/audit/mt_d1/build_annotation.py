"""MT-D1 Section 3C — build the annotation artifacts from manifest.json."""
import csv
import json
from pathlib import Path

OUT = Path("docs/audits/ECHANTILLON-DETECTION-2026-07-29")
man = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))

ROWS = []
for e in man["events"]:
    ROWS.append((e["id"], "3A-evenement-emis", e["tf"], e["ts"],
                 f"{e['type']} {e['direction']} niveau {e['level']}",
                 "Cet événement ÉMIS est-il un vrai BOS/CHOCH ?", e["image"]))
for e in man["non_events"]:
    reason = {"wick_only_close_inside": "franchi en meche, cloture en deca",
              "level_already_broken": "niveau deja franchi (garde anti-repetition)"}[e["reason"]]
    ROWS.append((e["id"], "3B-non-evenement", e["tf"], e["ts"],
                 f"franchissement {e['side']} SANS evenement — {reason}",
                 "Le moteur a-t-il eu raison de NE PAS emettre ici ?", e["image"]))
for e in man["extra_events"]:
    ROWS.append((e["id"], "8-evenement-ajoute-si-meche", e["tf"], e["ts"],
                 f"{e['direction']} niveau {e['level']} (emis seulement en mode meche)",
                 "Cet evenement manquant est-il un vrai evenement ou du bruit ?", e["image"]))
oc = man["open_case"]
ROWS.append(("OPENCASE", "4-cas-ouvert", oc["tf"], oc["bos_ts"],
             "Derive 24->28 juil sans evenement ; creux protege jamais cloture sous",
             "D'accord : derive correcte sans CHOCH manque ?", oc["image"]))

# CSV — one blank verdict/comment pair; each trader fills their own copy.
with (OUT / "ANNOTATION.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["case_id", "section", "unite", "horodatage_UTC", "description",
                "question", "image", "annotateur", "verdict (d_accord/pas_d_accord/incertain)", "commentaire"])
    for r in ROWS:
        w.writerow([*r, "", "", ""])
print(f"ANNOTATION.csv: {len(ROWS)} cas")

# README viewer with embedded images grouped by section.
def block(title, ids, note):
    lines = [f"## {title}\n", note, ""]
    for r in ROWS:
        if r[0] in ids:
            lines += [f"### {r[0]} — {r[2]} — {r[3]}",
                      f"*{r[4]}*  ",
                      f"**Question : {r[5]}**  ",
                      f"Verdict : ☐ d'accord ☐ pas d'accord ☐ incertain — Commentaire : ______",
                      "", f"![{r[0]}](images/{r[6]})", ""]
    return "\n".join(lines)

ev_ids = [e["id"] for e in man["events"]]
non_ids = [e["id"] for e in man["non_events"]]
ex_ids = [e["id"] for e in man["extra_events"]]

readme = f"""# Échantillon de détection à annoter — XAUUSD H1 & M15 — 2026-07-29

**Mission MT-D1.** Ce dossier n'affirme rien : il **montre** ce que le moteur
détecte et ce qu'il écarte, pour que **vous, le trader**, jugiez. Aucune règle
n'a été modifiée pour le produire. Les images couvrent une période réelle de
~60 jours (données Twelve Data, horodatages **UTC**).

## Comment annoter

1. Copiez `ANNOTATION.csv` en `ANNOTATION-<votre-nom>.csv` (un fichier par
   trader, annotation **indépendante** — on comparera ensuite les jugements).
2. Pour chaque cas, renseignez `verdict` (`d_accord`, `pas_d_accord`,
   `incertain`) et un `commentaire` libre. Mettez votre nom dans `annotateur`.
3. Ce `README.md` est le **visualiseur** : il montre chaque graphique avec sa
   question. Les cases à cocher ici sont pour une lecture papier ; la saisie
   qui sera dépouillée est le CSV.

## Légende des graphiques

- Chandeliers verts/rouges = bougies. Ligne pointillée bleue = **la bougie du
  cas**. Ligne mauve = **niveau structurel** (creux/sommet fractal concerné).
- Section 3A (EVT) : événements **émis** par le moteur.
- Section 3B (NON) : moments où un swing a été **franchi sans** qu'aucun
  événement soit émis — avec la condition qui a bloqué. **C'est la partie que le
  produit ne montre jamais aujourd'hui.**
- Section 8 (EXTRA) : événements qui **seraient** émis si le franchissement se
  faisait « en mèche » au lieu de « en clôture » (le levier le plus influent de
  l'analyse de sensibilité) — vrais événements ou bruit ?
- OPENCASE : le cas ouvert DG-1 (24→28 juillet).

---

{block("Section 4 — Cas ouvert DG-1", ["OPENCASE"], "La dérive du 24 au 28 juillet a-t-elle été correctement laissée sans événement ? (ligne orange = creux protégé, jamais clôturé sous)")}

---

{block("Section 3A — 25 événements ÉMIS", ev_ids, "Pour chacun : est-ce un vrai BOS/CHOCH selon votre lecture ?")}

---

{block("Section 3B — 25 franchissements SANS événement (faux négatifs potentiels)", non_ids, "Pour chacun : le moteur a-t-il eu raison de NE PAS émettre ?")}

---

{block("Section 8 — 15 événements ajoutés en mode « mèche »", ex_ids, "Pour chacun : vrai événement manquant, ou bruit qu'on a raison d'écarter ?")}
"""
(OUT / "README.md").write_text(readme, encoding="utf-8")
print("README.md written")
