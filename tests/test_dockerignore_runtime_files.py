"""Les fichiers que le service LIT à l'exécution doivent entrer dans l'image.

Pourquoi ce fichier existe
--------------------------
Deux pannes de production identiques, à treize jours d'intervalle, ont la même
forme : un fichier présent dans le dépôt, absent de l'image déployée, et une
erreur qui ne ressemble pas à sa cause.

  * ``scripts/stripe_preflight.py`` — « No such file or directory » dans le Shell
    Render. Cause réelle : le fichier n'était pas sur ``main``.
  * ``docs/legal/*.md`` — ``/api/v1/terms``, ``/api/v1/legal/conditions`` et
    ``/api/v1/privacy`` répondaient **503** en production depuis le 2026-09-13.
    Cause : ``.dockerignore`` excluait ``docs/`` en bloc, alors que ces documents
    ne sont pas de la documentation — ce sont les DONNÉES que le service sert, et
    le contrat que le client doit lire avant de payer.

Le ``Dockerfile`` fait ``COPY . .`` : ce qui entre dans l'image est donc décidé
uniquement par ``.dockerignore``. Ce test rejoue ses règles de correspondance sur
les chemins réellement lus par le code, pour que l'écart se voie ici plutôt qu'en
production.

Les règles reproduites (moby/patternmatcher, et NON .gitignore) : chaque chemin
est confronté à TOUS les motifs dans l'ordre, un motif correspond s'il vise le
chemin ou l'un de ses parents, et **le dernier motif qui correspond l'emporte** —
c'est ce qui permet à ``!docs/legal/`` de rattraper ``docs/``. Un ``*`` ne
traverse pas les ``/``, donc ``*.md`` ne vise que la racine.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKERIGNORE = REPO_ROOT / ".dockerignore"


def _patterns() -> list[str]:
    lines = DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
    return [
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    ]


def _segments_match(pat: list[str], tgt: list[str]) -> bool:
    """Compare segment par segment ; ``**`` absorbe zéro segment ou plus."""
    if not pat:
        # Motif entièrement consommé : il vise ce chemin ou l'un de ses parents.
        return True
    if not tgt:
        # Motif plus spécifique que le chemin : il ne le vise pas.
        return False
    if pat[0] == "**":
        return any(_segments_match(pat[1:], tgt[i:]) for i in range(len(tgt) + 1))
    if not fnmatch.fnmatchcase(tgt[0], pat[0]):
        return False
    return _segments_match(pat[1:], tgt[1:])


def _pattern_hits(pattern: str, path: str) -> bool:
    """Le motif vise-t-il ce chemin, ou l'un de ses répertoires parents ?"""
    return _segments_match(
        [p for p in pattern.strip("/").split("/") if p],
        path.split("/"),
    )


def is_excluded(path: str) -> bool:
    """Le dernier motif qui correspond décide. Un ``!`` ré-inclut."""
    excluded = False
    for raw in _patterns():
        negated = raw.startswith("!")
        pattern = raw[1:] if negated else raw
        if _pattern_hits(pattern, path):
            excluded = not negated
    return excluded


# ---------------------------------------------------------------------------
# Le moteur de correspondance lui-même — sans quoi les tests suivants ne
# prouveraient rien (un matcher qui dit « inclus » pour tout passerait).
# ---------------------------------------------------------------------------

def test_the_matcher_reproduces_dockerignore_semantics():
    assert _pattern_hits("docs", "docs/legal/x.md")          # parent
    assert _pattern_hits("docs/legal", "docs/legal/x.md")    # parent plus précis
    assert not _pattern_hits("docs/legal", "docs/audits/x.md")
    # `*` ne traverse pas les `/` : `*.md` ne vise que la racine.
    assert _pattern_hits("*.md", "README.md")
    assert not _pattern_hits("*.md", "docs/legal/x.md")
    assert not _pattern_hits("test_*.py", "tests/test_x.py")
    # `**` absorbe zéro segment ou plus, mais la SUITE du motif doit correspondre.
    # Une implémentation qui renvoie vrai dès qu'elle voit `**` excluait tout le
    # dépôt à cause de `**/node_modules` — et prétendait donc que rien n'entre
    # dans l'image.
    assert _pattern_hits("**/node_modules", "webapp/node_modules/x.js")
    assert _pattern_hits("**/node_modules", "node_modules/x.js")
    assert not _pattern_hits("**/node_modules", "src/api/routes/legal.py")
    assert not _pattern_hits("**/.next", "config/pricing.json")


# ---------------------------------------------------------------------------
# Ce que le service LIT à l'exécution — doit entrer dans l'image
# ---------------------------------------------------------------------------

#: Chemin → qui le lit, et ce qui casse s'il manque.
RUNTIME_FILES = {
    "docs/legal/conditions-utilisation.fr.md":
        "src/api/routes/legal.py sert /api/v1/terms ; sans lui, 503",
    "docs/legal/conditions-utilisation.en.md":
        "idem en anglais",
    "docs/legal/conditions-utilisation.es.md":
        "idem en espagnol",
    "docs/legal/politique-confidentialite.fr.md":
        "src/api/routes/legal.py sert /api/v1/privacy",
    "docs/legal/politique-confidentialite.en.md":
        "idem en anglais",
    "docs/legal/politique-confidentialite.es.md":
        "idem en espagnol",
    "config/pricing.json":
        "src/billing/pricing.py — la source unique des montants",
    "config/markets.json":
        "le registre des marchés",
    "scripts/stripe_preflight.py":
        "commande du runbook de mise en vente, lancée dans le Shell Render",
    "src/api/routes/legal.py":
        "le code lui-même",
}


@pytest.mark.parametrize("path,why", sorted(RUNTIME_FILES.items()))
def test_runtime_file_reaches_the_image(path, why):
    assert (REPO_ROOT / path).is_file(), f"{path} absent du dépôt — {why}"
    assert not is_excluded(path), (
        f".dockerignore exclut {path} de l'image, alors que {why}. "
        f"Ajoute une exception `!` APRÈS le motif qui l'exclut."
    )


def test_the_terms_the_chatbot_quotes_are_the_ones_shipped():
    """demo_agent.py cite les conditions dans la connaissance produit de M.I.A.

    Le chemin est recopié ici : s'il bouge sans que ce test bouge, la citation
    devient silencieusement vide en production.
    """
    from src.intelligence.chatbot.demo_agent import TERMS_PATH

    rel = TERMS_PATH.resolve().relative_to(REPO_ROOT).as_posix()
    assert rel in RUNTIME_FILES, f"{rel} lu à l'exécution mais pas couvert ici"
    assert not is_excluded(rel)


def test_the_legal_directory_the_route_reads_is_the_one_shipped():
    from src.api.routes import legal

    rel = legal._LEGAL_DIR.resolve().relative_to(REPO_ROOT).as_posix()
    assert rel == "docs/legal", f"_LEGAL_DIR a bougé vers {rel} — mettre ce test à jour"


# ---------------------------------------------------------------------------
# L'exception ne doit pas rouvrir les vannes
# ---------------------------------------------------------------------------

#: Ce que `docs/` doit CONTINUER d'exclure — l'exception vise `docs/legal/` seul.
STILL_EXCLUDED = [
    "docs/audits/security_M0_2026.md",
    "docs/architecture/MIA_MARKETS_ARCHITECTURE.md",
    "docs/governance/decision_gate_review_v2.md",
    "docs/audits/ds-1/coverage/conditions--terminal--desktop.png",
]


@pytest.mark.parametrize("path", STILL_EXCLUDED)
def test_the_rest_of_docs_stays_out_of_the_image(path):
    assert is_excluded(path), (
        f"{path} entre désormais dans l'image : l'exception `!docs/legal/` a été "
        f"élargie par erreur et alourdit le conteneur avec des captures et des audits."
    )


def test_secrets_and_bulk_data_are_still_excluded():
    """Garde de non-régression sur ce que .dockerignore protège réellement."""
    for path in (".env", "data/XAU_15MIN_2019_2026.csv", "webapp/node_modules/x.js",
                 "tests/test_billing.py", "models/whatever.pkl"):
        assert is_excluded(path), f"{path} ne doit pas entrer dans l'image"
