"""Le préflight doit refuser un portail client qui contredit la clause 8 (LEG-2).

Pourquoi ce fichier existe
--------------------------
La clause 8.1 des conditions dit, depuis LEG-2, que la résiliation se fait « par
le bouton …, qui ouvre le portail de facturation hébergé par Stripe », et qu'elle
« prend effet à la fin de la période déjà payée ». Ces deux phrases ne dépendent
d'aucune ligne de code : elles dépendent d'une case à cocher dans le tableau de
bord Stripe. Décocher « Cancel subscriptions » ne casse aucun test, ne lève aucune
erreur, et rend le contrat faux — le client qui veut partir n'a simplement plus de
bouton.

``check_portal`` est ce qui rend cette dérive visible, et ces tests sont ce qui
garantit qu'elle reste visible.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, List

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "stripe_preflight.py"

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"

#: L'URL réellement codée dans le webapp — le préflight la relit sur le disque.
HARDCODED_URL = "https://billing.stripe.com/p/login/5kQeVd3D45gl7sW55G67S00"


@pytest.fixture()
def preflight():
    """Le script, chargé comme un module, avec son journal vidé."""
    spec = importlib.util.spec_from_file_location("stripe_preflight_portal_ut", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._results.clear()
    return module


class _Page:
    """Ce que renvoie ``Configuration.list`` : un objet qui se pagine."""

    def __init__(self, items: List[Dict[str, Any]]) -> None:
        self._items = items

    def auto_paging_iter(self):
        return iter(self._items)


class _FakeStripe:
    """Le strict minimum de la surface Stripe que ``check_portal`` touche."""

    def __init__(self, items: Any) -> None:
        outer = self

        class _Configuration:
            @staticmethod
            def list(**_kwargs):
                if isinstance(outer._items, Exception):
                    raise outer._items
                return _Page(outer._items)

        class _BillingPortal:
            Configuration = _Configuration

        self._items = items
        self.billing_portal = _BillingPortal


def _config(**overrides: Any) -> Dict[str, Any]:
    """Une configuration de portail conforme à la clause 8, à amender par test."""
    base: Dict[str, Any] = {
        "id": "bpc_1",
        "is_default": True,
        "active": True,
        "features": {
            "subscription_cancel": {"enabled": True, "mode": "at_period_end"},
            "payment_method_update": {"enabled": True},
            "invoice_history": {"enabled": True},
        },
        "login_page": {"enabled": True, "url": HARDCODED_URL},
    }
    base.update(overrides)
    return base


def levels(preflight) -> List[str]:
    return [level for level, _ in preflight._results]


def messages(preflight) -> str:
    return "\n".join(msg for _, msg in preflight._results)


# ─── Le cas conforme ──────────────────────────────────────────────────────


def test_a_conforming_portal_passes_without_a_single_warning(preflight):
    preflight.check_portal(_FakeStripe([_config()]))
    assert FAIL not in levels(preflight)
    assert WARN not in levels(preflight)
    assert PASS in levels(preflight)


# ─── Ce que la clause 8 rend obligatoire ──────────────────────────────────


def test_cancellation_disabled_is_a_blocker(preflight):
    """Sans annulation dans le portail, la clause 8.1 est une fausse déclaration."""
    portal = _config()
    portal["features"]["subscription_cancel"] = {"enabled": False}
    preflight.check_portal(_FakeStripe([portal]))
    assert FAIL in levels(preflight)
    assert "Cancel subscriptions" in messages(preflight)


def test_immediate_cancellation_is_a_blocker(preflight):
    """La clause promet l'accès jusqu'à la fin de la période payée."""
    portal = _config()
    portal["features"]["subscription_cancel"] = {"enabled": True, "mode": "immediately"}
    preflight.check_portal(_FakeStripe([portal]))
    assert FAIL in levels(preflight)
    assert "IMMÉDIATEMENT" in messages(preflight)


def test_no_configuration_at_all_is_a_blocker(preflight):
    preflight.check_portal(_FakeStripe([]))
    assert FAIL in levels(preflight)


def test_an_unknown_cancellation_mode_only_warns(preflight):
    """Un mode inconnu n'est pas une preuve de défaut : il demande un œil humain."""
    portal = _config()
    portal["features"]["subscription_cancel"] = {"enabled": True, "mode": "surprise"}
    preflight.check_portal(_FakeStripe([portal]))
    assert FAIL not in levels(preflight)
    assert WARN in levels(preflight)


# ─── Le lien de secours de /compte ────────────────────────────────────────


def test_a_login_url_from_another_account_is_a_blocker(preflight):
    """Le cas réel : une URL du compte de TEST laissée dans le code."""
    portal = _config()
    portal["login_page"] = {
        "enabled": True,
        "url": "https://billing.stripe.com/p/login/test_aaaaaaaaaa",
    }
    preflight.check_portal(_FakeStripe([portal]))
    assert FAIL in levels(preflight)
    assert "webapp/lib/billing/portal.ts" in messages(preflight)


def test_a_disabled_login_page_is_a_blocker_while_the_code_links_to_it(preflight):
    portal = _config()
    portal["login_page"] = {"enabled": False}
    preflight.check_portal(_FakeStripe([portal]))
    assert FAIL in levels(preflight)


def test_the_hardcoded_url_is_read_from_the_webapp(preflight):
    """La comparaison porte sur le fichier réel, pas sur une copie du test."""
    assert preflight._hardcoded_login_url() == HARDCODED_URL


# ─── Promesses d'interface, pas de contrat ────────────────────────────────


def test_missing_invoice_history_only_warns(preflight):
    """/compte annonce les factures ; ce n'est pas le contrat, donc pas un blocage."""
    portal = _config()
    portal["features"]["invoice_history"] = {"enabled": False}
    preflight.check_portal(_FakeStripe([portal]))
    assert FAIL not in levels(preflight)
    assert WARN in levels(preflight)


# ─── Robustesse ──────────────────────────────────────────────────────────


def test_a_stripe_failure_warns_and_never_raises(preflight):
    """Un préflight qui plante ne dit rien du portail — il doit le dire, pas crasher."""
    preflight.check_portal(_FakeStripe(RuntimeError("clé sans permission")))
    assert levels(preflight) == [WARN]


def test_it_inspects_the_DEFAULT_configuration(preflight):
    """create_billing_portal_session ne passe aucune configuration : c'est la
    configuration par défaut du compte qui s'applique, donc la seule qui compte."""
    broken_secondary = _config(id="bpc_2", is_default=False)
    broken_secondary["features"]["subscription_cancel"] = {"enabled": False}
    preflight.check_portal(_FakeStripe([broken_secondary, _config()]))
    # La défaillante n'est pas celle par défaut : aucun blocage.
    assert FAIL not in levels(preflight)


def test_without_any_default_it_warns_and_still_inspects(preflight):
    portal = _config(is_default=False)
    preflight.check_portal(_FakeStripe([portal]))
    assert WARN in levels(preflight)
    assert "par défaut" in messages(preflight)


def test_the_portal_check_runs_in_the_documented_command(preflight):
    """check_portal doit être appelée par main(), sinon elle ne sert à rien."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert "check_portal(stripe)" in source[source.index("def main("):]
