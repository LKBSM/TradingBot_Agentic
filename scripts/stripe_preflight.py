#!/usr/bin/env python3
"""Vérifie qu'une configuration Stripe encaisserait VRAIMENT de l'argent réel.

Pourquoi ce script existe
-------------------------
Passer en mode live ne se constate pas en lisant le code : tout se joue dans le
tableau de bord Stripe et dans les variables d'environnement. Quatre erreurs
sont possibles, toutes SILENCIEUSES jusqu'au premier client :

1. la clé est encore une clé de TEST → le client « paie », aucun argent ne bouge ;
2. les identifiants de prix pointent sur des prix de test, ou sur le mauvais
   montant → l'ouverture de session échoue, ou on facture un autre tarif ;
3. le webhook vise ``/api/v1/billing/webhook`` (le point d'entrée RETIRÉ) au lieu
   de ``/api/billing/webhook`` → l'argent entre, l'accès reste fermé ;
4. le webhook n'est pas abonné à tous les événements dont le mur d'accès a
   besoin → certains paiements n'ouvrent jamais rien ;
5. ``config/pricing.json`` lui-même annonce un montant qui n'a pas été décidé →
   Stripe et le site sont d'accord, sur le mauvais prix.

Ce script lit Stripe et l'environnement, et rend un verdict par ligne. Il
n'écrit RIEN, ni chez Stripe ni en base. Il n'affiche jamais une clé.

Les montants attendus viennent de ``config/pricing.json`` — la source unique du
projet. Les montants de la mise en vente (39,99 USD par mois, 359,88 USD par an)
sont en plus écrits dans ``GO_LIVE_CENTS`` comme SECOND TÉMOIN : sans cela, un
fichier de configuration erroné et un Stripe erroné se valideraient l'un l'autre.

Usage
-----
    # sur Render : onglet « Shell » du service mia-backend
    python scripts/stripe_preflight.py

    # en local, en visant explicitement le mode live
    STRIPE_SECRET_KEY=sk_live_... STRIPE_PRICE_MONTHLY=price_... \
    STRIPE_PRICE_ANNUAL=price_... API_PUBLIC_URL=https://api.exemple.com \
    python scripts/stripe_preflight.py

    # accepter une clé de test sans que ce soit compté comme un échec
    python scripts/stripe_preflight.py --allow-test-mode

Sortie : 0 si tout passe, 1 si au moins une vérification échoue.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.billing.pricing import list_paid_plans  # noqa: E402
from src.billing.stripe_client import ACCOUNT_SUBSCRIPTION_EVENTS  # noqa: E402

#: Les montants décidés pour la mise en vente, EN CENTS, par clé de cadence.
#:
#: Pourquoi une constante ici alors que la source unique est config/pricing.json :
#: comparer Stripe à config/pricing.json ne prouve que leur ACCORD. Si le fichier
#: de configuration était faux, les deux seraient faux ensemble et le script
#: dirait « tout passe ». Ces valeurs sont donc un SECOND TÉMOIN indépendant — la
#: décision commerciale du 2026-09-27, écrite une fois de plus, exprès. Elles ne
#: servent jamais à calculer un prix : uniquement à vérifier la configuration.
#:   MONTHLY  39,99 USD par mois
#:   ANNUAL  359,88 USD par an, soit 29,99 USD par mois (facturés en une fois)
GO_LIVE_CENTS = {"MONTHLY": 3999, "ANNUAL": 35988}

#: Le SEUL chemin que le mur d'accès alimente (src/api/routes/account_billing.py).
CORRECT_WEBHOOK_PATH = "/api/billing/webhook"
#: Le point d'entrée retiré — s'il est configuré chez Stripe, c'est une panne.
RETIRED_WEBHOOK_PATH = "/api/v1/billing/webhook"

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"

_results: List[Tuple[str, str]] = []


def record(level: str, message: str) -> None:
    _results.append((level, message))
    mark = {PASS: "  OK  ", WARN: " WARN ", FAIL: " FAIL "}[level]
    print(f"[{mark}] {message}")


def cadence_to_interval(cadence: str) -> Optional[str]:
    return {"monthly": "month", "annual": "year"}.get(cadence)


def check_env(allow_test: bool) -> Optional[str]:
    """Vérifie les variables et retourne la clé secrète (jamais affichée)."""
    key = os.environ.get("STRIPE_SECRET_KEY", "").strip()
    if not key:
        record(FAIL, "STRIPE_SECRET_KEY absente — rien ne peut être encaissé.")
        return None

    if key.startswith("sk_live_"):
        record(PASS, "STRIPE_SECRET_KEY est une clé LIVE (de l'argent réel peut circuler).")
    elif key.startswith("sk_test_"):
        level = WARN if allow_test else FAIL
        record(level, "STRIPE_SECRET_KEY est une clé de TEST — aucun argent réel ne circulera.")
    else:
        record(WARN, "STRIPE_SECRET_KEY a un préfixe inattendu (ni sk_live_ ni sk_test_).")

    secret = os.environ.get("STRIPE_WEBHOOK_SECRET", "").strip()
    if not secret:
        record(FAIL, "STRIPE_WEBHOOK_SECRET absente — toute notification Stripe sera rejetée en 400.")
    elif not secret.startswith("whsec_"):
        record(WARN, "STRIPE_WEBHOOK_SECRET ne commence pas par whsec_ — vérifie que c'est le bon secret.")
    else:
        record(PASS, "STRIPE_WEBHOOK_SECRET présente et bien formée.")

    if not os.environ.get("API_PUBLIC_URL", "").strip():
        record(WARN, "API_PUBLIC_URL absente — je ne peux pas vérifier que le webhook vise CE backend.")
    return key


def check_catalog() -> None:
    """config/pricing.json annonce-t-il bien les montants décidés, au cent près ?

    Se lance avant tout appel à Stripe : si la source unique est fausse, tout le
    reste vérifierait la cohérence d'une erreur.
    """
    for plan in list_paid_plans():
        expected = GO_LIVE_CENTS.get(plan.key)
        if expected is None:
            record(WARN, f"Cadence {plan.key} inconnue du préflight — montant non vérifié.")
            continue
        actual = round(plan.amount_usd * 100)
        if actual != expected:
            record(
                FAIL,
                f"config/pricing.json annonce {actual} cents pour {plan.key}, "
                f"alors que la décision de mise en vente est {expected} cents "
                f"({expected / 100:.2f} {plan.currency}). Le site afficherait un "
                f"prix qui n'a pas été décidé.",
            )
        else:
            record(
                PASS,
                f"config/pricing.json annonce bien {expected / 100:.2f} "
                f"{plan.currency} pour {plan.key}.",
            )

        if plan.cadence == "annual":
            # 35988 / 12 = 2999 : le « soit N par mois » affiché doit être exact,
            # pas un arrondi présenté comme exact.
            if actual % 12 != 0:
                record(
                    FAIL,
                    f"{actual} cents ne se divise pas par 12 — l'équivalent mensuel "
                    f"affiché serait un arrondi, présenté comme exact.",
                )
            else:
                record(
                    PASS,
                    f"Équivalent mensuel exact : {actual // 12 / 100:.2f} "
                    f"{plan.currency} par mois.",
                )


def check_prices(stripe: Any, key_is_live: bool) -> None:
    """Chaque prix Stripe doit correspondre à config/pricing.json, au cent près."""
    for plan in list_paid_plans():
        # Même nom que celui déclaré dans config/pricing.json (`stripeEnvVar`)
        # et lu par src/billing/pricing.py : STRIPE_PRICE_MONTHLY / _ANNUAL.
        env_name = f"STRIPE_PRICE_{plan.key}"
        price_id = os.environ.get(env_name, "").strip()
        label = f"{plan.key} ({plan.amount_usd:.2f} {plan.currency})"

        if not price_id:
            record(FAIL, f"{env_name} absente — le plan {label} ne peut pas être vendu.")
            continue

        try:
            price = stripe.Price.retrieve(price_id)
        except Exception as exc:  # noqa: BLE001 — on veut le message brut de Stripe
            record(FAIL, f"{env_name} = {price_id} introuvable chez Stripe : {exc}")
            continue

        if price.get("livemode") is not key_is_live:
            record(
                FAIL,
                f"{env_name} est un prix de "
                f"{'TEST' if not price.get('livemode') else 'LIVE'} alors que la clé est "
                f"{'LIVE' if key_is_live else 'TEST'} — Stripe refusera la session.",
            )
        if not price.get("active", False):
            record(FAIL, f"{env_name} est un prix ARCHIVÉ chez Stripe — il ne peut plus être vendu.")

        expected_cents = round(plan.amount_usd * 100)
        actual_cents = price.get("unit_amount")
        if actual_cents != expected_cents:
            record(
                FAIL,
                f"{env_name} facture {actual_cents} cents, mais config/pricing.json "
                f"annonce {expected_cents} cents ({label}). Le site mentirait sur son prix.",
            )
        else:
            record(PASS, f"{env_name} facture bien {expected_cents / 100:.2f} {plan.currency} — {label}.")

        actual_currency = (price.get("currency") or "").upper()
        if actual_currency != plan.currency.upper():
            record(FAIL, f"{env_name} est en {actual_currency}, attendu {plan.currency}.")

        wanted_interval = cadence_to_interval(plan.cadence)
        recurring = price.get("recurring") or {}
        if wanted_interval and recurring.get("interval") != wanted_interval:
            record(
                FAIL,
                f"{env_name} se renouvelle « {recurring.get('interval')} », "
                f"attendu « {wanted_interval} ».",
            )


def check_webhooks(stripe: Any) -> None:
    """Le webhook doit viser le bon chemin ET couvrir tous les événements utiles."""
    try:
        endpoints = list(stripe.WebhookEndpoint.list(limit=100).auto_paging_iter())
    except Exception as exc:  # noqa: BLE001
        record(WARN, f"Impossible de lister les webhooks (permission de la clé ?) : {exc}")
        return

    if not endpoints:
        record(FAIL, "Aucun webhook configuré chez Stripe — aucun paiement n'ouvrira jamais l'accès.")
        return

    api_base = os.environ.get("API_PUBLIC_URL", "").strip().rstrip("/")
    good = [e for e in endpoints if str(e.get("url", "")).endswith(CORRECT_WEBHOOK_PATH)]
    retired = [e for e in endpoints if str(e.get("url", "")).endswith(RETIRED_WEBHOOK_PATH)]

    for e in retired:
        record(
            FAIL,
            f"Un webhook vise le point d'entrée RETIRÉ : {e.get('url')} — il répond 410 et "
            f"n'ouvre aucun accès. Repointe-le sur {CORRECT_WEBHOOK_PATH}.",
        )

    if not good:
        record(
            FAIL,
            f"Aucun webhook ne vise {CORRECT_WEBHOOK_PATH} — c'est le SEUL chemin que le mur "
            f"d'accès lit. Un paiement réel n'ouvrirait rien.",
        )
        return

    for e in good:
        url = str(e.get("url", ""))
        if e.get("status") != "enabled":
            record(FAIL, f"Le webhook {url} est désactivé chez Stripe (status={e.get('status')}).")
        else:
            record(PASS, f"Webhook actif sur le bon chemin : {url}")

        if api_base and not url.startswith(api_base):
            record(
                WARN,
                f"Le webhook {url} ne vise pas API_PUBLIC_URL ({api_base}) — "
                f"vérifie qu'il pointe bien sur CE backend et pas sur un ancien déploiement.",
            )

        subscribed = set(e.get("enabled_events") or [])
        if "*" in subscribed:
            record(PASS, f"{url} est abonné à tous les événements.")
            continue
        missing = sorted(ACCOUNT_SUBSCRIPTION_EVENTS - subscribed)
        if missing:
            record(
                FAIL,
                f"{url} n'est pas abonné à : {', '.join(missing)} — "
                f"ces paiements/annulations ne seraient jamais appliqués.",
            )
        else:
            record(PASS, f"{url} couvre les {len(ACCOUNT_SUBSCRIPTION_EVENTS)} événements nécessaires.")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Préflight Stripe avant d'encaisser de l'argent réel.")
    parser.add_argument(
        "--allow-test-mode",
        action="store_true",
        help="Ne pas compter une clé sk_test_ comme un échec (répétition avant la bascule live).",
    )
    args = parser.parse_args(argv)

    print("=" * 74)
    print("  PRÉFLIGHT STRIPE — est-ce que ce déploiement encaisserait vraiment ?")
    print("=" * 74)

    key = check_env(args.allow_test_mode)
    if not key:
        print("\nArrêt : sans clé, rien d'autre n'est vérifiable.")
        return 1

    try:
        import stripe  # type: ignore
    except ImportError:
        record(FAIL, "Le paquet `stripe` n'est pas installé dans cet environnement.")
        return 1

    stripe.api_key = key
    key_is_live = key.startswith("sk_live_")

    print("-" * 74)
    check_catalog()
    print("-" * 74)
    check_prices(stripe, key_is_live)
    print("-" * 74)
    check_webhooks(stripe)
    print("=" * 74)

    fails = sum(1 for lvl, _ in _results if lvl == FAIL)
    warns = sum(1 for lvl, _ in _results if lvl == WARN)
    if fails:
        print(f"  VERDICT : {fails} BLOCAGE(S), {warns} avertissement(s).")
        print("  Ne mets pas le produit en vente tant qu'il reste un FAIL.")
        return 1
    print(f"  VERDICT : tout passe ({warns} avertissement(s)).")
    print("  Un paiement réel devrait ouvrir l'accès. Prouve-le avec un vrai achat.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
