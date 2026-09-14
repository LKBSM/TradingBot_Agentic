"""SMTP-1 — Prove that transactional email actually works, step by step.

The Brevo setup fails in a small number of very specific ways, and they all look
identical from the outside ("no mail arrived"). This walks the chain one link at
a time and names the link that broke:

    1. configuration   which variables are set, and what sender they produce
    2. TCP + STARTTLS  is the relay reachable, is the port right
    3. AUTH            the classic: the v3 API key used instead of the SMTP key
    4. send            a real message to a real address

Run it after setting the variables, before trusting the setup:

    python scripts/check_email_delivery.py --to vous@exemple.com

From the Render Shell it reads the service's own environment, so it tests the
exact configuration production will use — not a copy of it.

    --dry-run   stop after AUTH, send nothing
    --json      machine-readable report

Exit code 0 only when every step passed. The password is never printed.

Full procedure: docs/ops/envoi-courriels.md
"""

from __future__ import annotations

import argparse
import json
import socket
import smtplib
import ssl
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.mailer import (  # noqa: E402
    ENV_HOST,
    ENV_PASSWORD,
    ENV_PORT,
    ENV_USER,
    email_verification_enforced,
    sender_address,
    smtp_configured,
)

OK, BAD, WARN = "  OK  ", " FAIL ", " WARN "


@dataclass
class Report:
    steps: list[dict] = field(default_factory=list)
    ok: bool = True

    def add(self, name: str, passed: bool, detail: str, fix: str = "") -> None:
        self.steps.append({"step": name, "passed": passed, "detail": detail, "fix": fix})
        if not passed:
            self.ok = False
        mark = OK if passed else BAD
        print(f"[{mark}] {name}: {detail}")
        if fix and not passed:
            for line in fix.splitlines():
                print(f"         → {line}")

    def note(self, name: str, detail: str) -> None:
        self.steps.append({"step": name, "passed": True, "detail": detail, "fix": ""})
        print(f"[{WARN}] {name}: {detail}")


def _env(name: str) -> str:
    import os

    return (os.environ.get(name) or "").strip()


def _check_sender_is_authenticated(report: "Report", domain: str, relay_host: str) -> None:
    """Warn when the From domain carries no DKIM record for this relay.

    Uses nslookup rather than a DNS library so the check costs no dependency —
    it is advisory, and a machine without nslookup simply skips it.
    """
    selectors = {
        "brevo": ("brevo1._domainkey", "brevo2._domainkey"),
        "sendgrid": ("s1._domainkey", "s2._domainkey"),
        "mailgun": ("mailo._domainkey", "k1._domainkey"),
    }
    provider = next((p for p in selectors if p in relay_host.lower()), None)
    if provider is None:
        return  # unknown relay — no selector to look for, stay silent

    import shutil
    import subprocess

    if not shutil.which("nslookup"):
        return
    found = []
    for sel in selectors[provider]:
        try:
            out = subprocess.run(
                ["nslookup", "-type=CNAME", f"{sel}.{domain}"],
                capture_output=True, text=True, timeout=10,
            ).stdout.lower()
        except Exception:
            return  # no network / no resolver — advisory check, stay silent
        if "canonical name" in out or provider in out:
            found.append(sel)
    if not found:
        report.add(
            "sender domain",
            False,
            f"{domain!r} carries no {provider} DKIM record — this domain is not "
            f"authenticated at the relay",
            "The relay will still ACCEPT the message; it just goes out\n"
            "unauthenticated and lands in spam. Check SMTP_FROM for a typo\n"
            "(one wrong letter can point at a domain you do not own), and\n"
            "that the domain is authenticated at the provider.",
        )
    else:
        report.add("sender domain", True, f"{domain} is DKIM-authenticated at {provider}")


def check(to_email: str, *, dry_run: bool = False) -> Report:
    import os

    report = Report()

    # ---- 1. Configuration ------------------------------------------------ #
    host = _env(ENV_HOST)
    if not smtp_configured():
        report.add(
            "configuration",
            False,
            f"{ENV_HOST} is not set — nothing can be sent",
            "Set SMTP_HOST / SMTP_USER / SMTP_PASSWORD / SMTP_FROM.\n"
            "docs/ops/envoi-courriels.md section 1.3",
        )
        return report

    user, password = _env(ENV_USER), _env(ENV_PASSWORD)
    port = int(_env(ENV_PORT) or 587)
    sender = sender_address()
    report.add(
        "configuration",
        True,
        f"host={host} port={port} user={user or '(none)'} "
        f"password={'set' if password else 'MISSING'} from={sender}",
    )
    if not password:
        report.add(
            "credentials",
            False,
            f"{ENV_PASSWORD} is empty — the relay will refuse the session",
            "Brevo: SMTP & API → SMTP tab → Generate a new SMTP key.",
        )
        return report

    # The wall and the mailer are a pair; say which state this deploy is in, so
    # the person running this knows what a failure actually costs right now.
    if email_verification_enforced():
        print(
            "         (the email-verification wall is UP: without working "
            "delivery, no new account can gain access)"
        )

    sender_domain = sender.rsplit("@", 1)[-1].lower()
    if sender_domain in {"gmail.com", "outlook.com", "hotmail.com", "yahoo.com"}:
        report.note(
            "sender domain",
            f"{sender!r} is on a public mailbox provider — relaying it will be "
            "rejected or spam-foldered. Use an address on your own authenticated "
            "domain.",
        )
    else:
        # Does the From domain actually carry the relay's DKIM records?
        #
        # Added after a real incident: SMTP_FROM read "no-reply@mia.market" —
        # the authenticated domain is "mia.markets". One missing letter, and a
        # domain belonging to someone else entirely. The relay accepted every
        # message without complaint, so nothing anywhere said a word; the mail
        # simply went out unauthenticated, DKIM unaligned, straight to spam.
        #
        # A relay accepting a sender proves nothing about that sender being
        # yours. Resolving its DKIM selector does.
        _check_sender_is_authenticated(report, sender_domain, host)

    # ---- 2. Reachability + STARTTLS -------------------------------------- #
    started = time.time()
    try:
        server = smtplib.SMTP(host, port, timeout=15)
    except socket.gaierror as exc:
        report.add(
            "connection",
            False,
            f"cannot resolve {host!r} ({exc})",
            "Check SMTP_HOST for a typo. Brevo's relay is smtp-relay.brevo.com.",
        )
        return report
    except (OSError, smtplib.SMTPException) as exc:
        report.add(
            "connection",
            False,
            f"cannot reach {host}:{port} ({type(exc).__name__}: {exc})",
            "Port 587 blocked by the network, or the wrong port.\n"
            "587 = STARTTLS (default) · 465 = implicit TLS, which also needs "
            "SMTP_STARTTLS=0 · 2525 = fallback.",
        )
        return report

    try:
        report.add(
            "connection",
            True,
            f"{host}:{port} answered in {(time.time() - started) * 1000:.0f} ms",
        )
        use_tls = (os.environ.get("SMTP_STARTTLS", "1").strip().lower()
                   in ("1", "true", "yes", "on"))
        if use_tls:
            try:
                server.starttls(context=ssl.create_default_context())
                report.add("STARTTLS", True, "channel encrypted")
            except smtplib.SMTPException as exc:
                report.add(
                    "STARTTLS",
                    False,
                    f"the relay refused STARTTLS ({exc})",
                    "On port 465 the TLS is implicit: set SMTP_STARTTLS=0.",
                )
                return report
        else:
            report.note("STARTTLS", "disabled by SMTP_STARTTLS=0 — implicit TLS assumed")

        # ---- 3. Authentication ------------------------------------------- #
        try:
            server.login(user, password)
            report.add("authentication", True, f"accepted for {user!r}")
        except smtplib.SMTPAuthenticationError as exc:
            report.add(
                "authentication",
                False,
                f"rejected ({exc.smtp_code} {exc.smtp_error!r})",
                "THE usual cause: the v3 API key was used as SMTP_PASSWORD.\n"
                "They are different credentials — you need the SMTP KEY\n"
                "(Brevo: SMTP & API → SMTP tab → Generate a new SMTP key).\n"
                "Also check SMTP_USER is the Brevo SMTP login, not the sender "
                "address.",
            )
            return report
        except smtplib.SMTPException as exc:
            report.add("authentication", False, f"{type(exc).__name__}: {exc}")
            return report

        if dry_run:
            print("\n--dry-run: stopping before sending. The credentials work.")
            return report

        # ---- 4. A real message ------------------------------------------- #
        from email.message import EmailMessage

        from src.api.email_branding import attach_branded_html

        stamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        body = (
            "Ceci est un test d'envoi M.I.A Markets.\n\n"
            f"Émis le {stamp} par scripts/check_email_delivery.py.\n"
            f"Relais : {host}:{port}\n"
            f"Expéditeur : {sender}\n\n"
            "Si vous lisez ceci, la chaîne complète fonctionne : connexion, "
            "chiffrement, authentification et remise.\n\n"
            "Vérifiez aussi que ce message n'est PAS arrivé dans les "
            "indésirables — un code de confirmation qui y atterrit enferme le "
            "client dehors exactement comme un courriel non envoyé."
        )
        msg = EmailMessage()
        msg["Subject"] = f"Test d'envoi M.I.A Markets — {stamp}"
        msg["From"] = sender
        msg["To"] = to_email
        msg.set_content(body)
        try:
            attach_branded_html(msg, body)
        except Exception:
            report.note("branded HTML", "skipped (APP_PUBLIC_URL not set?) — text still sent")

        try:
            refused = server.send_message(msg)
        except smtplib.SMTPSenderRefused as exc:
            report.add(
                "send",
                False,
                f"sender {sender!r} refused ({exc.smtp_code} {exc.smtp_error!r})",
                "The From address must be on a domain authenticated at the "
                "provider.\nBrevo: Senders, Domains & Dedicated IPs → Domains.",
            )
            return report
        except smtplib.SMTPRecipientsRefused as exc:
            report.add("send", False, f"recipient refused ({exc.recipients})")
            return report
        except smtplib.SMTPException as exc:
            report.add("send", False, f"{type(exc).__name__}: {exc}")
            return report

        if refused:
            report.add("send", False, f"the relay refused: {refused}")
            return report
        report.add("send", True, f"accepted for delivery to {to_email}")
    finally:
        try:
            server.quit()
        except Exception:
            pass

    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="check_email_delivery",
        description="Walk the transactional-email chain and name the link that breaks.",
        epilog="Full procedure: docs/ops/envoi-courriels.md",
    )
    parser.add_argument("--to", help="address to send the test message to")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="stop after authentication, send nothing",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable report")
    args = parser.parse_args(argv)

    if not args.to and not args.dry_run:
        parser.error("give --to <address>, or --dry-run to stop after authentication")

    report = check(args.to or "", dry_run=args.dry_run)

    if args.json:
        print(json.dumps({"ok": report.ok, "steps": report.steps}, indent=2))
    elif report.ok:
        if args.dry_run:
            print("\nCredentials verified. Re-run with --to <address> to send for real.")
        else:
            print(
                f"\nThe relay accepted the message for {args.to}.\n"
                "Two things left, and neither is visible from here:\n"
                "  1. open that mailbox — and check the spam folder, not just the inbox;\n"
                "  2. Brevo → Transactional → Logs: 'delivered', not 'deferred' or 'blocked'.\n"
                "Acceptance by the relay is not delivery to a human."
            )
    else:
        print("\nEmail delivery is NOT working. The failing step is marked FAIL above.")
    return 0 if report.ok else 1


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv

        load_dotenv(override=False)
    except ImportError:  # pragma: no cover
        pass
    sys.exit(main())
