"""Branded HTML alternative for transactional emails (BRD-2).

The three transactional mails (email verification, password reset, annual
renewal notice) are plain-text first — that part is never removed, so a
text-only client still receives the exact same message. This module adds an
``text/html`` *alternative* carrying the M.I.A Markets logo at the top.

The logo is a hosted PNG (email clients block SVG): ``{APP_PUBLIC_URL}/brand/
email-logo.png``, served by the frontend route handler. It always ships with an
``alt`` text so the mail is legible even when images are blocked.

Purely presentational: no delivery logic changes here — the caller still builds
and sends the same ``EmailMessage``; we only attach a second representation.
"""

from __future__ import annotations

import html
import re
from email.message import EmailMessage

from .public_urls import app_public_url

_URL_RE = re.compile(r"(https?://[^\s]+)")


def email_logo_url() -> str:
    """Absolute, stable URL of the hosted email logo PNG."""
    return f"{app_public_url()}/brand/email-logo.png"


def _text_to_html_body(text_body: str) -> str:
    """Escape a plain-text body and turn bare URLs into links, keeping breaks."""
    parts: list[str] = []
    for para in text_body.split("\n\n"):
        lines = [
            _URL_RE.sub(
                lambda m: f'<a href="{html.escape(m.group(1))}">{html.escape(m.group(1))}</a>',
                html.escape(line),
            )
            for line in para.split("\n")
        ]
        parts.append(
            '<p style="margin:0 0 16px;font-size:15px;line-height:1.5;color:#0F1729">'
            + "<br>".join(lines)
            + "</p>"
        )
    return "\n".join(parts)


def branded_html(text_body: str) -> str:
    """Wrap a plain-text body into a minimal branded HTML email."""
    logo = html.escape(email_logo_url())
    body_html = _text_to_html_body(text_body)
    return (
        '<!DOCTYPE html><html lang="fr"><body style="margin:0;padding:0;'
        'background:#f4f6fb">'
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
        'style="background:#f4f6fb"><tr><td align="center" style="padding:32px 16px">'
        '<table role="presentation" width="480" cellpadding="0" cellspacing="0" '
        'style="max-width:480px;background:#ffffff;border-radius:12px;'
        'border:1px solid #e5e9f2">'
        '<tr><td style="padding:28px 32px 8px">'
        f'<img src="{logo}" alt="M.I.A Markets" width="200" height="50" '
        'style="display:block;border:0;outline:none;text-decoration:none;height:auto">'
        '</td></tr>'
        f'<tr><td style="padding:8px 32px 28px">{body_html}</td></tr>'
        '</table></td></tr></table></body></html>'
    )


def attach_branded_html(msg: EmailMessage, text_body: str) -> None:
    """Add the branded HTML alternative to a message whose text is already set."""
    msg.add_alternative(branded_html(text_body), subtype="html")
