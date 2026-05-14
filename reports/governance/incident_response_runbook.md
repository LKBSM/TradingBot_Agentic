# Smart Sentinel AI — Incident Response Runbook

> Sprint **RISK-2B.4** — Sofia (Governance) — Phase 2B
>
> One-page operational runbook for the four highest-probability
> incident classes during Phase 2B. Each section: **detect**, **contain**,
> **communicate**, **post-mortem**. Print, pin above the workstation,
> dry-run quarterly.

---

## 1. RAG hallucination detected in production narrative

**Detect**
- Compliance checker (`ComplianceChecker`) flags a published narrative
  with violations.
- Manual review of `worst_5` queue in `/api/v1/metrics/narrative-quality`
  shows a faithfulness < 0.5 narrative was delivered.
- User report via support@.

**Contain** (target: < 15 min)
1. Freeze new narrative generation:
   `POST /admin/freeze-llm` (or env var `NARRATIVE_FROZEN=1`).
2. Hide the offending insight from the webapp:
   pull `insight_id` from `/api/v1/insights/{id}`, mark
   `displayed=false` in the signal store.
3. Push corrected narrative or generic placeholder via
   `template_narrative_engine.render_fallback()`.

**Communicate** (within 1h)
- Telegram pinned message in the public channel:
  "Une analyse publiée ce matin contenait une formulation imprécise.
  Nous l'avons retirée et publions une version corrigée à 14:00.
  Le journal de transparence reflète ce correctif."
- Audit-log entry via `admin_action_log` with `action="narrative_correction"`.

**Post-mortem** (within 48h)
- Add the offending prompt + retrieved chunks to the eval fixtures
  (LLM-2B.3) so the regression gate catches recurrence.
- If pattern-class (not one-off), bump prompt version in
  `PromptRegistry` and re-deploy.

---

## 2. Stripe webhook outage / payment failures spiking

**Detect**
- `/health/deep` shows `stripe` subsystem unhealthy (when wired).
- Stripe dashboard shows `event_delivery_attempts.failed` > 10%.
- User reports "card charged but tier not upgraded".

**Contain** (target: < 30 min)
1. Switch ingestion to webhook backup endpoint (`/webhooks/stripe-backup`).
2. Drain the last 24h of Stripe events via
   `python scripts/stripe_replay_events.py --since=24h`.
3. Confirm `tier_manager.get_user_by_api_key()` returns the expected
   tier for affected users; if not, manually re-issue via
   `/admin/tier-override`.

**Communicate**
- In-app banner for affected users: "Mise à jour de votre abonnement
  en cours. Si votre accès n'est pas restauré sous 1h, contactez-nous."
- Slack/email to ops with the count of affected users.

**Post-mortem**
- Confirm webhook replay covered every missed event.
- Reconcile Stripe `invoice.paid` vs internal subscription records.

---

## 3. Data leak / unauthorized data access

**Detect**
- Sentry alert on unusual read pattern (admin_action_log shows
  unexpected actor reading a high volume of insights).
- API key rotation flow not used after a key was committed to git.
- External report (HackerOne, abuse@).

**Contain** (target: < 1h)
1. **Revoke the suspected key immediately** via `key_store.revoke_key`.
2. If the leak vector is a process-wide secret (ANTHROPIC_API_KEY,
   webhook signing secret), rotate it everywhere:
   - `python scripts/rotate_webhook_secret.py`
   - Anthropic console → revoke + new key → Railway env update.
3. Audit-log query for the suspect window:
   `GET /admin/audit-log?since=...&actor=key:N`.
4. If user data was exposed, snapshot the affected rows for
   forensics before any cleanup.

**Communicate** (legal-driven; CNIL within 72h if PII involved)
- Drafted incident notice (template in `reports/governance/data_breach_template.md`).
- Notify affected users by email within 72h (CNIL/GDPR Article 33).
- Public post-mortem within 7 days for B2B impact.

**Post-mortem**
- Root-cause: enumerate every place the leaked secret was used.
- Add a CI scanner (e.g., `truffleHog`) to prevent re-occurrence.

---

## 4. AMF inquiry / regulator contact

**Detect**
- Email or registered letter from AMF / ACPR / CNIL.
- Press inquiry asking about "trading signals" framing.
- User claim that a recommendation caused financial harm.

**Contain** (target: < 24h — DO NOT RESPOND TO REGULATOR DIRECTLY)
1. **Do not engage substantively without counsel.** Acknowledge
   receipt within 24h ("Nous accusons réception de votre demande,
   notre conseil revient vers vous sous X jours ouvrables").
2. Email retained fintech lawyer (Théo's INFRA-2B.4 contact)
   immediately.
3. Freeze any marketing content (Karim) that mentions edge,
   performance, or recommendations.
4. Snapshot `audit_ledger`, `admin_action_log`, `transparency_log`,
   `narrative_quality` summaries for the inquiry window via:
   `python scripts/audit_ledger_snapshot.py snapshot data/audit_ledger.db reports/inquiry_<date>.jsonl`.

**Communicate** (lawyer-driven)
- Internal: no public statement, no social media posts about the
  inquiry until lawyer green-lights.
- External: continue normal operations, no specific "we're under
  inquiry" messaging.

**Post-mortem**
- If positioning needs to change (e.g., MiFID-art-21 editorial framing),
  schedule a Phase rebrand with Inès + Karim.
- Document any required changes in `reports/governance/regulatory_response_<date>.md`.

---

## Dry-run schedule

Quarterly tabletop exercise (Sofia leads):
- **Q1**: scenario 1 (RAG hallucination)
- **Q2**: scenario 2 (Stripe outage)
- **Q3**: scenario 3 (data leak)
- **Q4**: scenario 4 (AMF inquiry)

Each dry-run: 90 minutes, all agents on call, mock execution of every
step. Output: revised runbook entries based on what broke.

## Contacts

- **Fintech lawyer**: TBD (set after INFRA-2B.4)
- **CNIL primary contact**: cil@cnil.fr (data leak)
- **AMF press**: communication@amf-france.org
- **Anthropic security**: security@anthropic.com (LLM-side incidents)
- **Railway support**: help@railway.app (infra outage)
