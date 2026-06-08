# Tests acceptance criteria — Vague 1

Critères d'acceptation pour valider chaque P0-strict-MVP. À transformer en tests automatisés.

## Index

| Fichier | Usage |
|---|---|
| `p0_acceptance_criteria.md` | Critères Given/When/Then par P0 (10 items) |
| `adversarial_chatbot_tests.md` | 30+ patterns adversariaux pour DG-112 |

## Stack tests

| Couche | Outil |
|---|---|
| Backend unit + integration | pytest + pytest-asyncio |
| Backend API | httpx + pytest |
| Frontend unit | Vitest + Testing Library |
| Frontend E2E | Playwright |
| Visual regression | Playwright snapshots |
| Adversarial chatbot | pytest paramétrisé |

## Coverage target

- Backend zones revenue (auth, quota_manager, chatbot, webhooks Stripe) : ≥ 85 %
- Backend autres : ≥ 70 %
- Frontend composants critiques (LectureView, ChatbotPanel, PricingPage) : ≥ 75 %
- Frontend autres : ≥ 50 %
- E2E parcours critiques (signup, signal_view, chatbot Q&A, upgrade, refund) : 100 %

## CI bloquante (DG-035)

Mode `warn` S3-S5 → mode `enforce` S6. Échec test = PR refusée.
