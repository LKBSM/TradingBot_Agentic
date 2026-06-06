"""MIA Markets V2 — Chantier 4 — Chatbot niveau 1.5 strict.

3-layer defence architecture (doc §4.1):
  - Couche 1 : adversarial input filter (``adversarial_filter``) — blocks risky
    questions BEFORE any LLM call.
  - Couche 2 : Haiku orchestrator with tool use (``chatbot``) + strict system prompt.
  - Couche 3 : forbidden-tokens output filter (``output_filter``) — scrubs the
    LLM response before it ever reaches the user.

Shared vocabulary (forbidden tokens + adversarial regex) + text normalisation
live in ``constants``. The chatbot describes market conditions; it never
recommends an action (niveau 1.5 strict, doc §1.2).
"""
