"""Production ASGI entry point for the MIA Markets API.

Run with::

    uvicorn src.api.asgi:app --host 0.0.0.0 --port 8000

Unlike ``src.intelligence.main`` (which also spins up the legacy Sentinel
scanner thread + MT5 data source), this entry point boots ONLY the FastAPI
application. The MarketReading engine, hybrid scheduler and the niveau-1.5
chatbot are wired by ``create_app``'s lifespan from the environment
(``BOOTSTRAP_ENABLED`` / ``SCHEDULER_ENABLED`` / ``CHATBOT_ENABLED`` /
``NEWS_PIPELINE_ENABLED``), so this is the lean way to serve the V2 product
(webapp + chatbot) that talks to Twelve Data + Anthropic.

``.env`` is loaded before the app is built so those env-gated flags and the
API keys are visible to the lifespan bootstrap.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

# Load .env into the process environment BEFORE create_app() so the lifespan
# bootstrap (which reads os.environ) sees TWELVE_DATA_API_KEY, ANTHROPIC_API_KEY
# and the *_ENABLED flags. override=False keeps any var already exported in the
# shell authoritative over the file.
load_dotenv(override=False)

from src.api.app import create_app  # noqa: E402 — must follow load_dotenv

# Module-level ASGI app uvicorn can import directly. No subsystem injection:
# everything the V2 product needs is built by the lifespan from env.
app = create_app()


__all__ = ["app"]
