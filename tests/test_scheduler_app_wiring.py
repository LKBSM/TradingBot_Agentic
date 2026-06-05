"""Tests for MarketReadingScheduler wiring into create_app (Chantier 3 Étape 5).

Verifies: (1) the scheduler is started on FastAPI startup; (2) the scheduler's
``stop`` is registered with the GracefulShutdownCoordinator and runs on
shutdown; (3) a scheduler that raises in ``start()`` does not abort app boot;
(4) the assembler endpoint still works regardless of scheduler wiring.
"""

from __future__ import annotations

from typing import List

from fastapi.testclient import TestClient

from src.api.app import create_app


class _StubScheduler:
    """Records start/stop calls; mimics the MarketReadingScheduler surface."""

    def __init__(self, raise_on_start: bool = False):
        self.start_calls = 0
        self.stop_calls = 0
        self._raise_on_start = raise_on_start
        self._running = False

    def start(self) -> None:
        self.start_calls += 1
        if self._raise_on_start:
            raise RuntimeError("boom")
        self._running = True

    def stop(self) -> None:
        self.stop_calls += 1
        self._running = False

    @property
    def running(self) -> bool:
        return self._running


def test_scheduler_start_invoked_on_app_startup(tmp_path):
    sched = _StubScheduler()
    app = create_app(market_reading_scheduler=sched)
    with TestClient(app):
        # Lifespan ran on enter.
        assert sched.start_calls == 1
    # Lifespan ran on exit → stop invoked via coordinator.
    assert sched.stop_calls == 1


def test_scheduler_stop_registered_with_coordinator(tmp_path):
    sched = _StubScheduler()
    app = create_app(market_reading_scheduler=sched)
    # Trigger the lifespan by entering/exiting the TestClient.
    with TestClient(app):
        coord = app.state.app_state.shutdown_coordinator
        registered_names = {r.name for r in coord._registrations}  # noqa: SLF001
        assert "market-reading-scheduler" in registered_names


def test_scheduler_start_failure_does_not_abort_app_boot(tmp_path):
    sched = _StubScheduler(raise_on_start=True)
    app = create_app(market_reading_scheduler=sched)
    # The exception must be swallowed; the app must still serve requests.
    with TestClient(app) as client:
        # /health endpoint always 200-ish even when subsystems are None
        resp = client.get("/health")
        # We don't assert a precise status — only that the app booted.
        assert resp.status_code in (200, 503)
        assert sched.start_calls == 1


def test_no_scheduler_wired_does_not_break_anything(tmp_path):
    """create_app must work with market_reading_scheduler=None (Chantier 2 default)."""
    app = create_app()  # no scheduler at all
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code in (200, 503)
    assert app.state.app_state.market_reading_scheduler is None
