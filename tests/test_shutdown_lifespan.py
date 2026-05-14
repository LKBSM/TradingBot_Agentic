"""Integration tests for INFRA-2B.11 lifespan-wired shutdown."""

from __future__ import annotations

import os

os.environ.setdefault("SENTINEL_TESTING_MODE", "1")

import pytest
from fastapi.testclient import TestClient

from src.api.app import _auto_register_default_handlers, create_app
from src.api.dependencies import AppState
from src.api.shutdown import GracefulShutdownCoordinator


class _CloseableStub:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_lifespan_runs_registered_handlers_on_shutdown():
    """Closing the TestClient context-manager triggers lifespan shutdown,
    which should run every registered handler."""
    ran = {"flag": False}

    coord = GracefulShutdownCoordinator()
    coord.register("custom", lambda: ran.update(flag=True))

    with TestClient(create_app(shutdown_coordinator=coord)) as client:
        client.get("/api/v1/health")

    # Lifespan shutdown fired on exit of the `with`.
    assert ran["flag"] is True
    assert coord.report is not None
    assert coord.is_shutting_down is True


def test_auto_register_picks_up_default_handlers():
    """SignalStore (always wired), audit_ledger, admin_action_log,
    scanner — each should be auto-registered when present."""
    coord = GracefulShutdownCoordinator()
    state = AppState(
        signal_store=_CloseableStub(),
        audit_ledger=_CloseableStub(),
        admin_action_log=_CloseableStub(),
        scanner=type("S", (), {"stop": lambda self: None})(),
    )
    _auto_register_default_handlers(coord, state)
    names = [r.name for r in coord._registrations]  # noqa: SLF001
    # scanner first, then ledgers, then store
    assert names == [
        "sentinel-scanner",
        "audit-ledger",
        "admin-action-log",
        "signal-store",
    ]


def test_auto_register_skips_already_registered_names():
    """Caller can override budgets by pre-registering."""
    coord = GracefulShutdownCoordinator()
    coord.register("audit-ledger", lambda: None, budget_s=999)
    state = AppState(
        signal_store=_CloseableStub(),
        audit_ledger=_CloseableStub(),
    )
    _auto_register_default_handlers(coord, state)
    audit_regs = [
        r for r in coord._registrations if r.name == "audit-ledger"  # noqa: SLF001
    ]
    assert len(audit_regs) == 1
    assert audit_regs[0].budget_s == 999  # caller's wins


def test_auto_register_skips_subsystems_without_close():
    """A scanner without .stop() — or a store without .close() —
    just isn't registered. No crash."""
    coord = GracefulShutdownCoordinator()
    state = AppState(
        signal_store=_CloseableStub(),  # has close()
        audit_ledger=object(),           # no close — must be skipped
        scanner=object(),                # no stop — must be skipped
    )
    _auto_register_default_handlers(coord, state)
    names = {r.name for r in coord._registrations}  # noqa: SLF001
    assert "signal-store" in names
    assert "audit-ledger" not in names
    assert "sentinel-scanner" not in names
