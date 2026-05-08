"""Tests for the API-2B.1 insight history endpoint + ledger.paginate()."""

from __future__ import annotations

import os

os.environ.setdefault("SENTINEL_TESTING_MODE", "1")

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.audit import HashChainLedger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _force_testing_mode():
    with (
        patch("src.api.auth.TESTING_MODE", True),
        patch("src.api.routes.insight_history.TESTING_MODE", True),
    ):
        yield


@pytest.fixture
def populated_ledger():
    led = HashChainLedger()
    for i in range(1, 11):  # 10 entries
        led.append({"id": f"insight-{i}", "value": i})
    return led


@pytest.fixture
def client(populated_ledger):
    return TestClient(create_app(audit_ledger=populated_ledger))


@pytest.fixture
def client_no_ledger():
    return TestClient(create_app())


# ---------------------------------------------------------------------------
# ledger.paginate() — direct unit tests
# ---------------------------------------------------------------------------


def test_paginate_default_returns_newest_first(populated_ledger):
    entries, next_cursor = populated_ledger.paginate(limit=3)
    assert [e.seq for e in entries] == [10, 9, 8]
    assert next_cursor == 8


def test_paginate_walks_to_exhaustion(populated_ledger):
    seen = []
    cursor = None
    while True:
        entries, cursor = populated_ledger.paginate(cursor=cursor, limit=4)
        seen.extend(e.seq for e in entries)
        if cursor is None:
            break
    assert seen == list(range(10, 0, -1))


def test_paginate_filter_by_insight_id(populated_ledger):
    populated_ledger.append({"id": "insight-3", "value": 99})  # second entry for id 3
    entries, next_cursor = populated_ledger.paginate(insight_id="insight-3", limit=10)
    assert {e.insight_id for e in entries} == {"insight-3"}
    assert len(entries) == 2
    assert next_cursor is None


def test_paginate_filter_by_time_range_future_window_returns_empty(populated_ledger):
    """Sanity: a window far in the future excludes everything."""
    entries, next_cursor = populated_ledger.paginate(
        since_iso="2099-01-01T00:00:00.000000Z", limit=50
    )
    assert entries == []
    assert next_cursor is None


def test_paginate_filter_by_time_range_past_window_returns_empty(populated_ledger):
    """Sanity: a window strictly in the past excludes everything."""
    entries, _ = populated_ledger.paginate(
        until_iso="1999-01-01T00:00:00.000000Z", limit=50
    )
    assert entries == []


def test_paginate_clamps_limit_to_500(populated_ledger):
    entries, _ = populated_ledger.paginate(limit=10_000)
    assert len(entries) == 10  # only 10 in fixture, but no error raised


def test_paginate_rejects_zero_limit(populated_ledger):
    with pytest.raises(ValueError):
        populated_ledger.paginate(limit=0)


def test_paginate_rejects_negative_cursor(populated_ledger):
    with pytest.raises(ValueError):
        populated_ledger.paginate(cursor=0)


# ---------------------------------------------------------------------------
# /api/v1/insights/history — endpoint tests
# ---------------------------------------------------------------------------


def test_history_default_returns_newest_50(client):
    resp = client.get("/api/v1/insights/history")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["entries"]) == 10
    assert body["entries"][0]["seq"] == 10
    assert body["entries"][-1]["seq"] == 1
    assert body["has_more"] is False
    assert body["next_cursor"] is None
    assert body["head_seq"] == 10
    assert len(body["head_hash"]) == 64


def test_history_etag_header_includes_head_hash(client):
    resp = client.get("/api/v1/insights/history")
    etag = resp.headers["etag"]
    assert etag.startswith('W/"')
    assert resp.headers["x-ledger-head-seq"] == "10"


def test_history_pagination_round_trip(client):
    page1 = client.get("/api/v1/insights/history?limit=4").json()
    assert [e["seq"] for e in page1["entries"]] == [10, 9, 8, 7]
    assert page1["next_cursor"] == 7

    page2 = client.get(
        f"/api/v1/insights/history?limit=4&cursor={page1['next_cursor']}"
    ).json()
    assert [e["seq"] for e in page2["entries"]] == [6, 5, 4, 3]

    page3 = client.get(
        f"/api/v1/insights/history?limit=4&cursor={page2['next_cursor']}"
    ).json()
    assert [e["seq"] for e in page3["entries"]] == [2, 1]
    assert page3["has_more"] is False


def test_history_filter_by_insight_id(client):
    resp = client.get("/api/v1/insights/history?insight_id=insight-7")
    body = resp.json()
    assert len(body["entries"]) == 1
    assert body["entries"][0]["insight_id"] == "insight-7"
    assert body["filters"]["insight_id"] == "insight-7"


def test_history_invalid_cursor_returns_422(client):
    # FastAPI's Query(ge=1) validation returns 422, not 400
    resp = client.get("/api/v1/insights/history?cursor=0")
    assert resp.status_code == 422


def test_history_invalid_limit_returns_422(client):
    resp = client.get("/api/v1/insights/history?limit=0")
    assert resp.status_code == 422
    resp = client.get("/api/v1/insights/history?limit=501")
    assert resp.status_code == 422


def test_history_503_when_no_ledger(client_no_ledger):
    resp = client_no_ledger.get("/api/v1/insights/history")
    assert resp.status_code == 503


def test_history_tier_gate_blocks_free_when_not_testing(populated_ledger):
    """Outside TESTING_MODE, FREE/ANALYST tiers are blocked."""
    with (
        patch("src.api.auth.TESTING_MODE", False),
        patch("src.api.routes.insight_history.TESTING_MODE", False),
    ):
        # Build a fake key_store that returns a FREE-tier subscriber.
        class _FakeStore:
            def verify_key(self, _):
                return {"key_id": 1, "label": "free", "created_at": "x"}

            def check_rate_limit(self, _):
                return True

            def record_usage(self, *a):
                pass

        c = TestClient(
            create_app(audit_ledger=populated_ledger, key_store=_FakeStore())
        )
        resp = c.get(
            "/api/v1/insights/history",
            headers={"X-API-Key": "sk_dev_free"},
        )
        assert resp.status_code == 403
        assert "STRATEGIST" in resp.json()["detail"]
