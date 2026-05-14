"""Tests for the API-2B.7 OpenAPI enrichment."""

from __future__ import annotations

import os

os.environ.setdefault("SENTINEL_TESTING_MODE", "1")

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.openapi_enrichment import _clean_operation_id


# ---------------------------------------------------------------------------
# Unit — _clean_operation_id
# ---------------------------------------------------------------------------


def test_clean_op_id_strips_api_v1_prefix():
    assert _clean_operation_id("GET", "/api/v1/insights/history").startswith(
        "get_insights"
    )


def test_clean_op_id_replaces_path_params_with_underscores():
    assert _clean_operation_id(
        "POST", "/api/v1/webhooks/deliveries/{delivery_id}/ack"
    ) == "post_webhooks_deliveries_delivery_id_ack"


def test_clean_op_id_collapses_consecutive_underscores():
    out = _clean_operation_id("GET", "/api/v1/a//{b}/c")
    assert "__" not in out
    assert out == "get_a_b_c"


def test_clean_op_id_handles_root_path():
    out = _clean_operation_id("GET", "/")
    assert out == "get_root"


# ---------------------------------------------------------------------------
# Integration — spec contains enriched fields
# ---------------------------------------------------------------------------


@pytest.fixture
def spec():
    client = TestClient(create_app())
    return client.get("/openapi.json").json()


def test_spec_includes_production_servers(spec):
    servers = spec.get("servers", [])
    urls = {s["url"] for s in servers}
    assert "https://api.smartsentinel.ai" in urls


def test_spec_tags_have_descriptions(spec):
    tags = {t["name"]: t.get("description", "") for t in spec.get("tags", [])}
    # Every expected tag carries a non-empty description.
    for name in ("insights", "audit", "webhooks", "health", "metrics", "admin"):
        assert name in tags, f"missing tag {name}"
        assert tags[name].strip(), f"empty description for {name}"


def test_every_operation_has_operation_id(spec):
    for path, ops in spec["paths"].items():
        for method, op in ops.items():
            if method in {"get", "post", "put", "delete", "patch"}:
                assert "operationId" in op, f"{method} {path} missing operationId"
                # snake_case prefix matches the method
                assert op["operationId"].startswith(method.lower() + "_") or (
                    op["operationId"] == method.lower() + "_root"
                )


def test_operation_ids_are_unique(spec):
    seen = []
    for path, ops in spec["paths"].items():
        for method, op in ops.items():
            if method in {"get", "post", "put", "delete", "patch"}:
                seen.append(op["operationId"])
    assert len(seen) == len(set(seen)), "duplicate operationId in OpenAPI spec"


def test_spec_description_carries_brand_blurb(spec):
    info = spec.get("info", {})
    assert "Smart Sentinel" in info.get("description", "")


def test_openapi_cached_after_first_call():
    """Second call returns the cached schema (not regenerated)."""
    app = create_app()
    s1 = app.openapi()
    s2 = app.openapi()
    assert s1 is s2  # same object → cache hit
