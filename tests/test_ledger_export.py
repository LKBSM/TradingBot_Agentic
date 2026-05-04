"""Tests for the DATA-2B.6 ledger export."""

from __future__ import annotations

import csv
import io
import json

import pytest

from src.audit import HashChainLedger
from src.audit.ledger_export import (
    CSV_COLUMNS,
    to_csv,
    to_csv_string,
    to_jsonl,
    to_jsonl_string,
)


@pytest.fixture
def populated_ledger():
    led = HashChainLedger()
    for i in range(1, 6):
        led.append({"id": f"insight-{i}", "value": i, "label": f"row-{i}"})
    return led


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def test_csv_first_yield_is_header(populated_ledger):
    gen = to_csv(populated_ledger)
    header = next(gen)
    assert header.strip().split(",") == list(CSV_COLUMNS)


def test_csv_streams_one_row_per_entry(populated_ledger):
    chunks = list(to_csv(populated_ledger))
    # 1 header + 5 data rows ⇒ 6 chunks
    assert len(chunks) == 6


def test_csv_string_round_trips_through_csv_module(populated_ledger):
    blob = to_csv_string(populated_ledger)
    rows = list(csv.DictReader(io.StringIO(blob)))
    assert len(rows) == 5
    assert rows[0]["seq"] == "1"
    assert rows[0]["insight_id"] == "insight-1"
    assert len(rows[0]["entry_hash"]) == 64


def test_csv_handles_unicode_payload():
    led = HashChainLedger()
    led.append({"id": "fr-1", "narrative": "Synthèse haussière sur XAU"})
    blob = to_csv_string(led)
    assert "Synthèse" in blob
    rows = list(csv.DictReader(io.StringIO(blob)))
    assert rows[0]["insight_id"] == "fr-1"


def test_csv_quotes_fields_with_commas():
    led = HashChainLedger()
    led.append({"id": "x", "narrative": "Hello, world, with commas"})
    blob = to_csv_string(led)
    rows = list(csv.DictReader(io.StringIO(blob)))
    # canonical_json contains the comma-laden string and should round-trip
    body = json.loads(rows[0]["canonical_json"])
    assert body["narrative"] == "Hello, world, with commas"


# ---------------------------------------------------------------------------
# JSONL
# ---------------------------------------------------------------------------


def test_jsonl_yields_one_json_per_line(populated_ledger):
    chunks = list(to_jsonl(populated_ledger))
    assert len(chunks) == 5
    for chunk in chunks:
        assert chunk.endswith("\n")
        json.loads(chunk)  # must parse


def test_jsonl_string_parses_fully(populated_ledger):
    blob = to_jsonl_string(populated_ledger)
    lines = blob.strip().split("\n")
    parsed = [json.loads(line) for line in lines]
    assert [p["seq"] for p in parsed] == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def test_min_seq_filter(populated_ledger):
    blob = to_csv_string(populated_ledger, min_seq=3)
    rows = list(csv.DictReader(io.StringIO(blob)))
    assert [int(r["seq"]) for r in rows] == [3, 4, 5]


def test_max_seq_filter(populated_ledger):
    blob = to_csv_string(populated_ledger, max_seq=2)
    rows = list(csv.DictReader(io.StringIO(blob)))
    assert [int(r["seq"]) for r in rows] == [1, 2]


def test_seq_window_filter(populated_ledger):
    blob = to_csv_string(populated_ledger, min_seq=2, max_seq=4)
    rows = list(csv.DictReader(io.StringIO(blob)))
    assert [int(r["seq"]) for r in rows] == [2, 3, 4]


def test_since_until_iso_filter(populated_ledger):
    # All entries are inserted within microseconds; pick a since=now-future
    # bound that excludes everything to verify the date filter at all.
    blob = to_csv_string(populated_ledger, since_iso="2099-01-01T00:00:00Z")
    rows = list(csv.DictReader(io.StringIO(blob)))
    assert rows == []


def test_since_iso_keeps_recent_entries(populated_ledger):
    """A since= bound from before the chain started lets every entry through."""
    blob = to_csv_string(populated_ledger, since_iso="2000-01-01T00:00:00Z")
    rows = list(csv.DictReader(io.StringIO(blob)))
    assert len(rows) == 5


# ---------------------------------------------------------------------------
# Iterator passthrough
# ---------------------------------------------------------------------------


def test_export_accepts_raw_iterable(populated_ledger):
    """Callers may pass an arbitrary iterable of LedgerEntry."""
    entries = list(populated_ledger.iter_entries())
    blob = to_csv_string(entries)
    rows = list(csv.DictReader(io.StringIO(blob)))
    assert len(rows) == 5


def test_jsonl_with_filter_returns_subset(populated_ledger):
    blob = to_jsonl_string(populated_ledger, min_seq=4)
    parsed = [json.loads(line) for line in blob.strip().split("\n")]
    assert [p["seq"] for p in parsed] == [4, 5]
