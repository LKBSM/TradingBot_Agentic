"""Audit trail subsystem — Sprint DATA-2B.4.

Append-only, hash-chained ledger of every InsightSignalV2 delivered to a
client. Designed so that B2B compliance auditors can independently verify
that the historical sequence of insights has not been tampered with.
"""

from src.audit.hash_chain_ledger import (
    HashChainLedger,
    LedgerEntry,
    VerificationResult,
    canonical_json,
)

__all__ = [
    "HashChainLedger",
    "LedgerEntry",
    "VerificationResult",
    "canonical_json",
]
