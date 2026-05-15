"""Conformal layer — Sprint 4 (calibration & confidence).

This package provides :

- ``mondrian`` — Mondrian conformal prediction stratified by regime
  (HMM low / normal / high or BOCPD changepoint state). Audit 3.6 P1-9
  recommends this over the existing :class:`TCPForecaster` to recover
  exchangeability within regime strata.

Re-exports the legacy :class:`TCPForecaster` for backwards compat — the
new Mondrian wrapper supersedes it for the calibration use case but the
legacy interface keeps existing tests green.

Status (Sprint 4 prep) : **scaffold + algorithmic core**. Empirical
validation OOS (Sprint 4 batch 4.3) requires the trade log produced by
the new Logistic L1 scorer pipeline.
"""

from src.intelligence.conformal.mondrian import MondrianConformal  # noqa: F401

__all__ = ["MondrianConformal"]
