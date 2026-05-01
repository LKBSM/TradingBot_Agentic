"""Tests for tools/governance/weekly_check.py (RISK-1.1 partial).

Focus on the pytest-output parser (the only piece complex enough to warrant
unit testing). The git/file collectors are I/O wrappers and are tested
implicitly by the smoke run.
"""

from __future__ import annotations

import pytest

from tools.governance.weekly_check import (
    FileFreshness,
    GitStats,
    PytestStats,
    _parse_pytest_output,
    overall_status,
)


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


def test_parse_pytest_summary_passed_only():
    text = "= 12 passed, 2 deselected, 1 warning in 7.13s ="
    s = _parse_pytest_output(text)
    assert s.available is True
    assert s.passed == 12
    assert s.failed == 0
    assert s.skipped == 0
    assert s.deselected == 2


def test_parse_pytest_summary_with_failures():
    text = "= 5 passed, 3 failed, 1 skipped in 4.21s ="
    s = _parse_pytest_output(text)
    assert s.passed == 5
    assert s.failed == 3
    assert s.skipped == 1


def test_parse_pytest_no_tests_ran():
    text = "= no tests ran in 0.05s ="
    s = _parse_pytest_output(text)
    assert s.passed == 0
    assert s.failed == 0


def test_parse_pytest_coverage_total_line():
    text = (
        "TOTAL                                238     46    81%\n"
        "Required test coverage of 70% reached. Total coverage: 81.09%\n"
        "= 12 passed, 2 deselected in 7s ="
    )
    s = _parse_pytest_output(text)
    assert s.coverage_pct == pytest.approx(81.0)


def test_parse_pytest_coverage_total_only():
    text = (
        "TOTAL                                238     46    77%\n"
        "= 10 passed in 5s ="
    )
    s = _parse_pytest_output(text)
    assert s.coverage_pct == pytest.approx(77.0)
    assert s.passed == 10


# ---------------------------------------------------------------------------
# Overall status logic
# ---------------------------------------------------------------------------


def _green_inputs() -> tuple[GitStats, PytestStats, FileFreshness]:
    git = GitStats(commits_count=5, last_commit_age_hours=24.0, branch="main")
    tests = PytestStats(
        available=True, passed=12, failed=0, coverage_pct=81.0
    )
    files = FileFreshness(
        kill_criteria_age_hours=24.0,
        blockers_age_hours=24.0,
        autonomous_log_age_hours=24.0,
    )
    return git, tests, files


def test_overall_status_green():
    git, tests, files = _green_inputs()
    assert overall_status(git, tests, files) == "green"


def test_overall_status_red_on_test_failure():
    git, tests, files = _green_inputs()
    tests.failed = 1
    assert overall_status(git, tests, files) == "red"


def test_overall_status_red_on_stale_commits():
    git, tests, files = _green_inputs()
    git.last_commit_age_hours = 14 * 24 + 1
    assert overall_status(git, tests, files) == "red"


def test_overall_status_yellow_when_tests_unavailable():
    git, tests, files = _green_inputs()
    tests.available = False
    assert overall_status(git, tests, files) == "yellow"


def test_overall_status_yellow_on_low_coverage():
    git, tests, files = _green_inputs()
    tests.coverage_pct = 65.0
    assert overall_status(git, tests, files) == "yellow"


def test_overall_status_yellow_on_stale_board():
    git, tests, files = _green_inputs()
    files.kill_criteria_age_hours = 11 * 24
    assert overall_status(git, tests, files) == "yellow"
