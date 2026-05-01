"""Sofia weekly check — Friday 16:00 ET status report.

Sprint RISK-1.1 partial (Sofia). Produces a one-page status report for the
weekly governance review. Designed to be RUN, not to auto-update the board:
auto-edits to a governance doc could mask important context. Sofia reviews
the printed output and updates `kill_criteria_board.md` manually.

Usage::

    python -m tools.governance.weekly_check                # full report
    python -m tools.governance.weekly_check --since 14d    # custom window
    python -m tools.governance.weekly_check --json         # machine-readable

Exit codes:
    0 = report produced (any KPI status)
    2 = at least one critical metric in red zone (CI / signal flow)
    3 = command failed (git not available, etc.)
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# Resolve repo root by walking up from this file.
REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GitStats:
    commits_count: int = 0
    commits_summary: list[str] = field(default_factory=list)
    last_commit_age_hours: float | None = None
    branch: str = ""


@dataclass
class PytestStats:
    available: bool = False
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    deselected: int = 0
    coverage_pct: float | None = None
    error: str | None = None


@dataclass
class FileFreshness:
    kill_criteria_age_hours: float | None = None
    blockers_age_hours: float | None = None
    autonomous_log_age_hours: float | None = None


@dataclass
class CheckResult:
    timestamp_utc: str
    since_window: str
    git: GitStats
    tests: PytestStats
    files: FileFreshness
    overall_status: str  # green | yellow | red


# ---------------------------------------------------------------------------
# Collectors
# ---------------------------------------------------------------------------


def _run_git(args: list[str]) -> str:
    """Run a git command from REPO_ROOT, return stdout. Empty string on error."""
    if not shutil.which("git"):
        return ""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return ""
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, OSError):
        return ""


def collect_git(since: str) -> GitStats:
    """Recent commit history + freshness."""
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    log = _run_git(["log", "--oneline", f"--since={since}"])
    summary = log.splitlines() if log else []

    last_commit_iso = _run_git(["log", "-1", "--format=%cI"])
    last_age: float | None = None
    if last_commit_iso:
        try:
            last_dt = datetime.fromisoformat(last_commit_iso)
            now = datetime.now(timezone.utc)
            last_age = (now - last_dt).total_seconds() / 3600
        except ValueError:
            last_age = None

    return GitStats(
        commits_count=len(summary),
        commits_summary=summary,
        last_commit_age_hours=last_age,
        branch=branch,
    )


def collect_tests(target_paths: list[str] | None = None) -> PytestStats:
    """Run pytest with coverage on the data sprint tests, parse output.

    Default target = the curated subset that already passes in CI. We do not
    invoke the full legacy suite here (many tests depend on local CSVs that
    aren't part of weekly check scope).
    """
    if not shutil.which("python"):
        return PytestStats(available=False, error="python not on PATH")

    targets = target_paths or [
        "tests/test_fred_provider.py",
        "tests/test_cot_provider.py",
    ]
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *targets,
        "-m",
        "not live",
        "--cov=src/agents/data",
        "--no-header",
        "-q",
        "--tb=no",
    ]
    try:
        # The pytest run can take 5-15s on the data tests.
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            env={"SKIP_LIVE_NETWORK_TESTS": "1", **dict_env()},
        )
    except subprocess.TimeoutExpired:
        return PytestStats(available=False, error="pytest timeout (>120s)")
    except OSError as exc:
        return PytestStats(available=False, error=f"pytest failed: {exc}")

    out = result.stdout + "\n" + result.stderr
    return _parse_pytest_output(out)


def dict_env() -> dict:
    """Lazy import of os to keep top-level minimal."""
    import os

    return os.environ.copy()


def _parse_pytest_output(text: str) -> PytestStats:
    """Parse the pytest summary line + coverage % from a captured output."""
    stats = PytestStats(available=True)

    # Find the last line that contains "passed" or "failed" or "no tests ran"
    summary_lines = [
        line
        for line in text.splitlines()
        if "passed" in line or "failed" in line or "no tests ran" in line
    ]
    if summary_lines:
        line = summary_lines[-1]
        # Search each count independently — much more reliable than one big
        # regex with optional groups (which can match an empty span).
        for attr, pat in [
            ("passed", r"(\d+) passed"),
            ("failed", r"(\d+) failed"),
            ("skipped", r"(\d+) skipped"),
            ("deselected", r"(\d+) deselected"),
        ]:
            m = re.search(pat, line)
            if m:
                setattr(stats, attr, int(m.group(1)))

    # Coverage line like "TOTAL ... 81%" or "Total coverage: 81.09%"
    cov_re = re.compile(r"(?:Total coverage|TOTAL).*?(\d+(?:\.\d+)?)%")
    for line in text.splitlines():
        m = cov_re.search(line)
        if m:
            stats.coverage_pct = float(m.group(1))
            break

    return stats


def collect_file_freshness() -> FileFreshness:
    """How stale are the governance docs? Late updates → drift signal."""
    paths = {
        "kill_criteria_age_hours": REPO_ROOT
        / "reports"
        / "governance"
        / "kill_criteria_board.md",
        "blockers_age_hours": REPO_ROOT
        / "reports"
        / "governance"
        / "BLOCKERS.md",
        "autonomous_log_age_hours": REPO_ROOT
        / "reports"
        / "governance"
        / "autonomous_session_log.md",
    }
    out: dict = {}
    now_ts = datetime.now(timezone.utc).timestamp()
    for attr, p in paths.items():
        if p.exists():
            mtime = p.stat().st_mtime
            out[attr] = (now_ts - mtime) / 3600
        else:
            out[attr] = None
    return FileFreshness(**out)


# ---------------------------------------------------------------------------
# Status assessment
# ---------------------------------------------------------------------------


def overall_status(git: GitStats, tests: PytestStats, files: FileFreshness) -> str:
    """Reduce all signals to green/yellow/red.

    Red: tests failing, or > 14 days no commit, or kill_criteria_board > 14 days stale.
    Yellow: tests not runnable, or > 7 days no commit, or board > 10 days stale.
    Green: everything inside thresholds.
    """
    if tests.available and tests.failed > 0:
        return "red"
    if git.last_commit_age_hours is not None and git.last_commit_age_hours > 14 * 24:
        return "red"
    if (
        files.kill_criteria_age_hours is not None
        and files.kill_criteria_age_hours > 14 * 24
    ):
        return "red"

    if not tests.available:
        return "yellow"
    if git.last_commit_age_hours is not None and git.last_commit_age_hours > 7 * 24:
        return "yellow"
    if (
        files.kill_criteria_age_hours is not None
        and files.kill_criteria_age_hours > 10 * 24
    ):
        return "yellow"
    if tests.coverage_pct is not None and tests.coverage_pct < 70:
        return "yellow"

    return "green"


def collect_all(since: str = "7 days ago") -> CheckResult:
    git = collect_git(since)
    tests = collect_tests()
    files = collect_file_freshness()
    status = overall_status(git, tests, files)
    return CheckResult(
        timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        since_window=since,
        git=git,
        tests=tests,
        files=files,
        overall_status=status,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_text(result: CheckResult) -> str:
    """One-page text report for the terminal."""
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append(f"Sofia weekly check - {result.timestamp_utc}")
    lines.append(f"Window: {result.since_window}    Branch: {result.git.branch}")
    lines.append(f"Overall status: {result.overall_status.upper()}")
    lines.append("=" * 70)

    lines.append("")
    lines.append("[GIT]")
    lines.append(f"  Commits in window:        {result.git.commits_count}")
    if result.git.last_commit_age_hours is not None:
        lines.append(
            f"  Last commit age:          {result.git.last_commit_age_hours:.1f} h"
        )
    if result.git.commits_summary:
        lines.append("  Commits:")
        for c in result.git.commits_summary[:15]:
            lines.append(f"    - {c}")
        if len(result.git.commits_summary) > 15:
            lines.append(
                f"    ... and {len(result.git.commits_summary) - 15} more"
            )

    lines.append("")
    lines.append("[TESTS]")
    if not result.tests.available:
        lines.append(f"  pytest unavailable: {result.tests.error}")
    else:
        lines.append(f"  Passed:    {result.tests.passed}")
        lines.append(f"  Failed:    {result.tests.failed}")
        lines.append(f"  Skipped:   {result.tests.skipped}")
        lines.append(f"  Deselected:{result.tests.deselected}")
        if result.tests.coverage_pct is not None:
            lines.append(
                f"  Coverage:  {result.tests.coverage_pct:.1f}% (gate 70%)"
            )

    lines.append("")
    lines.append("[GOVERNANCE FILE FRESHNESS]")
    f = result.files
    if f.kill_criteria_age_hours is not None:
        lines.append(
            f"  kill_criteria_board.md:   {f.kill_criteria_age_hours:.1f} h"
        )
    if f.blockers_age_hours is not None:
        lines.append(f"  BLOCKERS.md:              {f.blockers_age_hours:.1f} h")
    if f.autonomous_log_age_hours is not None:
        lines.append(
            f"  autonomous_session_log.md:{f.autonomous_log_age_hours:.1f} h"
        )

    lines.append("")
    lines.append("[NEXT ACTIONS]")
    lines.extend(_render_next_actions(result))
    lines.append("=" * 70)
    return "\n".join(lines)


def _render_next_actions(result: CheckResult) -> list[str]:
    """Suggest concrete next actions based on what's red/yellow."""
    actions: list[str] = []
    if result.tests.available and result.tests.failed > 0:
        actions.append(
            f"  - Fix {result.tests.failed} failing test(s) before next sprint"
        )
    if (
        result.tests.coverage_pct is not None
        and result.tests.coverage_pct < 70
    ):
        actions.append("  - Coverage < 70% gate; add tests to data sprints")
    if result.git.last_commit_age_hours and result.git.last_commit_age_hours > 7 * 24:
        actions.append(
            "  - Last commit > 7 days; check sprint progress in board"
        )
    if (
        result.files.kill_criteria_age_hours is not None
        and result.files.kill_criteria_age_hours > 10 * 24
    ):
        actions.append(
            "  - kill_criteria_board.md not updated > 10 days; review and refresh"
        )
    if not actions:
        actions.append(
            "  - All KPIs within thresholds. Pick next sprint per "
            "PLAN_12_MOIS.md and update board with intent."
        )
    return actions


def render_json(result: CheckResult) -> str:
    """Machine-readable variant for piping into dashboards."""
    return json.dumps(asdict(result), indent=2, default=str)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="Sofia weekly governance check — status report.",
    )
    parser.add_argument(
        "--since",
        default="7 days ago",
        help="Window for git stats (any expression accepted by `git log --since=`)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of the human-readable report.",
    )
    args = parser.parse_args(argv)

    result = collect_all(since=args.since)
    print(render_json(result) if args.json else render_text(result))

    if result.overall_status == "red":
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
