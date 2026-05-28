"""DG-041 CI gate — refuse PR/merge if SENTINEL_TESTING_MODE=1 lands in any
production-shipping config file.

What this catches
-----------------
- ``.env.production`` and variants (``*.env.production*``)
- ``fly.toml`` / ``fly.*.toml`` Fly.io secrets manifests
- ``infrastructure/Dockerfile`` *non-commented* RUN/ENV/CMD lines
- ``docker-compose*.yml`` / ``docker-compose*.yaml``

What this intentionally allows
------------------------------
- ``tests/**`` — pytest fixtures must be able to set =1 to test the
  TESTING_MODE branch itself.
- ``.github/workflows/algo_tests.yml`` — algo tests run with =1 to
  bypass auth in the CI pytest invocation; that's the documented
  pattern.
- Commented lines (``# SENTINEL_TESTING_MODE=1``) and docstrings —
  warning copy is fine.
- Local-only ``.env`` (untracked) — not visible to CI.

Exit
----
0 = clean, 1 = violation. Prints offending file/line on failure.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Files we actively scan
TARGET_PATTERNS = [
    "*.env.production*",
    "**/.env.production*",
    "fly.toml",
    "fly.*.toml",
    "infrastructure/fly.toml",
    "infrastructure/Dockerfile",
    "infrastructure/Dockerfile.*",
    "docker-compose*.yml",
    "docker-compose*.yaml",
    "infrastructure/docker-compose*.yml",
]

# Directories we explicitly avoid
EXCLUDE_DIRS = {
    "tests", "node_modules", ".git", "reports",
    "docs", "webapp", "scripts",
}

# Workflow files allowed to set TESTING_MODE=1 (CI / algo-tests jobs that
# legitimately exercise the bypassed-auth branch).
ALLOWLIST_WORKFLOWS = {
    ".github/workflows/algo_tests.yml",
    # ci.yml runs the gate itself; documentation copy in echo strings
    # would otherwise self-match.
    ".github/workflows/ci.yml",
}

MATCH_RE = re.compile(r'SENTINEL_TESTING_MODE\s*[:=]\s*"?1"?')


def is_commented(line: str) -> bool:
    s = line.lstrip()
    return s.startswith("#") or s.startswith("//") or s.startswith(";")


def collect_files() -> list[Path]:
    files: list[Path] = []
    # Targeted file patterns (explicit allow-list)
    for pattern in TARGET_PATTERNS:
        files.extend(ROOT.glob(pattern))
    # Plus all workflows except the allow-listed ones
    for wf in (ROOT / ".github" / "workflows").glob("*.y*ml"):
        rel = wf.relative_to(ROOT).as_posix()
        if rel not in ALLOWLIST_WORKFLOWS:
            files.append(wf)
    # De-duplicate, drop excluded dirs
    seen: set[Path] = set()
    out: list[Path] = []
    for f in files:
        if f in seen:
            continue
        if any(part in EXCLUDE_DIRS for part in f.relative_to(ROOT).parts[:-1]):
            continue
        if not f.is_file():
            continue
        seen.add(f)
        out.append(f)
    return out


def main() -> int:
    violations: list[tuple[str, int, str]] = []
    for f in collect_files():
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if is_commented(line):
                continue
            if MATCH_RE.search(line):
                violations.append((f.relative_to(ROOT).as_posix(), i, line.strip()))

    if not violations:
        print("DG-041 TESTING_MODE prod gate: OK (no violations)")
        return 0

    print("::error::SENTINEL_TESTING_MODE=1 found in production-tracked files:")
    for path, lineno, snippet in violations:
        print(f"  {path}:{lineno}: {snippet}")
    print()
    print("TESTING_MODE bypasses API key auth and grants INSTITUTIONAL access.")
    print("Set to 0 (or unset) in production env files; only tests/ may set =1.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
