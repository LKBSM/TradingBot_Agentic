# =============================================================================
# Sprint 15 Validation: Pin Colab Training to Verified Commit (SEC-1)
# =============================================================================
# Verifies that:
# 1. colab_training_full.py has VERIFIED_COMMIT constant
# 2. No more --depth 1 shallow clone without verification
# 3. Checksum verification function exists
# 4. hashlib is imported
#
# Run with: python -m pytest tests/test_sprint15_colab_pinning.py -v
# =============================================================================

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def _read_colab_script():
    """Read the colab training script."""
    path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'colab_training_full.py')
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: VERIFIED_COMMIT exists
# ─────────────────────────────────────────────────────────────────────────────
def test_verified_commit_exists():
    """Script should define VERIFIED_COMMIT constant."""
    content = _read_colab_script()
    assert 'VERIFIED_COMMIT' in content, "Missing VERIFIED_COMMIT in colab script"


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: No unverified shallow clone
# ─────────────────────────────────────────────────────────────────────────────
def test_no_unverified_shallow_clone():
    """Script should not use --depth 1 without commit verification."""
    content = _read_colab_script()
    # The old pattern was: git clone --depth 1 URL
    # Now should either not use --depth 1 or should checkout specific commit after
    assert 'git", "checkout", VERIFIED_COMMIT' in content, (
        "Script should checkout VERIFIED_COMMIT after clone"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: hashlib imported
# ─────────────────────────────────────────────────────────────────────────────
def test_hashlib_imported():
    """Script should import hashlib for checksum verification."""
    content = _read_colab_script()
    assert 'import hashlib' in content, "Missing hashlib import"


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Checksum verification function
# ─────────────────────────────────────────────────────────────────────────────
def test_checksum_verification_function():
    """Script should have a checksum verification function."""
    content = _read_colab_script()
    assert '_verify_checksums' in content, "Missing checksum verification function"
    assert 'sha256' in content, "Checksum should use SHA-256"


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: CRITICAL_FILE_CHECKSUMS exists
# ─────────────────────────────────────────────────────────────────────────────
def test_critical_file_checksums_exists():
    """Script should define CRITICAL_FILE_CHECKSUMS (even if None initially)."""
    content = _read_colab_script()
    assert 'CRITICAL_FILE_CHECKSUMS' in content


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: Sprint 15 comment present
# ─────────────────────────────────────────────────────────────────────────────
def test_sprint15_comment():
    """Script should have Sprint 15 documentation."""
    content = _read_colab_script()
    assert 'Sprint 15' in content
    assert 'supply-chain' in content.lower() or 'security' in content.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: Tamper detection message exists
# ─────────────────────────────────────────────────────────────────────────────
def test_tamper_detection_message():
    """Checksum failure should produce a clear security warning."""
    content = _read_colab_script()
    assert 'tampered' in content.lower(), "Missing tamper detection error message"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
