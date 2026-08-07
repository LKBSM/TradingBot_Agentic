"""AUTH-HARDENING — brute-force / abuse throttle on the auth endpoints.

Unit tests for the sliding-window ``AuthThrottle`` and route-level tests proving
login / register / password-reset are throttled, that a successful login clears
the brake (so a legit user who mistypes is never locked), and that the window
drains over time.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.account_store import AccountStore
from src.api.app import create_app
from src.api.auth_throttle import AuthThrottle


# =============================================================================
# Unit — AuthThrottle
# =============================================================================

class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def test_hit_allows_up_to_cap_then_refuses():
    clock = _Clock()
    t = AuthThrottle(max_attempts=3, window_s=60, clock=clock)
    assert t.hit("k") == (True, 0)
    assert t.hit("k") == (True, 0)
    assert t.hit("k") == (True, 0)
    allowed, retry = t.hit("k")
    assert allowed is False
    assert retry >= 1  # told when to come back


def test_blocked_is_read_only_and_reflects_cap():
    clock = _Clock()
    t = AuthThrottle(max_attempts=2, window_s=60, clock=clock)
    assert t.blocked("k") == (False, 0)
    t.hit("k")
    assert t.blocked("k") == (False, 0)
    t.hit("k")
    blocked, retry = t.blocked("k")
    assert blocked is True and retry >= 1
    # blocked() did not itself record anything — still exactly at cap, not beyond.
    assert t.blocked("k")[0] is True


def test_window_drains_over_time():
    clock = _Clock()
    t = AuthThrottle(max_attempts=2, window_s=60, clock=clock)
    t.hit("k")
    t.hit("k")
    assert t.blocked("k")[0] is True
    clock.advance(61)  # both hits age out
    assert t.blocked("k")[0] is False
    assert t.hit("k") == (True, 0)


def test_reset_clears_key():
    t = AuthThrottle(max_attempts=1, window_s=60)
    t.hit("k")
    assert t.blocked("k")[0] is True
    t.reset("k")
    assert t.blocked("k")[0] is False


def test_refused_attempt_does_not_extend_window():
    clock = _Clock()
    t = AuthThrottle(max_attempts=1, window_s=60, clock=clock)
    t.hit("k")               # at cap, recorded at t=1000
    clock.advance(30)
    assert t.hit("k")[0] is False  # refused, NOT recorded
    clock.advance(31)        # 61s since the ONLY real hit → drains
    assert t.blocked("k")[0] is False


def test_zero_max_disables_throttle():
    t = AuthThrottle(max_attempts=0, window_s=60)
    assert t.enabled is False
    for _ in range(100):
        assert t.hit("k") == (True, 0)
    assert t.blocked("k") == (False, 0)


def test_keys_are_independent():
    t = AuthThrottle(max_attempts=1, window_s=60)
    t.hit("a")
    assert t.blocked("a")[0] is True
    assert t.blocked("b")[0] is False


# =============================================================================
# Route-level — login / register / reset
# =============================================================================

VALID_REGISTER = {
    "username": "alice",
    "email": "alice@example.com",
    "password": "longpassword1",
    "age_confirmed": True,
    "accept_terms": True,
    "accept_privacy": True,
}


@pytest.fixture()
def make_client(tmp_path, monkeypatch):
    """Build a TestClient with a chosen throttle cap (env is read at app build)."""

    def _make(max_attempts: int):
        monkeypatch.setenv("SESSION_COOKIE_SECURE", "0")
        monkeypatch.setenv("SESSION_SECRET", "test-session-secret-value")
        monkeypatch.setenv("AUTH_THROTTLE_MAX_ATTEMPTS", str(max_attempts))
        monkeypatch.setenv("AUTH_THROTTLE_WINDOW_S", "300")
        store = AccountStore(db_path=str(tmp_path / "throttle_accounts.db"))
        app = create_app(account_store=store)
        return TestClient(app)

    return _make


def test_login_brute_force_returns_429(make_client):
    client = make_client(max_attempts=3)
    client.post("/api/auth/register", json=VALID_REGISTER)
    client.post("/api/auth/logout")
    bad = {"identifier": "alice@example.com", "password": "WRONGpassword9"}
    for _ in range(3):
        assert client.post("/api/auth/login", json=bad).status_code == 401
    # The 4th attempt is refused BEFORE credentials are checked.
    blocked = client.post("/api/auth/login", json=bad)
    assert blocked.status_code == 429
    assert "Retry-After" in blocked.headers
    # Even the CORRECT password is now refused (the IP is locked for the window).
    good = {"identifier": "alice@example.com", "password": "longpassword1"}
    assert client.post("/api/auth/login", json=good).status_code == 429


def test_successful_login_clears_the_counter(make_client):
    client = make_client(max_attempts=3)
    client.post("/api/auth/register", json=VALID_REGISTER)
    client.post("/api/auth/logout")
    bad = {"identifier": "alice@example.com", "password": "WRONGpassword9"}
    good = {"identifier": "alice@example.com", "password": "longpassword1"}
    # Two failures (under the cap), then a success clears the brake…
    assert client.post("/api/auth/login", json=bad).status_code == 401
    assert client.post("/api/auth/login", json=bad).status_code == 401
    assert client.post("/api/auth/login", json=good).status_code == 200
    client.post("/api/auth/logout")
    # …so two more failures do NOT immediately trip 429 (counter was reset).
    assert client.post("/api/auth/login", json=bad).status_code == 401
    assert client.post("/api/auth/login", json=bad).status_code == 401


def test_register_throttled_per_ip(make_client):
    client = make_client(max_attempts=2)
    # attempt 1 → 201, attempt 2 (conflict) → 409, attempt 3 → refused 429.
    assert client.post("/api/auth/register", json=VALID_REGISTER).status_code == 201
    client.post("/api/auth/logout")
    assert client.post("/api/auth/register", json=VALID_REGISTER).status_code == 409
    assert client.post("/api/auth/register", json=VALID_REGISTER).status_code == 429


def test_register_conflict_message_is_generic(make_client):
    # AUTH-HARDENING anti-enumeration: the conflict must NOT reveal whether the
    # email or the username is the one already taken.
    client = make_client(max_attempts=10)
    assert client.post("/api/auth/register", json=VALID_REGISTER).status_code == 201
    client.post("/api/auth/logout")
    # Same email, DIFFERENT username → still generic (does not say "email taken").
    dupe = {**VALID_REGISTER, "username": "bob"}
    resp = client.post("/api/auth/register", json=dupe)
    assert resp.status_code == 409
    detail = resp.json()["detail"].lower()
    assert "e-mail" not in detail and "email" not in detail
    assert "nom d'utilisateur" not in detail and "utilisateur" not in detail


def test_reset_request_throttled_per_ip(make_client):
    client = make_client(max_attempts=2)
    body = {"identifier": "someone@example.com"}
    assert client.post("/api/auth/password-reset/request", json=body).status_code == 200
    assert client.post("/api/auth/password-reset/request", json=body).status_code == 200
    refused = client.post("/api/auth/password-reset/request", json=body)
    assert refused.status_code == 429
    assert "Retry-After" in refused.headers
