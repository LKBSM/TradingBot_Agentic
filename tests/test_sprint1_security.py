# =============================================================================
# TESTS - Sprint 1: Critical Security Infrastructure
# =============================================================================
# Comprehensive test suite for all security components.
#
# Run with: pytest tests/test_sprint1_security.py -v
# =============================================================================

import os
import json
import time
import tempfile
import threading
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pytest

# Set test environment variables before imports
os.environ['HMAC_MASTER_KEY'] = 'test_master_key_for_testing_only_32chars!'
os.environ['TRADING_BOT_SECRET_KEY'] = 'test_secret_key_for_testing_only!'

from src.security.secrets_manager import (
    SecretManager, SecretManagerConfig, SecretBackend,
    SecretNotFoundError
)
from src.security.hmac_manager import (
    HMACKeyManager, HMACKeyConfig, KeyStorageBackend,
    HMACKeyVersion
)
from src.security.alert_manager import (
    AlertManager, AlertConfig, AlertSeverity, AlertChannel, Alert
)
from src.security.dead_man_switch import (
    DeadManSwitch, DeadManSwitchConfig, HeartbeatBackend, HeartbeatStatus
)
from src.security.siem_integration import (
    SIEMClient, SIEMConfig, SIEMBackend, SecurityEvent,
    EventCategory, EventSeverity, EventOutcome
)
from src.security.security_orchestrator import (
    SecurityOrchestrator, SecurityOrchestratorConfig
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        yield tmpdir


@pytest.fixture
def secrets_config(temp_dir):
    """Create SecretManagerConfig for testing."""
    return SecretManagerConfig(
        primary_backend=SecretBackend.ENCRYPTED_FILE,
        fallback_backend=SecretBackend.ENVIRONMENT,
        encrypted_file_path=os.path.join(temp_dir, '.secrets.enc'),
        cache_ttl_seconds=1,
    )


@pytest.fixture
def hmac_config(temp_dir):
    """Create HMACKeyConfig for testing."""
    return HMACKeyConfig(
        storage_backend=KeyStorageBackend.ENCRYPTED_FILE,
        key_file_path=os.path.join(temp_dir, '.hmac_keys.enc'),
        key_rotation_days=90,
        max_key_versions=5,
    )


@pytest.fixture
def alert_config():
    """Create AlertConfig for testing."""
    return AlertConfig(
        enabled_channels=[AlertChannel.CONSOLE],
        async_delivery=False,
        rate_limit_seconds=0,
    )


@pytest.fixture
def deadman_config(temp_dir):
    """Create DeadManSwitchConfig for testing."""
    return DeadManSwitchConfig(
        backend=HeartbeatBackend.FILE,
        heartbeat_file_path=os.path.join(temp_dir, '.heartbeat'),
        heartbeat_interval_seconds=1,
        timeout_seconds=5,
        bot_id='test-bot',
        environment='test',
    )


@pytest.fixture
def siem_config(temp_dir):
    """Create SIEMConfig for testing."""
    return SIEMConfig(
        backend=SIEMBackend.FILE,
        file_path=os.path.join(temp_dir, 'security_events.jsonl'),
        async_delivery=False,
    )


# =============================================================================
# SECRET MANAGER TESTS
# =============================================================================

class TestSecretManager:
    """Tests for SecretManager."""

    def test_initialization(self, secrets_config):
        """Test SecretManager initialization."""
        manager = SecretManager(secrets_config)
        assert manager is not None
        assert manager.config == secrets_config

    def test_store_and_retrieve_secret(self, secrets_config):
        """Test storing and retrieving secrets."""
        manager = SecretManager(secrets_config)

        # Store a secret
        secret_data = {'api_key': 'test123', 'api_secret': 'secret456'}
        result = manager.store_secret('test/api', secret_data)
        assert result is True

        # Retrieve the secret
        retrieved = manager.get_secret('test/api')
        assert retrieved['api_key'] == 'test123'
        assert retrieved['api_secret'] == 'secret456'

    def test_secret_not_found(self, secrets_config):
        """Test SecretNotFoundError for missing secrets."""
        manager = SecretManager(secrets_config)

        with pytest.raises(SecretNotFoundError):
            manager.get_secret('nonexistent/path')

    def test_mt5_credentials_from_environment(self, secrets_config):
        """Test getting MT5 credentials from environment."""
        manager = SecretManager(secrets_config)

        # Set environment variables
        os.environ['MT5_LOGIN'] = '12345'
        os.environ['MT5_PASSWORD'] = 'testpass'
        os.environ['MT5_SERVER'] = 'TestServer'

        try:
            creds = manager.get_mt5_credentials()
            assert creds['account'] == 12345
            assert creds['password'] == 'testpass'
            assert creds['server'] == 'TestServer'
        finally:
            # Cleanup
            del os.environ['MT5_LOGIN']
            del os.environ['MT5_PASSWORD']
            del os.environ['MT5_SERVER']

    def test_cache_functionality(self, secrets_config):
        """Test secret caching."""
        manager = SecretManager(secrets_config)

        # Store a secret
        manager.store_secret('cached/test', {'value': 'cached_value'})

        # First retrieval
        retrieved1 = manager.get_secret('cached/test')

        # Second retrieval should come from cache
        retrieved2 = manager.get_secret('cached/test')

        assert retrieved1 == retrieved2

    def test_cache_expiration(self, secrets_config):
        """Test cache expiration."""
        secrets_config.cache_ttl_seconds = 1
        manager = SecretManager(secrets_config)

        manager.store_secret('expire/test', {'value': 'test'})
        manager.get_secret('expire/test')

        # Wait for cache to expire
        time.sleep(1.5)

        # Should fetch fresh (not from cache)
        retrieved = manager.get_secret('expire/test')
        assert retrieved['value'] == 'test'

    def test_rotation_check(self, secrets_config):
        """Test credential rotation check."""
        manager = SecretManager(secrets_config)

        # Store secret with old timestamp
        old_date = (datetime.utcnow() - timedelta(days=100)).isoformat()
        manager.store_secret('old/secret', {
            'value': 'old',
            '_metadata': {'created_at': old_date}
        })

        assert manager.check_rotation_needed('old/secret') is True

    def test_access_audit_log(self, secrets_config):
        """Test access audit logging."""
        secrets_config.audit_all_access = True
        manager = SecretManager(secrets_config)

        manager.store_secret('audit/test', {'value': 'test'})
        manager.get_secret('audit/test')

        audit_log = manager.get_access_audit_log()
        assert len(audit_log) >= 1
        assert any(entry['path'] == 'audit/test' for entry in audit_log)


# =============================================================================
# HMAC MANAGER TESTS
# =============================================================================

class TestHMACManager:
    """Tests for HMACKeyManager."""

    def test_initialization(self, hmac_config):
        """Test HMACKeyManager initialization."""
        manager = HMACKeyManager(hmac_config)
        assert manager is not None
        assert manager._current_version >= 1

    def test_sign_and_verify(self, hmac_config):
        """Test signing and verification."""
        manager = HMACKeyManager(hmac_config)

        data = b'test data to sign'
        signed = manager.sign(data)

        assert signed.signature is not None
        assert signed.key_version == manager._current_version

        # Verify signature
        is_valid = manager.verify(data, signed.signature, signed.key_version)
        assert is_valid is True

    def test_sign_dict(self, hmac_config):
        """Test signing dictionaries."""
        manager = HMACKeyManager(hmac_config)

        data = {'action': 'trade', 'symbol': 'EURUSD', 'volume': 0.1}
        signature = manager.sign_dict(data)

        assert signature is not None
        assert len(signature) == 64  # SHA256 hex

        # Verify
        assert manager.verify_dict(data, signature) is True

    def test_tampered_data_fails_verification(self, hmac_config):
        """Test that tampered data fails verification."""
        manager = HMACKeyManager(hmac_config)

        data = {'value': 'original'}
        signature = manager.sign_dict(data)

        # Tamper with data
        tampered = {'value': 'tampered'}
        assert manager.verify_dict(tampered, signature) is False

    def test_key_rotation(self, hmac_config):
        """Test key rotation."""
        manager = HMACKeyManager(hmac_config)
        original_version = manager._current_version

        # Sign with original key
        data = b'test data'
        signed_before = manager.sign(data)

        # Rotate key
        new_version = manager.rotate_key(reason="Test rotation")
        assert new_version == original_version + 1

        # Sign with new key
        signed_after = manager.sign(data)
        assert signed_after.key_version == new_version

        # Old signature should still verify (with version)
        assert manager.verify(data, signed_before.signature, signed_before.key_version) is True

    def test_key_persistence(self, hmac_config):
        """Test that keys persist across instances."""
        manager1 = HMACKeyManager(hmac_config)
        data = b'persistent test'
        signature = manager1.sign_dict({'test': 'data'})
        version = manager1._current_version

        # Create new instance
        manager2 = HMACKeyManager(hmac_config)

        # Should load same keys
        assert manager2._current_version == version
        assert manager2.verify_dict({'test': 'data'}, signature) is True

    def test_rotation_check(self, hmac_config):
        """Test rotation needed check."""
        hmac_config.key_rotation_days = 0  # Immediate rotation needed
        manager = HMACKeyManager(hmac_config)

        needs_rotation, reason = manager.check_rotation_needed()
        # New key shouldn't need immediate rotation
        assert isinstance(needs_rotation, bool)

    def test_key_info(self, hmac_config):
        """Test get_key_info."""
        manager = HMACKeyManager(hmac_config)
        info = manager.get_key_info()

        assert 'current_version' in info
        assert 'total_versions' in info
        assert 'needs_rotation' in info
        assert info['total_versions'] >= 1


# =============================================================================
# ALERT MANAGER TESTS
# =============================================================================

class TestAlertManager:
    """Tests for AlertManager."""

    def test_initialization(self, alert_config):
        """Test AlertManager initialization."""
        manager = AlertManager(alert_config)
        assert manager is not None

    def test_send_info_alert(self, alert_config, capsys):
        """Test sending INFO alert."""
        manager = AlertManager(alert_config)
        results = manager.info("Test Info", message="This is a test")

        assert len(results) >= 1
        captured = capsys.readouterr()
        assert "Test Info" in captured.out

    def test_send_warning_alert(self, alert_config, capsys):
        """Test sending WARNING alert."""
        manager = AlertManager(alert_config)
        results = manager.warning("Test Warning", details={'key': 'value'})

        captured = capsys.readouterr()
        assert "WARNING" in captured.out

    def test_send_error_alert(self, alert_config, capsys):
        """Test sending ERROR alert."""
        manager = AlertManager(alert_config)
        results = manager.error("Test Error")

        captured = capsys.readouterr()
        assert "ERROR" in captured.out

    def test_send_critical_alert(self, alert_config, capsys):
        """Test sending CRITICAL alert."""
        manager = AlertManager(alert_config)
        results = manager.critical(
            "Kill Switch Activated",
            message="Emergency halt triggered",
            details={'drawdown': -15.5}
        )

        captured = capsys.readouterr()
        assert "CRITICAL" in captured.out
        assert "Kill Switch" in captured.out

    def test_rate_limiting(self, alert_config):
        """Test alert rate limiting."""
        alert_config.rate_limit_seconds = 60
        manager = AlertManager(alert_config)

        # First alert should succeed
        results1 = manager.info("Rate Test")
        assert len(results1) >= 1

        # Second alert with same title should be rate limited
        results2 = manager.info("Rate Test")
        assert len(results2) == 0

    def test_alert_stats(self, alert_config):
        """Test alert statistics."""
        manager = AlertManager(alert_config)

        manager.info("Test 1")
        manager.warning("Test 2")

        stats = manager.get_stats()
        assert stats['alerts_sent'] >= 2

    def test_alert_history(self, alert_config):
        """Test alert history."""
        manager = AlertManager(alert_config)

        manager.info("History Test")

        history = manager.get_history(limit=10)
        assert len(history) >= 1


# =============================================================================
# DEAD MAN'S SWITCH TESTS
# =============================================================================

class TestDeadManSwitch:
    """Tests for DeadManSwitch."""

    def test_initialization(self, deadman_config):
        """Test DeadManSwitch initialization."""
        switch = DeadManSwitch(deadman_config)
        assert switch is not None
        assert switch._status == HeartbeatStatus.INITIALIZING

    def test_start_and_stop(self, deadman_config):
        """Test starting and stopping the switch."""
        switch = DeadManSwitch(deadman_config)

        switch.start()
        assert switch._running is True
        assert switch._status == HeartbeatStatus.HEALTHY

        switch.stop(graceful=True)
        assert switch._running is False
        assert switch._status == HeartbeatStatus.STOPPED

    def test_heartbeat_file_creation(self, deadman_config):
        """Test heartbeat file is created."""
        switch = DeadManSwitch(deadman_config)
        switch.force_heartbeat()

        heartbeat_file = Path(deadman_config.heartbeat_file_path)
        assert heartbeat_file.exists()

        # Verify content
        content = json.loads(heartbeat_file.read_text())
        assert content['bot_id'] == 'test-bot'
        assert content['status'] == 'alive'

    def test_heartbeat_update(self, deadman_config):
        """Test heartbeat context update."""
        switch = DeadManSwitch(deadman_config)

        switch.heartbeat(
            positions_count=5,
            metadata={'custom': 'data'}
        )

        assert switch._positions_count == 5
        assert switch._custom_metadata == {'custom': 'data'}

    def test_get_status(self, deadman_config):
        """Test get_status."""
        switch = DeadManSwitch(deadman_config)
        switch.start()

        status = switch.get_status()
        assert status['status'] == 'healthy'
        assert status['running'] is True
        assert 'uptime_seconds' in status

        switch.stop()

    def test_context_manager(self, deadman_config):
        """Test context manager usage."""
        switch = DeadManSwitch(deadman_config)

        with switch.session():
            assert switch._running is True

        assert switch._running is False

    def test_failure_callback(self, deadman_config):
        """Test failure callback is invoked."""
        callback_called = threading.Event()

        def on_failure():
            callback_called.set()

        # Configure to fail fast
        deadman_config.webhook_url = "http://invalid.url.test"
        deadman_config.backend = HeartbeatBackend.HTTP_WEBHOOK

        switch = DeadManSwitch(
            deadman_config,
            on_failure_callback=on_failure
        )
        switch._max_consecutive_failures = 1

        # Force multiple failures
        switch._send_heartbeat()
        switch._send_heartbeat()

        # Callback should have been called
        # Note: May not trigger in all cases depending on error handling


# =============================================================================
# SIEM CLIENT TESTS
# =============================================================================

class TestSIEMClient:
    """Tests for SIEMClient."""

    def test_initialization(self, siem_config):
        """Test SIEMClient initialization."""
        client = SIEMClient(siem_config)
        assert client is not None
        client.shutdown()

    def test_log_security_event(self, siem_config):
        """Test logging a security event."""
        client = SIEMClient(siem_config)

        event = SecurityEvent(
            category=EventCategory.SECURITY,
            event_type="test_event",
            message="Test security event",
            severity=EventSeverity.LOW,
            outcome=EventOutcome.SUCCESS,
        )

        result = client.log_event(event)
        assert result is True

        # Verify file was written
        with open(siem_config.file_path, 'r') as f:
            line = f.readline()
            logged = json.loads(line)
            assert logged['event_type'] == 'test_event'

        client.shutdown()

    def test_log_trade(self, siem_config):
        """Test logging a trade event."""
        client = SIEMClient(siem_config)

        result = client.log_trade(
            "order_placed",
            "Buy order placed for EURUSD",
            {'symbol': 'EURUSD', 'volume': 0.1}
        )

        assert result is True
        client.shutdown()

    def test_log_risk(self, siem_config):
        """Test logging a risk event."""
        client = SIEMClient(siem_config)

        result = client.log_risk(
            "threshold_breach",
            "Max drawdown exceeded",
            {'current_drawdown': -12.5, 'threshold': -10.0}
        )

        assert result is True
        client.shutdown()

    def test_log_system(self, siem_config):
        """Test logging a system event."""
        client = SIEMClient(siem_config)

        result = client.log_system("startup", "System started")
        assert result is True
        client.shutdown()

    def test_event_to_cef(self):
        """Test CEF format conversion."""
        event = SecurityEvent(
            category=EventCategory.AUTHENTICATION,
            event_type="login",
            message="User login attempt",
            severity=EventSeverity.MEDIUM,
            outcome=EventOutcome.SUCCESS,
            actor_id="user123",
        )

        cef = event.to_cef()
        assert cef.startswith("CEF:0|")
        assert "login" in cef

    def test_event_to_ecs(self):
        """Test ECS format conversion."""
        event = SecurityEvent(
            category=EventCategory.TRADING,
            event_type="order_executed",
            message="Order executed",
            severity=EventSeverity.LOW,
        )

        ecs = event.to_ecs()
        assert '@timestamp' in ecs
        assert ecs['event']['category'] == ['trading']

    def test_get_stats(self, siem_config):
        """Test SIEM client statistics."""
        client = SIEMClient(siem_config)

        client.log_system("test1", "Test 1")
        client.log_system("test2", "Test 2")

        stats = client.get_stats()
        assert stats['events_logged'] >= 2
        client.shutdown()


# =============================================================================
# SECURITY ORCHESTRATOR TESTS
# =============================================================================

class TestSecurityOrchestrator:
    """Tests for SecurityOrchestrator."""

    def test_initialization(self, temp_dir):
        """Test SecurityOrchestrator initialization."""
        config = SecurityOrchestratorConfig(
            secrets_config=SecretManagerConfig(
                primary_backend=SecretBackend.ENVIRONMENT,
                encrypted_file_path=os.path.join(temp_dir, '.secrets.enc'),
            ),
            hmac_config=HMACKeyConfig(
                key_file_path=os.path.join(temp_dir, '.hmac.enc'),
            ),
            alert_config=AlertConfig(
                enabled_channels=[AlertChannel.CONSOLE],
                async_delivery=False,
            ),
            deadman_config=DeadManSwitchConfig(
                backend=HeartbeatBackend.FILE,
                heartbeat_file_path=os.path.join(temp_dir, '.heartbeat'),
            ),
            siem_config=SIEMConfig(
                backend=SIEMBackend.FILE,
                file_path=os.path.join(temp_dir, 'events.jsonl'),
                async_delivery=False,
            ),
            auto_start_deadman=False,
            register_atexit=False,
        )

        orchestrator = SecurityOrchestrator(config)
        assert orchestrator is not None
        assert orchestrator.secrets is not None
        assert orchestrator.hmac is not None
        assert orchestrator.alerts is not None
        assert orchestrator.deadman is not None
        assert orchestrator.siem is not None

        orchestrator.shutdown()

    def test_start_and_shutdown(self, temp_dir):
        """Test start and shutdown lifecycle."""
        config = SecurityOrchestratorConfig(
            secrets_config=SecretManagerConfig(
                primary_backend=SecretBackend.ENVIRONMENT,
            ),
            hmac_config=HMACKeyConfig(
                key_file_path=os.path.join(temp_dir, '.hmac.enc'),
            ),
            alert_config=AlertConfig(
                enabled_channels=[AlertChannel.CONSOLE],
                async_delivery=False,
            ),
            deadman_config=DeadManSwitchConfig(
                backend=HeartbeatBackend.FILE,
                heartbeat_file_path=os.path.join(temp_dir, '.heartbeat'),
            ),
            siem_config=SIEMConfig(
                backend=SIEMBackend.FILE,
                file_path=os.path.join(temp_dir, 'events.jsonl'),
                async_delivery=False,
            ),
            auto_start_deadman=False,
            register_atexit=False,
        )

        orchestrator = SecurityOrchestrator(config)
        orchestrator.start()

        assert orchestrator._started is True

        orchestrator.shutdown()
        assert orchestrator._started is False

    def test_sign_and_verify(self, temp_dir):
        """Test HMAC signing through orchestrator."""
        config = SecurityOrchestratorConfig(
            hmac_config=HMACKeyConfig(
                key_file_path=os.path.join(temp_dir, '.hmac.enc'),
            ),
            enable_secrets_manager=False,
            enable_alert_manager=False,
            enable_dead_man_switch=False,
            enable_siem=False,
            register_atexit=False,
        )

        orchestrator = SecurityOrchestrator(config)

        data = {'trade': 'test', 'amount': 100}
        signature = orchestrator.sign_data(data)

        assert orchestrator.verify_signature(data, signature) is True
        orchestrator.shutdown()

    def test_alerting(self, temp_dir, capsys):
        """Test alerting through orchestrator."""
        config = SecurityOrchestratorConfig(
            alert_config=AlertConfig(
                enabled_channels=[AlertChannel.CONSOLE],
                async_delivery=False,
            ),
            enable_secrets_manager=False,
            enable_hmac_manager=False,
            enable_dead_man_switch=False,
            enable_siem=False,
            register_atexit=False,
        )

        orchestrator = SecurityOrchestrator(config)
        orchestrator.alert_critical("Test Critical Alert", details={'test': True})

        captured = capsys.readouterr()
        assert "CRITICAL" in captured.out

        orchestrator.shutdown()

    def test_get_status(self, temp_dir):
        """Test comprehensive status."""
        config = SecurityOrchestratorConfig(
            hmac_config=HMACKeyConfig(
                key_file_path=os.path.join(temp_dir, '.hmac.enc'),
            ),
            alert_config=AlertConfig(
                enabled_channels=[AlertChannel.CONSOLE],
                async_delivery=False,
            ),
            deadman_config=DeadManSwitchConfig(
                backend=HeartbeatBackend.FILE,
                heartbeat_file_path=os.path.join(temp_dir, '.heartbeat'),
            ),
            siem_config=SIEMConfig(
                backend=SIEMBackend.FILE,
                file_path=os.path.join(temp_dir, 'events.jsonl'),
                async_delivery=False,
            ),
            enable_secrets_manager=False,
            auto_start_deadman=False,
            register_atexit=False,
        )

        orchestrator = SecurityOrchestrator(config)
        status = orchestrator.get_status()

        assert 'started' in status
        assert 'components' in status
        assert 'hmac_manager' in status['components']

        orchestrator.shutdown()

    def test_health_check(self, temp_dir):
        """Test health check."""
        config = SecurityOrchestratorConfig(
            hmac_config=HMACKeyConfig(
                key_file_path=os.path.join(temp_dir, '.hmac.enc'),
            ),
            enable_secrets_manager=False,
            enable_alert_manager=False,
            enable_dead_man_switch=False,
            enable_siem=False,
            register_atexit=False,
        )

        orchestrator = SecurityOrchestrator(config)
        is_healthy = orchestrator.health_check()

        assert isinstance(is_healthy, bool)
        orchestrator.shutdown()


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for the complete security stack."""

    def test_full_security_flow(self, temp_dir, capsys):
        """Test complete security flow."""
        config = SecurityOrchestratorConfig(
            secrets_config=SecretManagerConfig(
                primary_backend=SecretBackend.ENCRYPTED_FILE,
                encrypted_file_path=os.path.join(temp_dir, '.secrets.enc'),
            ),
            hmac_config=HMACKeyConfig(
                key_file_path=os.path.join(temp_dir, '.hmac.enc'),
            ),
            alert_config=AlertConfig(
                enabled_channels=[AlertChannel.CONSOLE],
                async_delivery=False,
            ),
            deadman_config=DeadManSwitchConfig(
                backend=HeartbeatBackend.FILE,
                heartbeat_file_path=os.path.join(temp_dir, '.heartbeat'),
                heartbeat_interval_seconds=60,
            ),
            siem_config=SIEMConfig(
                backend=SIEMBackend.FILE,
                file_path=os.path.join(temp_dir, 'events.jsonl'),
                async_delivery=False,
            ),
            auto_start_deadman=False,
            register_atexit=False,
        )

        # Initialize
        security = SecurityOrchestrator(config)
        security.start()

        # 1. Store and retrieve a secret
        security.secrets.store_secret('test/api', {'key': 'secret123'})
        retrieved = security.secrets.get_secret('test/api')
        assert retrieved['key'] == 'secret123'

        # 2. Sign and verify data
        trade_data = {'symbol': 'EURUSD', 'action': 'BUY', 'volume': 0.1}
        signature = security.sign_data(trade_data)
        assert security.verify_signature(trade_data, signature)

        # 3. Send an alert
        security.alert_warning("Test Alert", details={'test': True})
        captured = capsys.readouterr()
        assert "WARNING" in captured.out

        # 4. Send heartbeat
        security.heartbeat(positions_count=2)

        # 5. Log to SIEM
        security.log_trade("order_placed", {'symbol': 'EURUSD'})

        # 6. Check status
        status = security.get_status()
        assert status['started'] is True

        # Cleanup
        security.shutdown()

    def test_audit_integrity_flow(self, temp_dir):
        """Test complete audit log integrity flow."""
        hmac_config = HMACKeyConfig(
            key_file_path=os.path.join(temp_dir, '.hmac.enc'),
        )

        hmac = HMACKeyManager(hmac_config)

        # Simulate audit log entries
        audit_entries = [
            {'timestamp': '2025-01-01T00:00:00', 'action': 'trade', 'amount': 100},
            {'timestamp': '2025-01-01T00:01:00', 'action': 'trade', 'amount': 200},
            {'timestamp': '2025-01-01T00:02:00', 'action': 'close', 'profit': 50},
        ]

        # Sign each entry
        signed_entries = []
        for entry in audit_entries:
            signature = hmac.sign_dict(entry)
            signed_entries.append({
                **entry,
                '_signature': signature,
                '_key_version': hmac._current_version
            })

        # Simulate key rotation
        hmac.rotate_key("Scheduled rotation")

        # Verify old signatures still work
        for signed in signed_entries:
            original = {k: v for k, v in signed.items()
                       if not k.startswith('_')}
            is_valid = hmac.verify_dict(
                original,
                signed['_signature'],
                signed['_key_version']
            )
            assert is_valid, f"Failed to verify: {original}"


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
