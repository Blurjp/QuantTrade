"""
Tests for notification system.
"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

from notifications.notification_manager import NotificationConfig, NotificationManager
from notifications.signal_monitor import SignalMonitor


class TestNotificationConfig:
    """Tests for NotificationConfig class."""

    def test_default_initialization(self):
        """Test default configuration values."""
        with patch.dict(os.environ, {}, clear=True):
            config = NotificationConfig()
            
            assert config.email_enabled is False
            assert config.smtp_host == "smtp.gmail.com"
            assert config.smtp_port == 587
            assert config.email_to == [""]
            
            assert config.sms_enabled is False
            assert config.twilio_phone_to == [""]
            
            assert config.discord_enabled is False
            
            assert config.min_confidence == 70.0
            assert config.min_impact_score == 50.0
            assert config.quiet_hours_start == 23
            assert config.quiet_hours_end == 7
            assert config.notification_timezone == "America/New_York"

    def test_custom_configuration(self):
        """Test custom configuration from environment variables."""
        env = {
            "EMAIL_ENABLED": "true",
            "SMTP_HOST": "smtp.custom.com",
            "SMTP_PORT": "465",
            "SMTP_USERNAME": "user@example.com",
            "SMTP_PASSWORD": "password123",
            "EMAIL_FROM": "from@example.com",
            "EMAIL_TO": "to1@example.com,to2@example.com",
            
            "SMS_ENABLED": "true",
            "TWILIO_ACCOUNT_SID": "AC123",
            "TWILIO_AUTH_TOKEN": "token123",
            "TWILIO_PHONE_FROM": "+1234567890",
            "TWILIO_PHONE_TO": "+0987654321,+1111111111",
            
            "DISCORD_ENABLED": "true",
            "DISCORD_WEBHOOK_URL": "https://discord.com/api/webhooks/123",
            
            "NOTIFICATION_MIN_CONFIDENCE": "85.0",
            "NOTIFICATION_MIN_IMPACT": "60.0",
            "QUIET_HOURS_START": "22",
            "QUIET_HOURS_END": "6",
            "NOTIFICATION_TIMEZONE": "America/Los_Angeles",
        }
        
        with patch.dict(os.environ, env, clear=True):
            config = NotificationConfig()
            
            assert config.email_enabled is True
            assert config.smtp_host == "smtp.custom.com"
            assert config.smtp_port == 465
            assert config.smtp_username == "user@example.com"
            assert config.smtp_password == "password123"
            assert config.email_from == "from@example.com"
            assert config.email_to == ["to1@example.com", "to2@example.com"]
            
            assert config.sms_enabled is True
            assert config.twilio_account_sid == "AC123"
            assert config.twilio_auth_token == "token123"
            assert config.twilio_phone_from == "+1234567890"
            assert config.twilio_phone_to == ["+0987654321", "+1111111111"]
            
            assert config.discord_enabled is True
            assert config.discord_webhook_url == "https://discord.com/api/webhooks/123"
            
            assert config.min_confidence == 85.0
            assert config.min_impact_score == 60.0
            assert config.quiet_hours_start == 22
            assert config.quiet_hours_end == 6
            assert config.notification_timezone == "America/Los_Angeles"

    def test_is_quiet_hours_daytime(self):
        """Test quiet hours check during daytime."""
        with patch.dict(os.environ, {
            "QUIET_HOURS_START": "23",
            "QUIET_HOURS_END": "7",
            "NOTIFICATION_TIMEZONE": "America/New_York"
        }, clear=True):
            config = NotificationConfig()
            
            with patch("notifications.notification_manager.datetime") as mock_dt:
                mock_dt.now.return_value.hour = 14
                mock_dt.now.return_value = MagicMock(hour=14)
                from zoneinfo import ZoneInfo
                mock_dt.now.side_effect = lambda tz: MagicMock(hour=14)
                
                assert config.is_quiet_hours() is False

    def test_is_quiet_hours_nighttime(self):
        """Test quiet hours check during nighttime."""
        with patch.dict(os.environ, {
            "QUIET_HOURS_START": "23",
            "QUIET_HOURS_END": "7",
            "NOTIFICATION_TIMEZONE": "America/New_York"
        }, clear=True):
            config = NotificationConfig()
            
            with patch("notifications.notification_manager.datetime") as mock_dt:
                mock_dt.now.side_effect = lambda tz: MagicMock(hour=2)
                assert config.is_quiet_hours() is True
                
                mock_dt.now.side_effect = lambda tz: MagicMock(hour=23)
                assert config.is_quiet_hours() is True
                
                mock_dt.now.side_effect = lambda tz: MagicMock(hour=6)
                assert config.is_quiet_hours() is True

    def test_is_quiet_hours_spanning_midnight(self):
        """Test quiet hours that span midnight."""
        with patch.dict(os.environ, {
            "QUIET_HOURS_START": "23",
            "QUIET_HOURS_END": "7"
        }, clear=True):
            config = NotificationConfig()
            
            with patch("notifications.notification_manager.datetime") as mock_dt:
                mock_dt.now.side_effect = lambda tz: MagicMock(hour=0)
                assert config.is_quiet_hours() is True
                
                mock_dt.now.side_effect = lambda tz: MagicMock(hour=3)
                assert config.is_quiet_hours() is True
                
                mock_dt.now.side_effect = lambda tz: MagicMock(hour=12)
                assert config.is_quiet_hours() is False

    def test_is_quiet_hours_same_day(self):
        """Test quiet hours within same day."""
        with patch.dict(os.environ, {
            "QUIET_HOURS_START": "9",
            "QUIET_HOURS_END": "17"
        }, clear=True):
            config = NotificationConfig()
            
            with patch("notifications.notification_manager.datetime") as mock_dt:
                mock_dt.now.side_effect = lambda tz: MagicMock(hour=12)
                assert config.is_quiet_hours() is True
                
                mock_dt.now.side_effect = lambda tz: MagicMock(hour=8)
                assert config.is_quiet_hours() is False
                
                mock_dt.now.side_effect = lambda tz: MagicMock(hour=18)
                assert config.is_quiet_hours() is False


class TestNotificationManager:
    """Tests for NotificationManager class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        with patch.dict(os.environ, {
            "EMAIL_ENABLED": "false",
            "SMS_ENABLED": "false",
            "DISCORD_ENABLED": "false",
            "NOTIFICATION_MIN_CONFIDENCE": "70.0",
            "NOTIFICATION_MIN_IMPACT": "50.0",
        }, clear=True):
            return NotificationConfig()

    @pytest.fixture
    def manager(self, config, temp_dir):
        """Create a notification manager with temp directory."""
        with patch("notifications.notification_manager.Path") as mock_path:
            mock_path.return_value = Path(temp_dir) / "notification_history.json"
            manager = NotificationManager(config)
            manager.notification_history_file = Path(temp_dir) / "notification_history.json"
            return manager

    def test_initialization(self, manager):
        """Test manager initialization."""
        assert manager.config is not None
        assert manager.history == []

    def test_load_history_existing_file(self, temp_dir):
        """Test loading history from existing file."""
        history_file = Path(temp_dir) / "notification_history.json"
        existing_history = [
            {"signal_id": "test_123_2024-01-01", "timestamp": "2024-01-01T10:00:00"}
        ]
        with open(history_file, 'w') as f:
            json.dump(existing_history, f)
        
        with patch.dict(os.environ, {}, clear=True):
            config = NotificationConfig()
            manager = NotificationManager(config)
            manager.notification_history_file = history_file
            manager.history = manager._load_history()
            
            assert len(manager.history) == 1
            assert manager.history[0]["signal_id"] == "test_123_2024-01-01"

    def test_load_history_missing_file(self, temp_dir):
        """Test loading history when file doesn't exist."""
        with patch.dict(os.environ, {}, clear=True):
            config = NotificationConfig()
            manager = NotificationManager(config)
            manager.notification_history_file = Path(temp_dir) / "nonexistent.json"
            manager.history = manager._load_history()
            
            assert manager.history == []

    def test_save_history(self, temp_dir):
        """Test saving history to file."""
        history_file = Path(temp_dir) / "notification_history.json"
        
        with patch.dict(os.environ, {}, clear=True):
            config = NotificationConfig()
            manager = NotificationManager(config)
            manager.notification_history_file = history_file
            manager.history = [{"test": "data"}]
            manager._save_history()
            
            with open(history_file, 'r') as f:
                saved = json.load(f)
            
            assert saved == [{"test": "data"}]

    def test_save_history_keeps_last_100(self, temp_dir):
        """Test that history is limited to last 100 entries."""
        history_file = Path(temp_dir) / "notification_history.json"
        
        with patch.dict(os.environ, {}, clear=True):
            config = NotificationConfig()
            manager = NotificationManager(config)
            manager.notification_history_file = history_file
            manager.history = [{"id": i} for i in range(150)]
            manager._save_history()
            
            with open(history_file, 'r') as f:
                saved = json.load(f)
            
            assert len(saved) == 100
            assert saved[0]["id"] == 50
            assert saved[-1]["id"] == 149

    def test_should_notify_below_confidence(self, manager):
        """Test signal below confidence threshold."""
        signal = {
            "signal_type": "test",
            "region_id": "test_region",
            "date": "2024-01-01",
            "confidence": 50.0,
            "direction": "long"
        }
        
        assert manager.should_notify(signal) is False

    def test_should_notify_below_impact(self, manager):
        """Test signal below impact threshold."""
        signal = {
            "signal_type": "test",
            "region_id": "test_region",
            "date": "2024-01-01",
            "confidence": 80.0,
            "impact_score": 30.0,
            "direction": "long"
        }
        
        assert manager.should_notify(signal) is False

    def test_should_notify_already_notified(self, manager):
        """Test signal that was already notified."""
        signal = {
            "signal_type": "test",
            "region_id": "test_region",
            "date": "2024-01-01",
            "confidence": 80.0,
            "direction": "long"
        }
        
        manager.history = [
            {"signal_id": "test_test_region_2024-01-01"}
        ]
        
        assert manager.should_notify(signal) is False

    def test_should_notify_quiet_hours(self, manager):
        """Test signal during quiet hours."""
        signal = {
            "signal_type": "test",
            "region_id": "test_region",
            "date": "2024-01-01",
            "confidence": 80.0,
            "direction": "long"
        }
        
        with patch.object(manager.config, 'is_quiet_hours', return_value=True):
            assert manager.should_notify(signal) is False

    def test_should_notify_passes(self, manager):
        """Test signal that passes all checks."""
        signal = {
            "signal_type": "test",
            "region_id": "test_region",
            "date": "2024-01-01",
            "confidence": 80.0,
            "direction": "long"
        }
        
        with patch.object(manager.config, 'is_quiet_hours', return_value=False):
            assert manager.should_notify(signal) is True

    def test_send_email_disabled(self, manager):
        """Test sending email when disabled."""
        manager.config.email_enabled = False
        result = manager.send_email("Test Subject", "Test Body")
        assert result is False

    def test_send_email_no_recipients(self, manager):
        """Test sending email with no recipients."""
        manager.config.email_enabled = True
        manager.config.email_to = []
        result = manager.send_email("Test Subject", "Test Body")
        assert result is False

    @patch("notifications.notification_manager.smtplib.SMTP_SSL")
    def test_send_email_ssl_success(self, mock_smtp_ssl, manager):
        """Test successful email via SSL."""
        manager.config.email_enabled = True
        manager.config.email_to = ["test@example.com"]
        manager.config.smtp_host = "smtp.test.com"
        manager.config.smtp_port = 465
        manager.config.smtp_username = "user"
        manager.config.smtp_password = "pass"
        manager.config.email_from = "from@test.com"
        
        mock_server = MagicMock()
        mock_smtp_ssl.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_ssl.return_value.__exit__ = MagicMock(return_value=False)
        
        result = manager.send_email("Test Subject", "Test Body")
        
        assert result is True
        mock_server.login.assert_called_once_with("user", "pass")
        mock_server.sendmail.assert_called_once()

    @patch("notifications.notification_manager.smtplib.SMTP_SSL")
    @patch("notifications.notification_manager.smtplib.SMTP")
    def test_send_email_fallback_to_tls(self, mock_smtp, mock_smtp_ssl, manager):
        """Test email fallback to TLS when SSL fails."""
        manager.config.email_enabled = True
        manager.config.email_to = ["test@example.com"]
        manager.config.smtp_host = "smtp.test.com"
        manager.config.smtp_username = "user"
        manager.config.smtp_password = "pass"
        manager.config.email_from = "from@test.com"
        
        mock_smtp_ssl.side_effect = Exception("SSL failed")
        
        mock_tls_server = MagicMock()
        mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_tls_server)
        mock_smtp.return_value.__exit__ = MagicMock(return_value=False)
        
        result = manager.send_email("Test Subject", "Test Body")
        
        assert result is True
        mock_tls_server.starttls.assert_called_once()
        mock_tls_server.login.assert_called_once()

    def test_send_sms_disabled(self, manager):
        """Test sending SMS when disabled."""
        manager.config.sms_enabled = False
        result = manager.send_sms("Test Message")
        assert result is False

    def test_send_sms_no_recipients(self, manager):
        """Test sending SMS with no recipients."""
        manager.config.sms_enabled = True
        manager.config.twilio_phone_to = []
        result = manager.send_sms("Test Message")
        assert result is False

    def test_send_sms_success(self, manager):
        """Test successful SMS sending."""
        manager.config.sms_enabled = True
        manager.config.twilio_phone_to = ["+1234567890"]
        manager.config.twilio_account_sid = "AC123"
        manager.config.twilio_auth_token = "token"
        manager.config.twilio_phone_from = "+0987654321"
        
        mock_client = MagicMock()
        
        with patch.dict("sys.modules", {"twilio.rest": MagicMock(Client=MagicMock(return_value=mock_client))}):
            result = manager.send_sms("Test Message")
        
        assert result is True
        mock_client.messages.create.assert_called_once()

    def test_send_sms_import_error(self, manager):
        """Test SMS when Twilio not installed."""
        manager.config.sms_enabled = True
        manager.config.twilio_phone_to = ["+1234567890"]
        
        with patch.dict("sys.modules", {"twilio.rest": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                result = manager.send_sms("Test Message")
                assert result is False

    def test_send_discord_disabled(self, manager):
        """Test sending Discord when disabled."""
        manager.config.discord_enabled = False
        result = manager.send_discord("Test Message")
        assert result is False

    def test_send_discord_no_webhook(self, manager):
        """Test sending Discord without webhook URL."""
        manager.config.discord_enabled = True
        manager.config.discord_webhook_url = ""
        result = manager.send_discord("Test Message")
        assert result is False

    def test_send_discord_success(self, manager):
        """Test successful Discord notification."""
        manager.config.discord_enabled = True
        manager.config.discord_webhook_url = "https://discord.com/webhook"
        
        mock_response = MagicMock()
        mock_response.status_code = 204
        
        with patch("requests.post", return_value=mock_response):
            result = manager.send_discord("Test Message", {"title": "Test"})
        
        assert result is True

    def test_send_discord_failure(self, manager):
        """Test failed Discord notification."""
        manager.config.discord_enabled = True
        manager.config.discord_webhook_url = "https://discord.com/webhook"
        
        mock_response = MagicMock()
        mock_response.status_code = 400
        
        with patch("requests.post", return_value=mock_response):
            result = manager.send_discord("Test Message")
        
        assert result is False

    def test_notify_signal_does_not_meet_criteria(self, manager):
        """Test notifying signal that doesn't meet criteria."""
        signal = {
            "signal_type": "test",
            "confidence": 50.0,
            "direction": "long"
        }
        
        result = manager.notify_signal(signal)
        assert result == {}

    def test_notify_signal_success(self, manager, temp_dir):
        """Test successful signal notification."""
        manager.config.email_enabled = True
        manager.config.email_to = ["test@example.com"]
        manager.notification_history_file = Path(temp_dir) / "history.json"
        
        signal = {
            "signal_type": "test",
            "region_id": "region1",
            "region_name": "Test Region",
            "date": "2024-01-01",
            "confidence": 80.0,
            "direction": "long",
            "rationale": "Test rationale",
            "instruments": ["CORN", "SOYB"]
        }
        
        with patch.object(manager, 'send_email', return_value=True):
            with patch.object(manager.config, 'is_quiet_hours', return_value=False):
                result = manager.notify_signal(signal)
        
        assert "email" in result
        assert len(manager.history) == 1


class TestSignalMonitor:
    """Tests for SignalMonitor class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def monitor(self, temp_dir):
        """Create a signal monitor with temp directory."""
        with patch.dict(os.environ, {
            "NOTIFICATION_MIN_CONFIDENCE": "70.0",
            "NOTIFICATION_MIN_IMPACT": "50.0",
        }, clear=True):
            monitor = SignalMonitor(
                history_file=str(Path(temp_dir) / "state.json")
            )
            return monitor

    def test_initialization(self, monitor):
        """Test monitor initialization."""
        assert monitor.notification_manager is not None
        assert monitor.notified_signals == {"notified": [], "last_check": None}

    def test_get_signal_key(self, monitor):
        """Test signal key generation."""
        signal = {
            "signal_type": "precipitation",
            "region_id": "usa_midwest",
            "date": "2024-01-15"
        }
        
        key = monitor._get_signal_key(signal)
        assert key == "precipitation_usa_midwest_2024-01-15"

    def test_get_signal_key_with_facility(self, monitor):
        """Test signal key with facility_id instead of region_id."""
        signal = {
            "signal_type": "thermal",
            "facility_id": "plant_123",
            "date": "2024-01-15"
        }
        
        key = monitor._get_signal_key(signal)
        assert key == "thermal_plant_123_2024-01-15"

    def test_is_actionable_neutral_direction(self, monitor):
        """Test signal with neutral direction is not actionable."""
        signal = {
            "direction": "neutral",
            "confidence": 80.0
        }
        
        assert monitor.is_actionable(signal) is False

    def test_is_actionable_low_confidence(self, monitor):
        """Test signal with low confidence is not actionable."""
        signal = {
            "direction": "long",
            "confidence": 50.0
        }
        
        assert monitor.is_actionable(signal) is False

    def test_is_actionable_low_impact(self, monitor):
        """Test signal with low impact is not actionable."""
        signal = {
            "direction": "long",
            "confidence": 80.0,
            "impact_score": 30.0
        }
        
        assert monitor.is_actionable(signal) is False

    def test_is_actionable_already_notified(self, monitor):
        """Test already notified signal is not actionable."""
        signal = {
            "signal_type": "test",
            "region_id": "region1",
            "date": "2024-01-01",
            "direction": "long",
            "confidence": 80.0
        }
        
        monitor.notified_signals["notified"].append("test_region1_2024-01-01")
        
        assert monitor.is_actionable(signal) is False

    def test_is_actionable_passes(self, monitor):
        """Test actionable signal passes all checks."""
        signal = {
            "signal_type": "test",
            "region_id": "region1",
            "date": "2024-01-01",
            "direction": "long",
            "confidence": 80.0
        }
        
        assert monitor.is_actionable(signal) is True

    def test_check_and_notify_no_signals(self, monitor):
        """Test check with no signals."""
        result = monitor.check_and_notify([])
        
        assert result["total_signals"] == 0
        assert result["actionable_signals"] == 0
        assert result["notified"] == 0

    def test_check_and_notify_with_signals(self, monitor):
        """Test check with actionable signals."""
        signals = [
            {
                "signal_type": "test",
                "region_id": "region1",
                "date": "2024-01-01",
                "direction": "long",
                "confidence": 80.0,
                "rationale": "Test"
            }
        ]
        
        with patch.object(monitor.notification_manager, 'notify_signal', return_value={"email": True}):
            result = monitor.check_and_notify(signals)
        
        assert result["total_signals"] == 1
        assert result["actionable_signals"] == 1
        assert result["notified"] == 1
        assert len(result["notifications"]) == 1

    def test_check_and_notify_keeps_last_1000(self, monitor):
        """Test that notified signals are limited to 1000."""
        for i in range(1100):
            monitor.notified_signals["notified"].append(f"signal_{i}")
        
        signals = [
            {
                "signal_type": "test",
                "region_id": "region1",
                "date": "2024-01-01",
                "direction": "long",
                "confidence": 80.0,
                "rationale": "Test"
            }
        ]
        
        with patch.object(monitor.notification_manager, 'notify_signal', return_value={"email": True}):
            monitor.check_and_notify(signals)
        
        assert len(monitor.notified_signals["notified"]) == 1000

    def test_state_persistence(self, monitor, temp_dir):
        """Test that state is persisted to file."""
        signals = [
            {
                "signal_type": "test",
                "region_id": "region1",
                "date": "2024-01-01",
                "direction": "long",
                "confidence": 80.0,
                "rationale": "Test"
            }
        ]
        
        with patch.object(monitor.notification_manager, 'notify_signal', return_value={"email": True}):
            monitor.check_and_notify(signals)
        
        state_file = Path(temp_dir) / "state.json"
        with open(state_file, 'r') as f:
            state = json.load(f)
        
        assert "test_region1_2024-01-01" in state["notified"]
        assert state["last_check"] is not None
