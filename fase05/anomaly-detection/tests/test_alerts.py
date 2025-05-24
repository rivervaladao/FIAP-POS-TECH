import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import logging

# --- Start of sys.path modification ---
# Ensure the project root (/app) is in sys.path to allow imports from anomaly_detection
# This is crucial if running tests directly from the 'tests' directory or if Python struggles with discovery.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End of sys.path modification ---

# Now, attempt to import the modules
from anomaly_detection.utils.alerts import ConsoleAlert, EmailAlert

# Configure logging to avoid "No handler found" warnings during tests
# and to capture debug messages from ConsoleAlert
logging.basicConfig(level=logging.DEBUG)

class TestConsoleAlert(unittest.TestCase):

    @patch('anomaly_detection.utils.alerts.logging.debug') # Path to logging in the alerts module
    def test_send_alert_console(self, mock_logging_debug):
        alert = ConsoleAlert()
        message = "Test console alert"
        alert.send_alert(message)
        mock_logging_debug.assert_called_once_with(f"ALERT: {message}")

class TestEmailAlert(unittest.TestCase):

    @patch.dict(os.environ, {
        "EMAIL_FROM": "sender@example.com",
        "EMAIL_PASSWORD": "password"
    })
    @patch('anomaly_detection.utils.alerts.smtplib.SMTP_SSL')
    def test_send_alert_email_success(self, mock_smtp_ssl_class):
        mock_server = MagicMock()
        mock_smtp_ssl_class.return_value.__enter__.return_value = mock_server

        recipient_email = "recipient@example.com"
        alert = EmailAlert(recipient_email)
        message = "Test email alert"
        
        alert.send_alert(message)

        mock_smtp_ssl_class.assert_called_once_with('smtp.gmail.com', 465)
        mock_server.login.assert_called_once_with("sender@example.com", "password")
        
        self.assertTrue(mock_server.sendmail.called)
        args, _ = mock_server.sendmail.call_args
        
        self.assertEqual(args[0], "sender@example.com")
        self.assertEqual(args[1], recipient_email)
        self.assertIn("Subject: Alerta de Segurança", args[2])
        self.assertIn(f"Alerta: objeto perigoso detectado ({message})!", args[2])

    @patch.dict(os.environ, {
        "EMAIL_FROM": "sender@example.com",
        "EMAIL_PASSWORD": "password"
    })
    @patch('anomaly_detection.utils.alerts.smtplib.SMTP_SSL')
    @patch('anomaly_detection.utils.alerts.logging.error')
    def test_send_alert_email_failure(self, mock_logging_error, mock_smtp_ssl_class):
        mock_server = MagicMock()
        mock_smtp_ssl_class.return_value.__enter__.return_value = mock_server
        mock_server.login.side_effect = Exception("SMTP login failed")

        recipient_email = "recipient@example.com"
        alert = EmailAlert(recipient_email)
        message = "Test email alert failure"
        
        alert.send_alert(message)

        mock_logging_error.assert_called_with(f"Failed to send email alert to {recipient_email}: SMTP login failed")

    @patch.dict(os.environ, {}, clear=True) 
    def test_email_alert_init_missing_email_from(self):
        with self.assertRaises(ValueError) as context:
            EmailAlert("recipient@example.com")
        self.assertIn("EMAIL_FROM environment variable not set.", str(context.exception))

    @patch.dict(os.environ, {"EMAIL_FROM": "sender@example.com"}, clear=True)
    def test_email_alert_init_missing_password(self):
        with self.assertRaises(ValueError) as context:
            EmailAlert("recipient@example.com")
        self.assertIn("EMAIL_PASSWORD environment variable not set.", str(context.exception))

    @patch.dict(os.environ, {
        "EMAIL_FROM": "sender@example.com",
        "EMAIL_PASSWORD": "password"
    })
    def test_email_alert_init_success(self):
        try:
            EmailAlert("recipient@example.com")
        except ValueError:
            self.fail("EmailAlert initialization failed unexpectedly with valid environment variables.")

if __name__ == "__main__":
    unittest.main()
