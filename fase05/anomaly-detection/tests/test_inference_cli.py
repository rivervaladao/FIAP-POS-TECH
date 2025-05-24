import unittest
import sys
import os
from unittest.mock import patch

# --- Start of sys.path modification ---
# Ensure the project root (/app) is in sys.path to allow imports from anomaly_detection
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End of sys.path modification ---

# Assuming parse_arguments is in anomaly_detection.inference
# We will need to ensure it's importable.
from anomaly_detection.inference import parse_arguments, DEFAULT_MODEL_PATH, DEFAULT_THRESHOLDS
import json # For comparing threshold defaults

class TestInferenceCLI(unittest.TestCase):

    def run_parser(self, args_list):
        """Helper function to run the parser with a list of arguments."""
        with patch.object(sys, 'argv', ['inference.py'] + args_list):
            return parse_arguments()

    def test_only_required_file_path(self):
        args = self.run_parser(['test.jpg'])
        self.assertEqual(args.file_path, 'test.jpg')
        self.assertEqual(args.alert_type, 'console') # Default
        self.assertIsNone(args.recipient_email) # Default
        self.assertEqual(args.model_path, DEFAULT_MODEL_PATH) # Default
        self.assertEqual(args.thresholds, DEFAULT_THRESHOLDS) # Default

    def test_file_path_and_console_alert(self):
        args = self.run_parser(['test.mp4', '--alert-type', 'console'])
        self.assertEqual(args.file_path, 'test.mp4')
        self.assertEqual(args.alert_type, 'console')

    def test_file_path_and_email_alert(self):
        recipient = 'test@example.com'
        args = self.run_parser(['test.jpg', '--alert-type', 'email', '--recipient-email', recipient])
        self.assertEqual(args.file_path, 'test.jpg')
        self.assertEqual(args.alert_type, 'email')
        self.assertEqual(args.recipient_email, recipient)

    def test_file_path_and_model_path(self):
        model_p = '/path/to/custom/model.pt'
        args = self.run_parser(['test.png', '--model-path', model_p])
        self.assertEqual(args.file_path, 'test.png')
        self.assertEqual(args.model_path, model_p)

    def test_file_path_and_thresholds(self):
        threshold_str = '{"obj1": 0.7, "obj2": 0.8}'
        expected_thresholds = json.loads(threshold_str)
        args = self.run_parser(['test.avi', '--thresholds', threshold_str])
        self.assertEqual(args.file_path, 'test.avi')
        self.assertEqual(args.thresholds, expected_thresholds)

    def test_all_arguments_provided(self):
        file_p = 'video.mov'
        alert_t = 'email'
        recipient = 'notify@domain.com'
        model_p = 'another_model.pt'
        threshold_str = '{"knife": 0.6, "scissors": 0.65}'
        expected_thresholds = json.loads(threshold_str)

        args = self.run_parser([
            file_p,
            '--alert-type', alert_t,
            '--recipient-email', recipient,
            '--model-path', model_p,
            '--thresholds', threshold_str
        ])
        self.assertEqual(args.file_path, file_p)
        self.assertEqual(args.alert_type, alert_t)
        self.assertEqual(args.recipient_email, recipient)
        self.assertEqual(args.model_path, model_p)
        self.assertEqual(args.thresholds, expected_thresholds)

    def test_missing_file_path(self):
        # argparse raises SystemExit for missing required arguments
        with self.assertRaises(SystemExit):
            self.run_parser([]) # No arguments

        with self.assertRaises(SystemExit):
            self.run_parser(['--alert-type', 'console']) # Only optional

    def test_email_alert_missing_recipient(self):
        # argparse raises SystemExit if a dependent required argument is missing
        # (this is handled by parser.error in the main script)
        with self.assertRaises(SystemExit):
            self.run_parser(['test.jpg', '--alert-type', 'email'])

    def test_invalid_threshold_json(self):
        with self.assertRaises(SystemExit):
            self.run_parser(['test.jpg', '--thresholds', 'not_json'])
            
    def test_invalid_alert_type(self):
        with self.assertRaises(SystemExit):
            self.run_parser(['test.jpg', '--alert-type', 'sms'])


if __name__ == '__main__':
    unittest.main()
