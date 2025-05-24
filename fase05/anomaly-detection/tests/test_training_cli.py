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

# Assuming parse_arguments is in anomaly_detection.training_yolo
from anomaly_detection.training_yolo import parse_arguments

class TestTrainingCLI(unittest.TestCase):

    def run_parser(self, args_list):
        """Helper function to run the parser with a list of arguments."""
        with patch.object(sys, 'argv', ['training-yolo.py'] + args_list):
            return parse_arguments()

    def test_only_required_dataset_yaml(self):
        args = self.run_parser(['data.yaml'])
        self.assertEqual(args.dataset_yaml, 'data.yaml')
        self.assertEqual(args.model_name, 'yolov8n.pt')  # Default
        self.assertEqual(args.epochs, 30)  # Default
        self.assertEqual(args.imgsz, 640)  # Default

    def test_dataset_yaml_and_model_name(self):
        model_name = 'yolov8s.pt'
        args = self.run_parser(['data.yaml', '--model-name', model_name])
        self.assertEqual(args.dataset_yaml, 'data.yaml')
        self.assertEqual(args.model_name, model_name)
        self.assertEqual(args.epochs, 30)  # Default
        self.assertEqual(args.imgsz, 640)  # Default

    def test_dataset_yaml_and_epochs(self):
        epochs = 50
        args = self.run_parser(['data.yaml', '--epochs', str(epochs)])
        self.assertEqual(args.dataset_yaml, 'data.yaml')
        self.assertEqual(args.model_name, 'yolov8n.pt')  # Default
        self.assertEqual(args.epochs, epochs)
        self.assertEqual(args.imgsz, 640)  # Default

    def test_dataset_yaml_and_imgsz(self):
        imgsz = 1280
        args = self.run_parser(['data.yaml', '--imgsz', str(imgsz)])
        self.assertEqual(args.dataset_yaml, 'data.yaml')
        self.assertEqual(args.model_name, 'yolov8n.pt')  # Default
        self.assertEqual(args.epochs, 30)  # Default
        self.assertEqual(args.imgsz, imgsz)

    def test_all_arguments_provided(self):
        dataset = 'custom_data.yaml'
        model = 'yolov8m.pt'
        epochs_val = 100
        imgsz_val = 320
        args = self.run_parser([
            dataset,
            '--model-name', model,
            '--epochs', str(epochs_val),
            '--imgsz', str(imgsz_val)
        ])
        self.assertEqual(args.dataset_yaml, dataset)
        self.assertEqual(args.model_name, model)
        self.assertEqual(args.epochs, epochs_val)
        self.assertEqual(args.imgsz, imgsz_val)

    def test_missing_dataset_yaml(self):
        # argparse raises SystemExit for missing required arguments
        with self.assertRaises(SystemExit):
            self.run_parser([])  # No arguments

        with self.assertRaises(SystemExit):
            self.run_parser(['--model-name', 'yolov8n.pt']) # Only optional

    def test_invalid_epochs_value(self):
        # argparse raises SystemExit for type errors
        with self.assertRaises(SystemExit):
            self.run_parser(['data.yaml', '--epochs', 'not_an_integer'])

    def test_invalid_imgsz_value(self):
        # argparse raises SystemExit for type errors
        with self.assertRaises(SystemExit):
            self.run_parser(['data.yaml', '--imgsz', 'not_an_integer'])

if __name__ == '__main__':
    unittest.main()
