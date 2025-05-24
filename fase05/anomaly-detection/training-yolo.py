import argparse
from ultralytics import YOLO

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a YOLO model.")
    parser.add_argument("dataset_yaml", help="Path to the dataset YAML file.")
    parser.add_argument("--model-name", default="yolov8n.pt",
                        help="Name of the pre-trained YOLO model to use (default: yolov8n.pt).")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs (default: 30).")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size for training (default: 640).")
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Initialize YOLO model
    model = YOLO(args.model_name)

    # Train the model
    model.train(data=args.dataset_yaml, epochs=args.epochs, imgsz=args.imgsz)

if __name__ == "__main__":
    main()
