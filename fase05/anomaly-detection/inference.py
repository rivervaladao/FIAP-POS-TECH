import logging
import sys
import os
import cv2
from ultralytics import YOLO
import json
import argparse
from anomaly_detection.utils.alerts import Alert, ConsoleAlert, EmailAlert

logger = logging.getLogger(__name__)
logging.basicConfig(filename="logs.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Default values
DEFAULT_MODEL_PATH = "/home/river/workspace/experiment/fiap/FIAP-POS-TECH/fase05/datasets/runs/detect/train/weights/best.pt"
DEFAULT_THRESHOLDS = {"knife": 0.55, "scissors": 0.55}

def process_frame(frame, model, alert_handler: Alert, class_thresholds, timestamp: str = ""):
    results = model(frame)
    boxes = results[0].boxes

    for box in boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        conf = float(box.conf[0])
        threshold = class_thresholds.get(label, 0.6)

        if conf > threshold:
            message = f"Detectado: {label} com confiança {conf:.2f}"
            if timestamp:
                message += f" em {timestamp}"
            logger.debug(message)
            alert_handler.send_alert(message)

    img_bgr = cv2.cvtColor(results[0].plot(), cv2.COLOR_RGB2BGR)
    return img_bgr

def process_video(video_path: str, model, alert_handler: Alert, class_thresholds):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Erro ao abrir vídeo: {video_path}")
        return

    cv2.namedWindow("Detecção", cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        current_time_sec = current_time_ms / 1000
        timestamp = f"{int(current_time_sec // 3600):02}:{int((current_time_sec % 3600) // 60):02}:{int(current_time_sec % 60):02}"

        processed_frame = process_frame(frame, model, alert_handler, class_thresholds, timestamp)
        cv2.imshow("Detecção", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

def process_image(image_path: str, model, alert_handler: Alert, class_thresholds):
    frame = cv2.imread(image_path)
    if frame is None:
        logger.error(f"Erro ao carregar imagem: {image_path}")
        return
    processed_frame = process_frame(frame, model, alert_handler, class_thresholds)
    cv2.namedWindow("Detecção", cv2.WINDOW_NORMAL)
    cv2.imshow("Detecção", processed_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Detect anomalies in images or videos.")
    parser.add_argument("file_path", help="Path to the input video or image file.")
    parser.add_argument("--alert-type", default="console", choices=["console", "email"],
                        help="Type of alert to use (default: console).")
    parser.add_argument("--recipient-email", help="Recipient email address (required if alert_type is 'email').")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH,
                        help=f"Path to the YOLO model file (default: {DEFAULT_MODEL_PATH}).")
    parser.add_argument("--thresholds", default=json.dumps(DEFAULT_THRESHOLDS),
                        help=f"JSON string for class confidence thresholds (default: '{json.dumps(DEFAULT_THRESHOLDS)}').")
    
    args = parser.parse_args()

    if args.alert_type == "email" and not args.recipient_email:
        parser.error("--recipient-email is required when --alert-type is 'email'")

    try:
        args.thresholds = json.loads(args.thresholds)
    except json.JSONDecodeError:
        parser.error("Invalid JSON string for --thresholds.")
        
    return args

def main():
    args = parse_arguments()

    if args.alert_type == "email":
        alert_system = EmailAlert(args.recipient_email)
    else:
        alert_system = ConsoleAlert()
    
    model = YOLO(args.model_path)

    if not os.path.exists(args.file_path):
        logger.error(f"Arquivo não encontrado: {args.file_path}")
        sys.exit(1)

    ext = os.path.splitext(args.file_path)[1].lower()
    if ext in [".mp4", ".avi", ".mov", ".mkv"]:
        process_video(args.file_path, model, alert_system, args.thresholds)
    elif ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        process_image(args.file_path, model, alert_system, args.thresholds)
    else:
        logger.error("Formato de arquivo não suportado.")

if __name__ == "__main__":
    main()