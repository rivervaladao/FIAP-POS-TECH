import cv2
from deepface import DeepFace
import numpy as np
from tqdm import tqdm

def detect_emotions(video_path, outpath_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(outpath_path, fourcc, fps, (width, height))

    for _ in tqdm(range(total_frames), desc='Processing Video'):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale (optional, helps with face detection)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], detector_backend='retinaface', enforce_detection=False)
            print(result)  # Debugging output

            for face in result:
                if 'region' in face:
                    x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']

                    # Draw rectangle and emotion label
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    dominant_emotion = face.get('dominant_emotion', 'unknown')
                    cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        
        except Exception as e:
            print(f"Error processing frame: {e}")

        out.write(frame)

    cap.release()
    out.release()

    try:
        cv2.destroyAllWindows()
    except:
        pass

# Paths
input_video_path_ = "./data/video.mp4"
output_video_path_ = "./data/output_video.mp4"

# Run emotion detection
detect_emotions(input_video_path_, output_video_path_)
