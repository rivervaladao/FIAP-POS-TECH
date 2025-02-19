import cv2
import numpy as np
from deepface import DeepFace

# Load video
video_path = "./data/video_input.mp4"
output_path = "./data/video_output.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    try:
        # Detect emotions
        results = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
        
        for result in results:
            x, y, w, h = result['region']['x'], result['region']['y'], result['region']['w'], result['region']['h']
            emotion = result['dominant_emotion']
            
            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    except Exception as e:
        print(f"Error processing frame: {e}")
    
    # Write frame to output
    out.write(frame)
    
    # Display (optional)
    cv2.imshow("Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
# cv2.destroyAllWindows()

print(f"Emotion detection video saved at {output_path}")
