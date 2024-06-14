import cv2
import numpy as np
import pyttsx3
from ultralytics import YOLO

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # You can change to 'yolov8s.pt', 'yolov8m.pt', etc.

# Initialize camera
cap = cv2.VideoCapture(0)

def speak(text):
    engine.say(text)
    engine.runAndWait()

def detect_objects(frame):
    results = model(frame)
    detected_objects = []

    for result in results:
        boxes = result.boxes.xyxy.numpy()  # Bounding box coordinates
        scores = result.boxes.conf.numpy()  # Confidence scores
        class_ids = result.boxes.cls.numpy()  # Class IDs

        for i in range(len(boxes)):
            if scores[i] > 0.5:
                x1, y1, x2, y2 = boxes[i]
                label = result.names[int(class_ids[i])]
                detected_objects.append(label)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame, detected_objects

try:
    while True:
        ret, frame = cap.read()
        frame, detected_objects = detect_objects(frame)

        if detected_objects:
            for obj in detected_objects:
                speak(f"Warning: {obj} detected ahead!")
        
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    cap.release()
    cv2.destroyAllWindows()
    engine.stop()
