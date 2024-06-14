import cv2
import numpy as np
from ultralytics import YOLO
import pyttsx3

# Initialize the YOLOv8 model (nano model for better performance)
model = YOLO('yolov8n.pt')

# Initialize camera
cap = cv2.VideoCapture(0)

# Initialize pyttsx3
engine = pyttsx3.init()

# Known constants
KNOWN_HEIGHT = 1.7  # Average height of a person in meters
FOCAL_LENGTH = None  # Will be calculated

def speak(text):
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Failed to play sound: {e}")

def estimate_focal_length(known_distance, pixel_height):
    return (pixel_height * known_distance) / KNOWN_HEIGHT

def estimate_distance(bbox_height):
    if FOCAL_LENGTH is None:
        return None
    return (KNOWN_HEIGHT * FOCAL_LENGTH) / bbox_height

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
                bbox_height = y2 - y1
                distance = estimate_distance(bbox_height)
                detected_objects.append((label, (x1, y1, x2, y2), distance))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                if distance:
                    cv2.putText(frame, f"{label} {distance:.2f}m", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame, detected_objects

# Calibrate the focal length with known distance and object size
def calibrate_focal_length(frame, known_distance):
    global FOCAL_LENGTH
    frame, detected_objects = detect_objects(frame)
    for obj, (x1, y1, x2, y2), _ in detected_objects:
        if obj == "person":  # Assuming calibration with a person
            bbox_height = y2 - y1
            FOCAL_LENGTH = estimate_focal_length(known_distance, bbox_height)
            print(f"Focal length calibrated: {FOCAL_LENGTH}")
            break
    return frame

try:
    frame_count = 0
    calibrated = False

    while True:
        ret, frame = cap.read()
        frame_count += 1

        # Only process every 5th frame to reduce load
        if frame_count % 5 != 0:
            continue

        # Resize frame to reduce computation
        frame = cv2.resize(frame, (320, 240))

        if not calibrated:
            known_distance = 2.0  # Known distance in meters
            frame = calibrate_focal_length(frame, known_distance)
            calibrated = True
            continue

        frame, detected_objects = detect_objects(frame)

        # Sort detected objects by distance
        detected_objects.sort(key=lambda x: x[2])

        for obj, bbox, distance in detected_objects:
            if distance:
                speak(f"{obj} detected at {distance:.2f} meters ahead.")

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()
