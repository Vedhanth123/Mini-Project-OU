import cv2
import numpy as np
from gtts import gTTS
import os
from playsound import playsound

# Paths to the YOLO files
weights_path = "yolo/yolov3.weights"
config_path = "yolo/yolov3.cfg"
names_path = "yolo/coco.names"

# Load YOLO
net = cv2.dnn.readNet(weights_path, config_path)
layer_names = net.getLayerNames()

# Get output layer names
try:
    layer_outputs = net.getUnconnectedOutLayers()
    if isinstance(layer_outputs, list) or len(layer_outputs.shape) == 1:
        output_layers = [layer_names[i - 1] for i in layer_outputs]
    else:
        output_layers = [layer_names[i[0] - 1] for i in layer_outputs]
except Exception as e:
    print(f"Error getting output layers: {e}")
    output_layers = []

# Load the COCO class labels
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize camera
cap = cv2.VideoCapture(0)

def speak(text):
    tts = gTTS(text=text, lang='en')
    audio_file = "output.mp3"
    tts.save(audio_file)
    playsound(audio_file)
    os.remove(audio_file)

def detect_objects(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    detected_objects = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            detected_objects.append(label)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame, detected_objects

try:
    while True:
        ret, frame = cap.read()
        frame, detected_objects = detect_objects(frame)

        if detected_objects:
            for obj in detected_objects:
                speak(f"Warning: {obj} detected ahead.")
        
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    cap.release()
    cv2.destroyAllWindows()
