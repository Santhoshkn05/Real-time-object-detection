from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import time
import cv2
import imutils

# COCO class labels
CLASSES = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
    7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
    13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
    18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
    24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
    32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
    37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
    41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
    46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl',
    52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli',
    57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair',
    63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet',
    72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'mobile phone', 76: 'keyboard',
    77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
    82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
    88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'
}
CLASS_IDS = list(CLASSES.keys())
COLORS = np.random.uniform(0, 255, size=(len(CLASS_IDS), 3))

# Load model (fixed paths with raw strings)
MODEL_PATH = r"C:\Users\ACER\Downloads\miniproject\Real time object d-t-dl-cv\real-time-object-detection\frozen_inference_graph.pb"
CONFIG_PATH = r"C:\Users\ACER\Downloads\miniproject\Real time object d-t-dl-cv\real-time-object-detection\ssd_mobilenet_v2_coco_2018_03_29.pbtxt"

print("[INFO] Loading model...")
net = cv2.dnn.readNetFromTensorflow(MODEL_PATH, CONFIG_PATH)

print("[INFO] Starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# Real-time loop
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.4:
            class_id = int(detections[0, 0, i, 1])

            if class_id in CLASSES:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                label = f"{CLASSES[class_id]}: {confidence * 100:.1f}%"

                # Map class ID to index in COLORS array
                color_index = CLASS_IDS.index(class_id)
                color = COLORS[color_index]

                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.putText(frame, label, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame
    cv2.imshow("Real-Time Object Detection", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    fps.update()

# Clean up
fps.stop()
print(f"[INFO] Elapsed time: {fps.elapsed():.2f}")
print(f"[INFO] Approx. FPS: {fps.fps():.2f}")
cv2.destroyAllWindows()
vs.stop()