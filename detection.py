import os
from pathlib import Path
import numpy as np
from PIL import Image
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception:
    ULTRALYTICS_AVAILABLE = False
import cv2

MODEL_PATH = Path("models") / "best.pt"

def load_model():
    """
    Load YOLO model if ultralytics is installed and best.pt exists and non-empty.
    Returns None if not available.
    """
    if ULTRALYTICS_AVAILABLE and MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        model = YOLO(str(MODEL_PATH))
        return model
    return None

def predict_with_model(model, img, conf=0.25):
    """
    Run YOLO prediction on BGR numpy image.
    Returns list of detections (dicts).
    """
    results = model.predict(img, conf=conf, imgsz=640)
    detections = []
    boxes = results[0].boxes
    names = results[0].names
    for i, box in enumerate(boxes.data.tolist()):
        x1, y1, x2, y2, score, cls = box
        detections.append({
            "id": i,
            "class": names[int(cls)],
            "confidence": float(score),
            "x": int(x1), "y": int(y1), "w": int(x2 - x1), "h": int(y2 - y1)
        })
    return detections

class SyntheticDetector:
    """
    Simple contour-based detector used as demo fallback.
    """
    def __init__(self):
        pass

    def detect(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets = []
        for i, c in enumerate(cnts):
            x, y, w, h = cv2.boundingRect(c)
            if w * h < 100:
                continue
            dets.append({"id": i, "class": "particle", "confidence": 0.8, "x": int(x), "y": int(y), "w": int(w), "h": int(h)})
        return dets

    def draw(self, img, detections):
        for d in detections:
            x, y, w, h = d["x"], d["y"], d["w"], d["h"]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return img
