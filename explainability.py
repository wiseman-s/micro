import numpy as np
import cv2

def simulated_heatmap(img, detections):
    """
    Create a simulated activation heatmap from detected boxes.
    For real models, replace with Grad-CAM or similar.
    """
    heat = img.copy().astype("float32")
    h, w = heat.shape[:2]
    mask = np.zeros((h, w), dtype="float32")
    for d in detections:
        x, y, ww, hh = d["x"], d["y"], d["w"], d["h"]
        # bounds safety
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + ww), min(h, y + hh)
        mask[y1:y2, x1:x2] += 1.0
    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    norm = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, colored, 0.4, 0)
    return overlay
