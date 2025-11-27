
import cv2

class SyntheticDetector:
    def detect(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7,7), 0)
        edges = cv2.Canny(blur, 50, 120)

        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            if w*h < 80:
                continue
            detections.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
        return detections
