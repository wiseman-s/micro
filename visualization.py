
import cv2

def draw_boxes(img, detections):
    for d in detections:
        x,y,w,h = d["x"],d["y"],d["w"],d["h"]
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    return img
