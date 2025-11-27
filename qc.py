import cv2
import numpy as np
from matplotlib import pyplot as plt

def analyze_quality(img):
    """
    Simple QC: sharpness (variance of Laplacian), brightness (mean), noise (stddev).
    Returns dict with fields and passes flag.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = float(gray.mean())
    noise = float(gray.std())
    passes = (fm > 70) and (60 < brightness < 220)
    return {"sharpness": round(float(fm), 2), "brightness": round(brightness, 2), "noise": round(noise, 2), "passes": bool(passes)}

def brightness_histogram_plot(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.hist(gray.ravel(), bins=64)
    ax.set_title("Brightness histogram")
    return fig
