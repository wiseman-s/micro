import numpy as np
from matplotlib import pyplot as plt

def compute_morphology_metrics(detections):
    """
    Compute simple morphological metrics in pixels.
    Returns mean/median area and aspect ratio.
    """
    metrics = {}
    sizes = []
    aspect = []
    for d in detections:
        w = d.get("w", 0); h = d.get("h", 0)
        area = w * h
        sizes.append(area)
        aspect.append(round(w / (h + 1e-6), 2) if h > 0 else 0.0)
    metrics["count"] = len(detections)
    metrics["mean_area_px"] = float(np.mean(sizes)) if sizes else 0.0
    metrics["median_area_px"] = float(np.median(sizes)) if sizes else 0.0
    metrics["mean_aspect"] = float(np.mean(aspect)) if aspect else 0.0
    return metrics

def plot_size_distribution(detections):
    areas = [d["w"] * d["h"] for d in detections] if detections else []
    fig, ax = plt.subplots(figsize=(4, 2))
    if areas:
        ax.hist(areas, bins=10)
    ax.set_title("Size distribution (px²)")
    return fig

def plot_aspect_ratio(detections):
    ratios = [d["w"] / (d["h"] + 1e-6) for d in detections] if detections else []
    fig, ax = plt.subplots(figsize=(4, 2))
    if ratios:
        ax.hist(ratios, bins=10)
    ax.set_title("Aspect ratio distribution")
    return fig
