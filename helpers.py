import os, csv
from pathlib import Path
from datetime import datetime

def ensure_dirs():
    Path("models").mkdir(exist_ok=True)
    Path("sample_images").mkdir(exist_ok=True)
    Path("uploads").mkdir(exist_ok=True)
    Path("uploads/raw").mkdir(parents=True, exist_ok=True)
    Path("uploads/processed").mkdir(parents=True, exist_ok=True)
    Path("uploads/metadata").mkdir(parents=True, exist_ok=True)
    # create metadata csv if missing
    meta = Path("uploads/metadata.csv")
    if not meta.exists():
        with open(meta, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "filename", "location", "contributor", "notes", "count"])

def save_upload(img_bytes, filename, location, contributor, notes, detections):
    t = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    base = Path("uploads")
    rawp = base / "raw" / f"{t}_{filename or 'sample'}.png"
    with open(rawp, "wb") as f:
        f.write(img_bytes)
    # record metadata
    meta = base / "metadata.csv"
    count = len(detections) if detections else 0
    with open(meta, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([t, rawp.name, location, contributor, notes, count])
    return str(rawp)
