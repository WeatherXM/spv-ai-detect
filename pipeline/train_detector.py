import os
import sys
import json
import shutil
import random
import logging
from pathlib import Path

logger = logging.getLogger("Trainer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
PHOTOS_DIR = BASE_DIR / "photos"
METADATA_DIR = BASE_DIR / "metadata"
CACHE_DIR = BASE_DIR / "cache"
DATASET_DIR = BASE_DIR / "dataset"
WEIGHTS_DIR = BASE_DIR / "weights"

ANNOTATIONS_FILE = METADATA_DIR / "training_annotations.json"
if not ANNOTATIONS_FILE.exists() and (CACHE_DIR / "training_annotations.json").exists():
    ANNOTATIONS_FILE = CACHE_DIR / "training_annotations.json"

METADATA_FILE = METADATA_DIR / "weatherxm_photos_metadata.json"

def calculate_distance_category(height_pct):
    if height_pct >= 35.0:
        return "CLOSE_UP"
    elif height_pct >= 15.0:
        return "MEDIUM"
    elif height_pct >= 6.0:
        return "DISTANT"
    else:
        return "TOO_FAR"

def export_dataset():
    if not ANNOTATIONS_FILE.exists():
        return {"status": "error", "message": "No annotations file found in cache."}

    with open(ANNOTATIONS_FILE, "r") as f:
        annotations = json.load(f)

    if not annotations:
        return {"status": "error", "message": "Annotation list is empty. Please annotate at least 15-20 photos."}

    logger.info(f"Exporting dataset with {len(annotations)} annotated photos...")

    # Prepare directories
    train_img_dir = DATASET_DIR / "images" / "train"
    val_img_dir = DATASET_DIR / "images" / "val"
    train_lbl_dir = DATASET_DIR / "labels" / "train"
    val_lbl_dir = DATASET_DIR / "labels" / "val"

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Clean old symlinks / labels
    for p in (DATASET_DIR / "images").glob("**/*"):
        if p.is_file() or p.is_symlink():
            p.unlink()
    for p in (DATASET_DIR / "labels").glob("**/*"):
        if p.is_file() or p.is_symlink():
            p.unlink()

    # Split train / val (85% train, 15% val)
    random.seed(42)
    items = list(annotations.items())
    random.shuffle(items)

    val_count = max(1, int(len(items) * 0.15))
    val_items = dict(items[:val_count])
    train_items = dict(items[val_count:])

    def process_split(split_items, img_dest_dir, lbl_dest_dir):
        for fname, data in split_items.items():
            src_img = PHOTOS_DIR / fname
            if not src_img.exists():
                src_img = BASE_DIR / fname
            if not src_img.exists():
                logger.warning(f"Image {fname} not found on disk, skipping.")
                continue

            # Link image
            dest_img = img_dest_dir / fname
            if dest_img.exists() or dest_img.is_symlink():
                dest_img.unlink()
            try:
                dest_img.symlink_to(src_img)
            except Exception:
                shutil.copy2(src_img, dest_img)

            # Write YOLO label: class_id cx cy w h
            bbox_norm = data.get("bbox_norm")
            if not bbox_norm and "bbox_pixels" in data:
                x1, y1, x2, y2 = data["bbox_pixels"]
                img_w = data["img_width"]
                img_h = data["img_height"]
                cx = ((x1 + x2) / 2.0) / img_w
                cy = ((y1 + y2) / 2.0) / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h
                bbox_norm = [cx, cy, w, h]

            if bbox_norm:
                cx, cy, w, h = bbox_norm
                lbl_file = lbl_dest_dir / f"{src_img.stem}.txt"
                with open(lbl_file, "w") as lf:
                    lf.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    process_split(train_items, train_img_dir, train_lbl_dir)
    process_split(val_items, val_img_dir, val_lbl_dir)

    # Write data.yaml
    yaml_content = f"""path: {DATASET_DIR}
train: images/train
val: images/val

names:
  0: weatherxm_sensor_head
"""
    yaml_path = DATASET_DIR / "data.yaml"
    with open(yaml_path, "w") as yf:
        yf.write(yaml_content)

    # Generate comprehensive metadata file
    all_metadata = {}
    for fname, data in annotations.items():
        img_w = data.get("img_width", 1920)
        img_h = data.get("img_height", 1080)
        x1, y1, x2, y2 = data.get("bbox_pixels", [0, 0, 0, 0])
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        h_pct = (box_h / img_h) * 100.0
        w_pct = (box_w / img_w) * 100.0

        target_canvas_station_h = 1080 * 0.40  # 432 pixels
        rec_zoom = target_canvas_station_h / box_h if box_h > 0 else 1.0

        all_metadata[fname] = {
            "filename": fname,
            "station_detected": True,
            "source_width": img_w,
            "source_height": img_h,
            "bbox_sensor_head": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "center_pixel": [round((x1 + x2) / 2.0, 1), round((y1 + y2) / 2.0, 1)],
            "center_norm": [round(((x1 + x2) / 2.0) / img_w, 4), round(((y1 + y2) / 2.0) / img_h, 4)],
            "sensor_head_width_px": round(box_w, 1),
            "sensor_head_height_px": round(box_h, 1),
            "station_height_pct": round(h_pct, 2),
            "station_width_pct": round(w_pct, 2),
            "aspect_ratio": round(box_w / box_h, 3),
            "distance_category": calculate_distance_category(h_pct),
            "recommended_zoom_to_canvas_target": round(rec_zoom, 2),
            "is_manually_verified": True
        }

    with open(METADATA_FILE, "w") as mf:
        json.dump(all_metadata, mf, indent=2)

    cache_meta = CACHE_DIR / "weatherxm_photos_metadata.json"
    with open(cache_meta, "w") as cf:
        json.dump(all_metadata, cf, indent=2)

    logger.info(f"Dataset exported: {len(train_items)} train, {len(val_items)} val. Metadata saved to {METADATA_FILE}")
    return {
        "status": "success",
        "train_count": len(train_items),
        "val_count": len(val_items),
        "total_annotations": len(annotations),
        "yaml_path": str(yaml_path),
        "metadata_path": str(METADATA_FILE)
    }

def train_yolo(epochs=35, imgsz=640, batch_size=16):
    export_res = export_dataset()
    if export_res.get("status") != "success":
        return export_res

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO
        import torch

        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Starting YOLOv8s fine-tuning on device: {device} (imgsz={imgsz}, batch={batch_size}) for {epochs} epochs...")

        model = YOLO("yolov8s.pt")
        yaml_path = DATASET_DIR / "data.yaml"

        results = model.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            project=str(WEIGHTS_DIR),
            name="weatherxm_sensor_head",
            exist_ok=True,
            verbose=True
        )

        best_pt = WEIGHTS_DIR / "weatherxm_sensor_head" / "weights" / "best.pt"
        target_pt = WEIGHTS_DIR / "weatherxm_sensor_head_best.pt"

        if best_pt.exists():
            shutil.copy2(best_pt, target_pt)
            logger.info(f"Model training complete! Best weights saved to {target_pt}")
            return {
                "status": "success",
                "message": "Model trained successfully!",
                "weights_path": str(target_pt)
            }
        else:
            return {"status": "error", "message": "Training finished but best.pt was not found."}

    except Exception as e:
        logger.error(f"Training failed: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        train_yolo()
    else:
        res = export_dataset()
        print(json.dumps(res, indent=2))
