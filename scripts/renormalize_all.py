#!/usr/bin/env python3
"""
Re-normalizes all active photos in the dataset using the updated config:
- Center preservation: detected station center mapped exactly to (960, 540)
- Zoom cut-off: max_auto_scale = 2.5 (prevents digital upscaling blur on distant stations)
- Top border check: flags is_too_high = (ty > 0)
- Uses ProcessPoolExecutor for high-speed multi-core execution
- Atomically saves cache/normalized_registry.json and updates metadata
"""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import time
import json
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed

CONFIG_PATH = BASE_DIR / "config.yaml"
PHOTOS_DIR = BASE_DIR / "photos"
NORM_DIR = BASE_DIR / "cache" / "normalized_images"
REGISTRY_PATH = BASE_DIR / "cache" / "normalized_registry.json"
DET_FILE = BASE_DIR / "cache" / "detections.json"
META_FILE = BASE_DIR / "metadata" / "weatherxm_photos_metadata.json"
ADJ_FILE = BASE_DIR / "cache" / "manual_adjustments.json"

def worker_task(item):
    import sys
    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parent.parent
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))

    fname, bbox, adj, config = item
    from pipeline.normalizer import ImageNormalizer
    normalizer = ImageNormalizer(config)
    img_path = (PHOTOS_DIR / fname) if (PHOTOS_DIR / fname).exists() else (BASE_DIR / fname)
    out_path = NORM_DIR / fname
    det = {"status": "valid", "bbox": bbox}
    success, norm_data = normalizer._normalize_single_image(img_path, det, out_path, adj)
    return fname, success, norm_data

def main():
    print(f"Loading configurations and metadata...")
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    with open(META_FILE) as f:
        all_meta = json.load(f)

    with open(DET_FILE) as f:
        detections = json.load(f)

    adjustments = {}
    if ADJ_FILE.exists():
        try:
            with open(ADJ_FILE) as f:
                adjustments = json.load(f)
        except Exception:
            adjustments = {}

    # Determine bboxes for all photos
    tasks = []
    for fname in detections.keys():
        meta = all_meta.get(fname, {})
        human = meta.get("human_evaluation") or {}
        ai = meta.get("ai_evaluation") or {}
        det_entry = detections.get(fname) or {}

        bbox = human.get("bbox") or det_entry.get("bbox") or ai.get("bbox")
        if not bbox:
            continue

        adj = adjustments.get(fname, {}) or (human.get("adjustment") or {})
        tasks.append((fname, bbox, adj, config))

    print(f"Starting parallel normalization of {len(tasks)} images across {os.cpu_count()} CPU cores...")
    t0 = time.time()
    registry = {}
    distant_count = 0
    too_high_count = 0
    success_count = 0

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(worker_task, item): item[0] for item in tasks}
        completed = 0
        for fut in as_completed(futures):
            completed += 1
            fname, success, norm_data = fut.result()
            if success:
                success_count += 1
                registry[fname] = norm_data
                if norm_data.get("is_distant"):
                    distant_count += 1
                if norm_data.get("is_too_high"):
                    too_high_count += 1
                if fname in all_meta:
                    all_meta[fname]["is_too_high"] = bool(norm_data.get("is_too_high"))
            if completed % 100 == 0 or completed == len(tasks):
                print(f"Progress: [{completed}/{len(tasks)}] photos processed ({time.time() - t0:.1f}s)")

    # Save registry atomically
    tmp_reg = REGISTRY_PATH.with_suffix(".tmp")
    with open(tmp_reg, "w") as f:
        json.dump(registry, f, indent=2)
    tmp_reg.replace(REGISTRY_PATH)

    # Save updated metadata atomically
    tmp_meta = META_FILE.with_suffix(".tmp")
    with open(tmp_meta, "w") as f:
        json.dump(all_meta, f, indent=2)
    tmp_meta.replace(META_FILE)

    print(f"Finished re-normalization in {time.time() - t0:.2f}s!")
    print(f"Total photos normalized: {success_count}/{len(tasks)}")
    print(f"Distant stations (scale capped at 2.5): {distant_count}")
    print(f"Too High stations (ty > 0): {too_high_count}")
    print(f"Registry updated: {REGISTRY_PATH}")
    print(f"Metadata updated: {META_FILE}")

if __name__ == "__main__":
    main()
