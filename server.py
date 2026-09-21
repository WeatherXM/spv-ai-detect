import os
import sys
import json
import shutil
import logging
import threading
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import yaml
import torch
import cv2
import numpy as np

logger = logging.getLogger("MatchCutServer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = Path(__file__).resolve().parent
PHOTOS_DIR = BASE_DIR / "photos"
METADATA_DIR = BASE_DIR / "metadata"
CACHE_DIR = BASE_DIR / "cache"
NORM_DIR = CACHE_DIR / "normalized_images"
CONFIG_PATH = BASE_DIR / "config.yaml"
ANNOTATIONS_FILE = METADATA_DIR / "training_annotations.json"
META_FILE = METADATA_DIR / "weatherxm_photos_metadata.json"
DIM_CACHE_FILE = CACHE_DIR / "image_dimensions.json"
REG_FILE = CACHE_DIR / "normalized_registry.json"
ADJ_FILE = METADATA_DIR / "manual_adjustments.json"
CACHE_ADJ_FILE = CACHE_DIR / "manual_adjustments.json"
DET_FILE = CACHE_DIR / "detections.json"
EXCL_FILE = METADATA_DIR / "review_exclusions.json"
CACHE_EXCL_FILE = CACHE_DIR / "review_exclusions.json"
SEQ_OVERRIDE_FILE = METADATA_DIR / "sequence_override.json"
CACHE_SEQ_OVERRIDE_FILE = CACHE_DIR / "sequence_override.json"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

io_lock = threading.RLock()

_photo_hashes_cache = None

def get_photo_hashes():
    global _photo_hashes_cache
    if _photo_hashes_cache is not None:
        return _photo_hashes_cache
    import hashlib
    hashes = {}
    if PHOTOS_DIR.exists():
        for p in PHOTOS_DIR.glob("*.*"):
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                try:
                    h = hashlib.md5(p.read_bytes()).hexdigest()
                    hashes[p.name] = h
                except Exception:
                    pass
    _photo_hashes_cache = hashes
    return _photo_hashes_cache

def deduplicate_sequence(seq):
    if not seq:
        return []
    hashes = get_photo_hashes()
    seen_hashes = set()
    seen_names = set()
    deduped = []
    for fname in seq:
        if fname in seen_names:
            continue
        seen_names.add(fname)
        h = hashes.get(fname)
        if h:
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
        deduped.append(fname)
    return deduped

def load_metadata_safe():
    if META_FILE.exists():
        try:
            with open(META_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, dict) and len(data) > 0:
                    return data
        except Exception as e:
            logger.error(f"Error loading {META_FILE}: {e}")
    return {}

def save_metadata_safe(meta_dict):
    if not isinstance(meta_dict, dict) or len(meta_dict) < 50:
        logger.warning(f"Refusing to save empty or truncated metadata of size {len(meta_dict) if isinstance(meta_dict, dict) else 0}")
        return False
    with io_lock:
        try:
            tmp_file = META_FILE.with_suffix(".tmp")
            with open(tmp_file, "w") as f:
                json.dump(meta_dict, f, indent=2)
            tmp_file.replace(META_FILE)
            # Also sync cache copy
            try:
                cache_meta = CACHE_DIR / "weatherxm_photos_metadata.json"
                cache_meta.parent.mkdir(parents=True, exist_ok=True)
                tmp_cache = cache_meta.with_suffix(".tmp")
                with open(tmp_cache, "w") as f:
                    json.dump(meta_dict, f, indent=2)
                tmp_cache.replace(cache_meta)
            except Exception as ce:
                logger.warning(f"Could not sync metadata cache: {ce}")
            return True
        except Exception as e:
            logger.error(f"Error saving {META_FILE}: {e}")
            return False

def load_registry_safe():
    if REG_FILE.exists():
        try:
            with open(REG_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, dict) and len(data) > 0:
                    return data
        except Exception as e:
            logger.error(f"Error loading {REG_FILE}: {e}")
    return {}

def save_registry_entry_safe(filename, norm_data):
    with io_lock:
        reg = load_registry_safe()
        if len(reg) < 50 and REG_FILE.exists() and REG_FILE.stat().st_size > 5000:
            logger.warning(f"Refusing to save truncated registry of size {len(reg)} over existing registry file")
            return False
        reg[filename] = norm_data
        try:
            tmp_file = REG_FILE.with_suffix(".tmp")
            with open(tmp_file, "w") as f:
                json.dump(reg, f, indent=2)
            tmp_file.replace(REG_FILE)
            return True
        except Exception as e:
            logger.error(f"Error saving {REG_FILE}: {e}")
            return False

def load_adjustments_safe():
    for p in [ADJ_FILE, CACHE_ADJ_FILE]:
        if p.exists():
            try:
                with open(p, "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
            except Exception as e:
                logger.error(f"Error loading {p}: {e}")
    return {}

def save_adjustment_entry_safe(filename, adj):
    with io_lock:
        all_adj = load_adjustments_safe()
        all_adj[filename] = adj
        try:
            for target_path in [ADJ_FILE, CACHE_ADJ_FILE]:
                tmp_file = target_path.with_suffix(".tmp")
                with open(tmp_file, "w") as f:
                    json.dump(all_adj, f, indent=2)
                tmp_file.replace(target_path)
            return True
        except Exception as e:
            logger.error(f"Error saving {ADJ_FILE}: {e}")
            return False

def load_detections_safe():
    if DET_FILE.exists():
        try:
            with open(DET_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, dict) and len(data) > 0:
                    return data
        except Exception as e:
            logger.error(f"Error loading {DET_FILE}: {e}")
    return {}

def save_detection_entry_safe(filename, det_entry):
    with io_lock:
        dets = load_detections_safe()
        dets[filename] = det_entry
        try:
            tmp_file = DET_FILE.with_suffix(".tmp")
            with open(tmp_file, "w") as f:
                json.dump(dets, f, indent=2)
            tmp_file.replace(DET_FILE)
            return True
        except Exception as e:
            logger.error(f"Error saving {DET_FILE}: {e}")
            return False

def load_exclusions_safe():
    for p in [EXCL_FILE, CACHE_EXCL_FILE]:
        if p.exists():
            try:
                with open(p, "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
            except Exception as e:
                logger.error(f"Error loading {p}: {e}")
    return {"excluded": [], "manual_excluded": [], "manual_included": []}

def save_exclusions_safe(excl_dict):
    with io_lock:
        try:
            for target_path in [EXCL_FILE, CACHE_EXCL_FILE]:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_file = target_path.with_suffix(".tmp")
                with open(tmp_file, "w") as f:
                    json.dump(excl_dict, f, indent=2)
                tmp_file.replace(target_path)
            return True
        except Exception as e:
            logger.error(f"Error saving exclusions: {e}")
            return False

def save_photo_exclusion_metadata(filename, is_excluded, reason="manual_review"):
    with io_lock:
        # 1. Update master weatherxm_photos_metadata.json permanently
        meta = load_metadata_safe()
        if filename in meta and isinstance(meta[filename], dict):
            meta[filename]["manual_excluded"] = bool(is_excluded)
            meta[filename]["is_excluded"] = bool(is_excluded)
            now_iso = datetime.now(timezone.utc).isoformat()
            if is_excluded:
                meta[filename]["manual_exclusion_timestamp"] = now_iso
                meta[filename]["exclusion_reason"] = reason
                meta[filename]["manual_included"] = False
            else:
                meta[filename]["manual_included"] = True
                meta[filename]["manual_inclusion_timestamp"] = now_iso
                meta[filename]["manual_excluded"] = False
                meta[filename]["is_excluded"] = False
                meta[filename]["exclusion_reason"] = None
            save_metadata_safe(meta)

        # 2. Update review_exclusions.json permanently
        excl = load_exclusions_safe()
        manual_excluded = set(excl.get("manual_excluded", []))
        manual_included = set(excl.get("manual_included", []))
        excluded = set(excl.get("excluded", []))

        if is_excluded:
            manual_excluded.add(filename)
            manual_included.discard(filename)
            excluded.add(filename)
        else:
            manual_excluded.discard(filename)
            manual_included.add(filename)
            excluded.discard(filename)

        excl["manual_excluded"] = sorted(list(manual_excluded))
        excl["manual_included"] = sorted(list(manual_included))
        excl["excluded"] = sorted(list(excluded))
        save_exclusions_safe(excl)

        # 3. Update sequence_override.json if photo is in sequence
        for seq_path in [SEQ_OVERRIDE_FILE, CACHE_SEQ_OVERRIDE_FILE]:
            if seq_path.exists():
                try:
                    with open(seq_path, "r") as f:
                        seq_data = json.load(f)
                    curr_seq = seq_data.get("sequence", [])
                    if is_excluded and filename in curr_seq:
                        curr_seq = [x for x in curr_seq if x != filename]
                        seq_data["sequence"] = curr_seq
                        seq_data["count"] = len(curr_seq)
                        tmp_s = seq_path.with_suffix(".tmp")
                        with open(tmp_s, "w") as f:
                            json.dump(seq_data, f, indent=2)
                        tmp_s.replace(seq_path)
                except Exception as e:
                    logger.error(f"Error updating sequence override for {seq_path}: {e}")

        return {
            "status": "success",
            "filename": filename,
            "is_excluded": is_excluded,
            "total_manual_excluded": len(manual_excluded),
            "total_excluded": len(excluded)
        }

training_status = {"running": False, "progress": "", "result": None}

def run_training_thread():
    global training_status
    training_status = {"running": True, "progress": "Starting YOLO fine-tuning on Apple Silicon (mps)...", "result": None}
    python_bin = BASE_DIR / ".venv" / "bin" / "python3.12"
    train_script = BASE_DIR / "pipeline" / "train_detector.py"
    log_file = BASE_DIR / "weights" / "training.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(log_file, "w") as out_f:
            p = subprocess.Popen(
                [str(python_bin), str(train_script), "--train"],
                cwd=str(BASE_DIR),
                stdout=out_f,
                stderr=subprocess.STDOUT
            )
            p.wait()
        if p.returncode == 0:
            training_status = {"running": False, "progress": "Completed", "result": {"status": "success"}}
        else:
            training_status = {"running": False, "progress": f"Error code {p.returncode}", "result": {"status": "error"}}
    except Exception as e:
        logger.error(f"Training subprocess error: {e}")
        training_status = {"running": False, "progress": f"Error: {e}", "result": {"status": "error", "message": str(e)}}

_DETECTOR = None
_NORMALIZER = None
_MODEL_LOCK = threading.Lock()

def get_detector():
    global _DETECTOR
    with _MODEL_LOCK:
        if _DETECTOR is None:
            with open(CONFIG_PATH) as f:
                cfg = yaml.safe_load(f)
            from pipeline.detector import StationDetector
            _DETECTOR = StationDetector(cfg)
            _DETECTOR._init_model()
    return _DETECTOR

def get_normalizer():
    global _NORMALIZER
    with _MODEL_LOCK:
        if _NORMALIZER is None:
            with open(CONFIG_PATH) as f:
                cfg = yaml.safe_load(f)
            from pipeline.normalizer import ImageNormalizer
            _NORMALIZER = ImageNormalizer(cfg)
    return _NORMALIZER

METRICS_FILE = CACHE_DIR / "image_metrics.json"

def compute_single_photo_metric(filename, meta=None, dims=None, reg=None):
    if dims is None:
        dims = {}
        if DIM_CACHE_FILE.exists():
            try:
                with open(DIM_CACHE_FILE) as f:
                    dims = json.load(f)
            except Exception:
                pass
    if meta is None:
        meta = load_metadata_safe()

    if reg is None:
        reg = {}
        reg_file = CACHE_DIR / "normalized_registry.json"
        if reg_file.exists():
            try:
                with open(reg_file) as rf:
                    reg = json.load(rf)
            except Exception:
                pass

    entry = meta.get(filename, {})
    d = dims.get(filename, {})
    img_h = float(d.get("height", 1080))
    reg_entry = reg.get(filename, {})

    h_eval = entry.get("human_evaluation") or {}
    a_eval = entry.get("ai_evaluation") or {}

    # Prefer actual rendered natural canvas station height if registered
    if reg_entry.get("natural_station_h_pct") is not None:
        pct = float(reg_entry["natural_station_h_pct"])
    elif h_eval.get("station_height_pct") is not None:
        pct = float(h_eval["station_height_pct"])
    elif h_eval.get("bbox"):
        b = h_eval["bbox"]
        pct = (abs(b[3] - b[1]) / img_h) * 100.0
    elif a_eval.get("bbox"):
        b = a_eval["bbox"]
        pct = (abs(b[3] - b[1]) / img_h) * 100.0
    else:
        pct = 15.0

    norm_path = NORM_DIR / filename
    img_path = norm_path if norm_path.exists() else (PHOTOS_DIR / filename)
    brightness = 128.0
    hex_color = "#888888"

    if img_path.exists():
        try:
            img = cv2.imread(str(img_path))
            if img is not None:
                small = cv2.resize(img, (80, 45), interpolation=cv2.INTER_NEAREST)
                lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
                brightness = float(np.mean(lab[:, :, 0]))
                mean_bgr = np.mean(small, axis=(0, 1))
                hex_color = '#{:02x}{:02x}{:02x}'.format(int(mean_bgr[2]), int(mean_bgr[1]), int(mean_bgr[0]))
        except Exception:
            pass

    # AI Confidence
    conf = 0.0
    if a_eval.get("confidence") is not None:
        conf = float(a_eval["confidence"])
    else:
        det_file = CACHE_DIR / "detections.json"
        if det_file.exists():
            try:
                with open(det_file) as df:
                    d_all = json.load(df)
                    if filename in d_all and d_all[filename].get("confidence") is not None:
                        conf = float(d_all[filename]["confidence"])
            except Exception:
                pass

    return {
        "zoom_height_pct": round(pct, 2),
        "brightness": round(brightness, 1),
        "color_hex": hex_color,
        "confidence": round(conf, 4),
        "confidence_pct": round(conf * 100, 1)
    }

def update_photo_metric(filename):
    metrics = {}
    if METRICS_FILE.exists():
        try:
            with open(METRICS_FILE) as f:
                metrics = json.load(f)
        except Exception:
            metrics = {}
    metrics[filename] = compute_single_photo_metric(filename)
    try:
        with open(METRICS_FILE, "w") as f:
            json.dump(metrics, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save image_metrics.json: {e}")

def get_all_image_metrics():
    meta = load_metadata_safe()
    metrics = {}
    if METRICS_FILE.exists():
        try:
            with open(METRICS_FILE) as f:
                metrics = json.load(f)
        except Exception:
            metrics = {}

    reg_file = CACHE_DIR / "normalized_registry.json"
    reg_data = {}
    if reg_file.exists():
        try:
            with open(reg_file) as rf:
                reg_data = json.load(rf)
        except Exception:
            pass

    missing_keys = [k for k in meta.keys() if k not in metrics or metrics[k].get("zoom_height_pct") is None]
    if missing_keys:
        dims = {}
        if DIM_CACHE_FILE.exists():
            try:
                with open(DIM_CACHE_FILE) as f:
                    dims = json.load(f)
            except Exception:
                pass
        for fname in missing_keys:
            metrics[fname] = compute_single_photo_metric(fname, meta=meta, dims=dims, reg=reg_data)
        try:
            with open(METRICS_FILE, "w") as f:
                json.dump(metrics, f, indent=2)
        except Exception:
            pass
    return metrics

batch_redetect_status = {
    "running": False,
    "completed": False,
    "total": 0,
    "current": 0,
    "current_file": "",
    "detected_valid": 0,
    "skipped_human": 0,
    "error": None,
    "abort": False
}

def run_batch_redetect_thread(file_list=None, protect_human=True):
    global batch_redetect_status
    batch_redetect_status["running"] = True
    batch_redetect_status["completed"] = False
    batch_redetect_status["abort"] = False
    batch_redetect_status["detected_valid"] = 0
    batch_redetect_status["skipped_human"] = 0
    batch_redetect_status["error"] = None

    try:
        detector = get_detector()
        normalizer = get_normalizer()

        if not file_list:
            seq_file = METADATA_DIR / "sequence_override.json"
            if not seq_file.exists():
                seq_file = CACHE_DIR / "sequence_override.json"
            if not seq_file.exists():
                seq_file = METADATA_DIR / "sequence.json"
            if not seq_file.exists():
                seq_file = CACHE_DIR / "sequence.json"

            if seq_file.exists():
                with open(seq_file) as f:
                    file_list = json.load(f).get("sequence", [])
            else:
                exts = {".jpg", ".jpeg", ".png", ".webp"}
                file_list = sorted([f.name for f in PHOTOS_DIR.iterdir() if f.suffix.lower() in exts])

        batch_redetect_status["total"] = len(file_list)
        det_file = CACHE_DIR / "detections.json"
        reg_file = CACHE_DIR / "normalized_registry.json"
        adj_file = CACHE_DIR / "manual_adjustments.json"

        all_meta = load_metadata_safe()

        detections = {}
        if det_file.exists():
            try:
                with open(det_file) as f:
                    detections = json.load(f)
            except Exception:
                pass

        registry = {}
        if reg_file.exists():
            try:
                with open(reg_file) as f:
                    registry = json.load(f)
            except Exception:
                pass

        all_adj = {}
        if adj_file.exists():
            try:
                with open(adj_file) as f:
                    all_adj = json.load(f)
            except Exception:
                pass

        valid_count = 0
        skipped_human_count = 0

        for i, fname in enumerate(file_list):
            if batch_redetect_status.get("abort"):
                logger.info("Batch redetection aborted by user.")
                break

            batch_redetect_status["current"] = i + 1
            batch_redetect_status["current_file"] = fname

            img_path = (PHOTOS_DIR / fname) if (PHOTOS_DIR / fname).exists() else (BASE_DIR / fname)
            if not img_path.exists():
                continue

            # Check if photo has human evaluation
            meta_entry = all_meta.get(fname, {})
            has_human = meta_entry.get("human_evaluation") is not None
            if not has_human and fname in all_adj:
                adj_item = all_adj[fname]
                if adj_item.get("zoom", 1.0) != 1.0 or adj_item.get("shift_x", 0) != 0 or adj_item.get("shift_y", 0) != 0:
                    has_human = True

            det = detector.detect_single(img_path)
            detections[fname] = det

            # Update AI evaluation in metadata
            if fname not in all_meta:
                all_meta[fname] = {"filename": fname}

            if det.get("status") == "valid":
                valid_count += 1
                batch_redetect_status["detected_valid"] = valid_count
                all_meta[fname]["ai_evaluation"] = {
                    "model": "weatherxm_sensor_head_best.pt",
                    "confidence": round(det.get("confidence", 0.0), 4),
                    "confidence_pct": round(det.get("confidence", 0.0) * 100, 1),
                    "bbox": det.get("bbox"),
                    "updated_at": datetime.now(timezone.utc).isoformat()
                }

            if protect_human and has_human:
                # Keep human evaluation protected!
                skipped_human_count += 1
                batch_redetect_status["skipped_human"] = skipped_human_count
            else:
                # Apply AI detection as the active normalization
                if det.get("status") == "valid":
                    out_path = NORM_DIR / fname
                    if fname in all_adj:
                        del all_adj[fname]
                    success, norm_data = normalizer._normalize_single_image(img_path, det, out_path, {"zoom": 1.0, "shift_x": 0.0, "shift_y": 0.0})
                    if success:
                        registry[fname] = norm_data
                        all_meta[fname]["active_source"] = "ai"

            if (i + 1) % 10 == 0 or (i + 1) == len(file_list):
                with open(det_file, "w") as f:
                    json.dump(detections, f, indent=2)
                with open(reg_file, "w") as f:
                    json.dump(registry, f, indent=2)
                with open(adj_file, "w") as f:
                    json.dump(all_adj, f, indent=2)
                save_metadata_safe(all_meta)

        with open(det_file, "w") as f:
            json.dump(detections, f, indent=2)
        with open(reg_file, "w") as f:
            json.dump(registry, f, indent=2)
        with open(adj_file, "w") as f:
            json.dump(all_adj, f, indent=2)
        save_metadata_safe(all_meta)

        batch_redetect_status["running"] = False
        batch_redetect_status["completed"] = True
    except Exception as e:
        logger.exception(f"Batch redetect error: {e}")
        batch_redetect_status["error"] = str(e)
        batch_redetect_status["running"] = False

batch_rebuild_status = {
    "running": False,
    "completed": False,
    "total": 0,
    "current": 0,
    "current_file": "",
    "recomputed": 0,
    "error": None,
    "abort": False,
    "message": ""
}

def run_batch_rebuild_thread(force=False):
    global batch_rebuild_status
    batch_rebuild_status["running"] = True
    batch_rebuild_status["completed"] = False
    batch_rebuild_status["abort"] = False
    batch_rebuild_status["total"] = 0
    batch_rebuild_status["current"] = 0
    batch_rebuild_status["recomputed"] = 0
    batch_rebuild_status["error"] = None
    batch_rebuild_status["message"] = "Initializing preview rebuild..."

    try:
        normalizer = get_normalizer()
        with open(DET_FILE) as f:
            detections = json.load(f)

        valid_detections = {
            filename: det for filename, det in detections.items()
            if det.get("status") == "valid"
        }
        total = len(valid_detections)
        batch_rebuild_status["total"] = total

        registry = {}
        if REG_FILE.exists() and not force:
            try:
                with open(REG_FILE, "r") as f:
                    registry = json.load(f)
            except Exception:
                pass

        adj_file = CACHE_ADJ_FILE
        if not adj_file.exists() and ADJ_FILE.exists():
            adj_file = ADJ_FILE
        adjustments = {}
        if adj_file.exists():
            try:
                with open(adj_file, "r") as f:
                    adjustments = json.load(f)
            except Exception:
                pass

        recomputed_count = 0
        for i, (fname, det) in enumerate(valid_detections.items()):
            if batch_rebuild_status.get("abort"):
                logger.info("Batch rebuild previews aborted by user.")
                break

            batch_rebuild_status["current"] = i + 1
            batch_rebuild_status["current_file"] = fname

            out_path = normalizer.norm_dir / fname
            natural_out = normalizer.natural_dir / fname
            adj = adjustments.get(fname, {})
            has_adj = len(adj) > 0
            reg_entry = registry.get(fname, {})
            missing_natural_geom = (
                reg_entry.get("natural_reg_bbox") is None
                or reg_entry.get("natural_scale") is None
                or reg_entry.get("natural_canvas_offset") is None
            )

            if not out_path.exists() or not natural_out.exists() or force or has_adj or fname not in registry or missing_natural_geom:
                img_path = PHOTOS_DIR / fname
                if not img_path.exists():
                    img_path = BASE_DIR / fname
                if img_path.exists():
                    success, norm_data = normalizer._normalize_single_image(img_path, det, out_path, adj)
                    if success:
                        registry[fname] = norm_data
                        recomputed_count += 1
                        batch_rebuild_status["recomputed"] = recomputed_count

            if (i + 1) % 25 == 0 or (i + 1) == total:
                with open(REG_FILE, "w") as f:
                    json.dump(registry, f, indent=2)
                batch_rebuild_status["message"] = f"Processed {i + 1}/{total} (Recomputed {recomputed_count})..."

        with open(REG_FILE, "w") as f:
            json.dump(registry, f, indent=2)

        batch_rebuild_status["running"] = False
        batch_rebuild_status["completed"] = True
        batch_rebuild_status["message"] = f"Rebuilt previews: processed {total}, updated {recomputed_count}."
        logger.info(batch_rebuild_status["message"])
    except Exception as e:
        logger.exception(f"Batch rebuild error: {e}")
        batch_rebuild_status["error"] = str(e)
        batch_rebuild_status["running"] = False
        batch_rebuild_status["message"] = f"Error: {e}"

class MatchCutHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)

    def do_HEAD(self):
        clean_path = self.path.split("?")[0]
        img_exts = {".jpg", ".jpeg", ".png", ".webp"}
        rel_req = clean_path.lstrip("/")
        if any(rel_req.lower().endswith(ext) for ext in img_exts):
            if (PHOTOS_DIR / rel_req).exists() and not (BASE_DIR / rel_req).exists():
                self.path = f"/photos/{rel_req}"
        return super().do_HEAD()

    def do_GET(self):
        clean_path = self.path.split("?")[0]

        if clean_path in ["/", "/index.html"]:
            self.path = "/review_tool.html"
            return super().do_GET()
        elif clean_path in ["/training", "/annotate", "/training_tool"]:
            self.path = "/training_tool.html"
            return super().do_GET()
        elif clean_path == "/api/sequence":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            
            # ALWAYS load master catalog of all 801 photos from sequence.json
            master_seq_file = CACHE_DIR / "sequence.json"
            if not master_seq_file.exists():
                master_seq_file = METADATA_DIR / "sequence.json"

            reg_file = CACHE_DIR / "normalized_registry.json"
            det_file = CACHE_DIR / "detections.json"
            adj_file = CACHE_DIR / "manual_adjustments.json"

            sequence_data = []
            if master_seq_file.exists():
                with open(master_seq_file) as f:
                    seq_json = json.load(f)
                    raw_seq = seq_json if isinstance(seq_json, list) else seq_json.get("sequence", [])
                    sequence_data = deduplicate_sequence(raw_seq)

            registry_data = {}
            if reg_file.exists():
                with open(reg_file) as f:
                    registry_data = json.load(f)

            detections_data = {}
            if det_file.exists():
                with open(det_file) as f:
                    detections_data = json.load(f)

            adjustments_data = {}
            if adj_file.exists():
                with open(adj_file) as f:
                    adjustments_data = json.load(f)

            metadata_data = load_metadata_safe()
            metrics_data = get_all_image_metrics()

            # Load saved exclusions and sort mode (prefer METADATA_DIR first via load_exclusions_safe)
            ex_data = load_exclusions_safe()
            saved_excluded = ex_data.get("excluded", [])
            saved_sort_mode = ex_data.get("sort_mode", "default")
            saved_tag = ex_data.get("tag", "")
            saved_fps = int(ex_data.get("fps", 15))
            saved_min_res_mp = float(ex_data.get("min_res_mp", 0.0))
            saved_manual_excluded = ex_data.get("manual_excluded", [])
            saved_manual_included = ex_data.get("manual_included", [])

            # Fallback if empty and sequence_override exists
            if not saved_manual_excluded and not saved_excluded:
                for ov_path in [SEQ_OVERRIDE_FILE, CACHE_SEQ_OVERRIDE_FILE]:
                    if ov_path.exists():
                        try:
                            with open(ov_path) as f:
                                ov_data = json.load(f)
                                saved_sort_mode = ov_data.get("sort_mode", saved_sort_mode)
                                saved_tag = ov_data.get("tag", saved_tag)
                                saved_fps = int(ov_data.get("fps", saved_fps))
                                saved_min_res_mp = float(ov_data.get("min_res_mp", saved_min_res_mp))
                                if ov_data.get("manual_excluded"):
                                    saved_manual_excluded = ov_data.get("manual_excluded")
                                if ov_data.get("manual_included"):
                                    saved_manual_included = ov_data.get("manual_included")
                                break
                        except Exception:
                            pass

            # Ground truth merge: Any photo marked manual_excluded or is_excluded in weatherxm_photos_metadata.json
            meta_manual_excluded = set()
            meta_manual_included = set()
            for fname, meta_info in metadata_data.items():
                if isinstance(meta_info, dict):
                    if meta_info.get("manual_excluded") is True or meta_info.get("is_excluded") is True:
                        meta_manual_excluded.add(fname)
                    elif meta_info.get("manual_included") is True:
                        meta_manual_included.add(fname)

            saved_manual_excluded = sorted(list(set(saved_manual_excluded or []) | meta_manual_excluded))
            saved_manual_included = sorted(list(set(saved_manual_included or []) | meta_manual_included))
            saved_excluded = sorted(list(set(saved_excluded or []) | set(saved_manual_excluded)))

            dim_file = METADATA_DIR / "image_dimensions.json" if (METADATA_DIR / "image_dimensions.json").exists() else (CACHE_DIR / "image_dimensions.json")
            dimensions_data = {}
            if dim_file.exists():
                try:
                    with open(dim_file) as f:
                        dimensions_data = json.load(f)
                except Exception:
                    pass

            response = {
                "sequence": sequence_data,
                "registry": registry_data,
                "detections": detections_data,
                "adjustments": adjustments_data,
                "metadata": metadata_data,
                "metrics": metrics_data,
                "dimensions": dimensions_data,
                "saved_sort_mode": saved_sort_mode,
                "saved_tag": saved_tag,
                "saved_fps": saved_fps,
                "saved_min_res_mp": saved_min_res_mp,
                "saved_manual_excluded": saved_manual_excluded,
                "saved_manual_included": saved_manual_included,
                "excluded": saved_excluded
            }
            self.wfile.write(json.dumps(response).encode("utf-8"))
            return

        elif clean_path == "/api/training_data":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()

            # Load dimensions
            dims = {}
            if DIM_CACHE_FILE.exists():
                with open(DIM_CACHE_FILE) as f:
                    dims = json.load(f)

            # Load annotations
            annotations = {}
            if ANNOTATIONS_FILE.exists():
                with open(ANNOTATIONS_FILE) as f:
                    annotations = json.load(f)
            elif (CACHE_DIR / "training_annotations.json").exists():
                with open(CACHE_DIR / "training_annotations.json") as f:
                    annotations = json.load(f)

            # Load detection suggestions
            det_file = CACHE_DIR / "detections.json"
            detections = {}
            if det_file.exists():
                with open(det_file) as f:
                    detections = json.load(f)

            # Get all image filenames from photos/
            img_exts = {".jpg", ".jpeg", ".png", ".webp"}
            img_source = PHOTOS_DIR if PHOTOS_DIR.exists() else BASE_DIR
            all_images = sorted([p.name for p in img_source.iterdir() if p.suffix.lower() in img_exts])

            # Prepare list of photo items
            items = []
            for fname in all_images:
                dim = dims.get(fname, {"width": 1920, "height": 1080})
                anno = annotations.get(fname)
                det = detections.get(fname, {})

                suggested_box = None
                conf = 0.0
                if det.get("status") == "valid" and "bbox" in det:
                    suggested_box = det["bbox"]
                    conf = det.get("confidence", 0.0)

                items.append({
                    "filename": fname,
                    "width": dim.get("width", 1920),
                    "height": dim.get("height", 1080),
                    "is_annotated": anno is not None,
                    "annotation": anno,
                    "suggested_box": suggested_box,
                    "detection_conf": conf,
                    "is_saliency": det.get("saliency_fallback", False)
                })

            response = {
                "total_images": len(items),
                "annotated_count": len(annotations),
                "target_count": 50,
                "items": items,
                "annotations": annotations,
                "training_status": training_status
            }
            self.wfile.write(json.dumps(response).encode("utf-8"))
            return

        elif clean_path == "/api/training_status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(training_status).encode("utf-8"))
            return

        elif clean_path == "/api/batch_progress":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(batch_redetect_status).encode("utf-8"))
            return

        elif clean_path == "/api/rebuild_previews_status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(batch_rebuild_status).encode("utf-8"))
            return

        elif clean_path == "/api/output_info":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "output_dir": str(OUTPUT_DIR.resolve()),
                "exists": OUTPUT_DIR.exists()
            }).encode("utf-8"))
            return

        elif clean_path == "/api/render_progress":
            prog_file = CACHE_DIR / "render_progress.json"
            if prog_file.exists():
                try:
                    with open(prog_file, "r") as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        data["output_dir"] = str(OUTPUT_DIR.resolve())
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(data).encode("utf-8"))
                    return
                except Exception:
                    pass
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "running": False,
                "completed": False,
                "percent": 0.0,
                "step": 0,
                "total_steps": 8,
                "treatment": "",
                "format": "",
                "label": "Ready to render",
                "sub_label": "",
                "error": None,
                "output_dir": str(OUTPUT_DIR.resolve())
            }).encode("utf-8"))
            return

        elif clean_path == "/api/model_info":
            weights_file = BASE_DIR / "weights" / "weatherxm_sensor_head_best.pt"
            has_model = weights_file.exists()
            size_mb = round(weights_file.stat().st_size / (1024 * 1024), 2) if has_model else 0
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "has_trained_model": has_model,
                "model_name": "weatherxm_sensor_head_best.pt",
                "model_size_mb": size_mb,
                "device": "mps" if torch.backends.mps.is_available() else "cpu"
            }).encode("utf-8"))
            return

        else:
            # Seamless fallback: if root requested an image that is now in photos/
            rel_req = clean_path.lstrip("/")
            img_exts = {".jpg", ".jpeg", ".png", ".webp"}
            if any(rel_req.lower().endswith(ext) for ext in img_exts):
                if (PHOTOS_DIR / rel_req).exists() and not (BASE_DIR / rel_req).exists():
                    self.path = f"/photos/{rel_req}"
            return super().do_GET()

    def do_POST(self):
        global training_status, batch_redetect_status, batch_rebuild_status
        clean_path = self.path.split("?")[0]

        if clean_path == "/api/save_sequence":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            payload = json.loads(post_data.decode("utf-8"))

            sequence = deduplicate_sequence(payload.get("sequence", []))
            excluded = payload.get("excluded", [])
            sort_mode = payload.get("sort_mode", "default")
            tag = payload.get("tag", "")
            fps = int(payload.get("fps", 15))
            min_res_mp = float(payload.get("min_res_mp", 0.0))
            manual_excluded = payload.get("manual_excluded", None)
            manual_included = payload.get("manual_included", None)

            # Preserve existing manual exclusions if not explicitly sent
            existing_excl = load_exclusions_safe()
            if manual_excluded is None:
                manual_excluded = existing_excl.get("manual_excluded", [])
            if manual_included is None:
                manual_included = existing_excl.get("manual_included", [])

            # Ensure all manual_excluded / manual_included are permanently synced to photo metadata
            all_meta = load_metadata_safe()
            meta_dirty = False
            now_iso = datetime.now(timezone.utc).isoformat()
            if manual_excluded:
                for f in manual_excluded:
                    if f in all_meta and isinstance(all_meta[f], dict) and not all_meta[f].get("manual_excluded"):
                        all_meta[f]["manual_excluded"] = True
                        all_meta[f]["is_excluded"] = True
                        all_meta[f]["manual_exclusion_timestamp"] = now_iso
                        meta_dirty = True
            if manual_included:
                for f in manual_included:
                    if f in all_meta and isinstance(all_meta[f], dict) and all_meta[f].get("manual_excluded"):
                        all_meta[f]["manual_excluded"] = False
                        all_meta[f]["is_excluded"] = False
                        all_meta[f]["manual_included"] = True
                        meta_dirty = True
            if meta_dirty:
                save_metadata_safe(all_meta)

            override_data = {
                "count": len(sequence),
                "sequence": sequence,
                "sort_mode": sort_mode,
                "tag": tag,
                "fps": fps,
                "min_res_mp": min_res_mp,
                "manual_excluded": manual_excluded,
                "manual_included": manual_included
            }

            override_path = METADATA_DIR / "sequence_override.json"
            cache_override_path = CACHE_DIR / "sequence_override.json"

            with open(override_path, "w") as f:
                json.dump(override_data, f, indent=2)
            with open(cache_override_path, "w") as f:
                json.dump(override_data, f, indent=2)

            exclusion_data = {
                "excluded": excluded,
                "sort_mode": sort_mode,
                "tag": tag,
                "fps": fps,
                "min_res_mp": min_res_mp,
                "manual_excluded": manual_excluded,
                "manual_included": manual_included
            }

            save_exclusions_safe(exclusion_data)

            res_info = f", min_res: {min_res_mp}MP" if min_res_mp > 0 else ""
            logger.info(f"Auto-saved sequence with {len(sequence)} photos, {len(excluded)} exclusions ({len(manual_excluded)} manual) @ {fps}fps{res_info} (sort: {sort_mode}, tag: {tag})")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success", "count": len(sequence), "excluded_count": len(excluded), "manual_excluded_count": len(manual_excluded), "sort_mode": sort_mode, "tag": tag, "fps": fps, "min_res_mp": min_res_mp}).encode("utf-8"))
            return

        elif clean_path == "/api/toggle_exclude":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            payload = json.loads(post_data.decode("utf-8"))

            filename = payload.get("filename")
            is_excluded = bool(payload.get("excluded", True))
            reason = payload.get("reason", "manual_review")

            if not filename:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": "Missing filename parameter"}).encode("utf-8"))
                return

            res = save_photo_exclusion_metadata(filename, is_excluded, reason=reason)
            logger.info(f"Permanent exclusion toggle: {filename} -> excluded={is_excluded} (Total manual excluded: {res.get('total_manual_excluded')})")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))
            return

        elif clean_path == "/api/save_adjustments":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            payload = json.loads(post_data.decode("utf-8"))

            filename = payload.get("filename")
            adj = payload.get("adjustment", {})

            # 1. Safely save adjustment
            save_adjustment_entry_safe(filename, adj)

            # 2. Safely update metadata
            all_meta = load_metadata_safe()
            if filename not in all_meta:
                all_meta[filename] = {"filename": filename}
            all_meta[filename]["active_source"] = "human"
            existing_human = all_meta[filename].get("human_evaluation") or {}
            existing_human["source"] = existing_human.get("source", "reviewer_adjustment")
            existing_human["verified"] = True
            existing_human["adjustment"] = adj
            existing_human["updated_at"] = datetime.now(timezone.utc).isoformat()
            all_meta[filename]["human_evaluation"] = existing_human
            save_metadata_safe(all_meta)

            # 3. Determine base bbox: prefer human bbox first, then detections, then ai
            detections = load_detections_safe()
            meta = all_meta.get(filename) or {}
            human = meta.get("human_evaluation") or {}
            ai = meta.get("ai_evaluation") or {}
            det_entry = detections.get(filename) or {}

            bbox = None
            if human.get("bbox"):
                bbox = human["bbox"]
            elif det_entry.get("bbox"):
                bbox = det_entry["bbox"]
            elif ai.get("bbox"):
                bbox = ai["bbox"]

            det = {"status": "valid", "bbox": bbox} if bbox else det_entry

            from pipeline.normalizer import ImageNormalizer
            with open(CONFIG_PATH) as f:
                config = yaml.safe_load(f)

            normalizer = ImageNormalizer(config)
            out_path = NORM_DIR / filename
            img_path = (PHOTOS_DIR / filename) if (PHOTOS_DIR / filename).exists() else (BASE_DIR / filename)
            success, norm_data = normalizer._normalize_single_image(img_path, det, out_path, adj)
            if success:
                save_registry_entry_safe(filename, norm_data)
                all_meta = load_metadata_safe()
                if filename in all_meta:
                    all_meta[filename]["is_too_high"] = bool(norm_data.get("is_too_high"))
                    save_metadata_safe(all_meta)

            update_photo_metric(filename)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success", "norm_data": norm_data}).encode("utf-8"))
            return

        elif clean_path == "/api/save_annotation":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            payload = json.loads(post_data.decode("utf-8"))

            filename = payload.get("filename")
            bbox = payload.get("bbox")  # [x1, y1, x2, y2] or [x, y, w, h]

            img_path = (PHOTOS_DIR / filename) if (PHOTOS_DIR / filename).exists() else (BASE_DIR / filename)
            real_img = cv2.imread(str(img_path))
            if real_img is not None:
                img_h, img_w = real_img.shape[:2]
            else:
                img_w = float(payload.get("img_width", 1920))
                img_h = float(payload.get("img_height", 1080))

            v0, v1, v2, v3 = [float(v) for v in bbox]
            if v2 > v0 and v3 > v1:
                # [x1, y1, x2, y2]
                x1, y1, x2, y2 = v0, v1, v2, v3
                box_w = max(1.0, x2 - x1)
                box_h = max(1.0, y2 - y1)
            else:
                # [x, y, w, h]
                x1, y1 = v0, v1
                box_w = max(1.0, v2)
                box_h = max(1.0, v3)
                x2 = x1 + box_w
                y2 = y1 + box_h

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            h_pct = (box_h / img_h) * 100.0
            w_pct = (box_w / img_w) * 100.0

            # Determine distance category
            if h_pct >= 35.0:
                dist_cat = "CLOSE_UP"
            elif h_pct >= 15.0:
                dist_cat = "MEDIUM"
            elif h_pct >= 6.0:
                dist_cat = "DISTANT"
            else:
                dist_cat = "TOO_FAR"

            target_canvas_station_h = 1080 * 0.40
            rec_zoom = min(2.5, target_canvas_station_h / box_h) if box_h > 0 else 1.0

            annotation_entry = {
                "filename": filename,
                "bbox_pixels": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "bbox_norm": [round(cx / img_w, 6), round(cy / img_h, 6), round(box_w / img_w, 6), round(box_h / img_h, 6)],
                "img_width": int(img_w),
                "img_height": int(img_h),
                "station_height_pct": round(h_pct, 2),
                "station_width_pct": round(w_pct, 2),
                "aspect_ratio": round(box_w / box_h, 3),
                "distance_category": dist_cat,
                "recommended_zoom": round(rec_zoom, 2),
                "is_manually_verified": True
            }

            all_annotations = {}
            if ANNOTATIONS_FILE.exists():
                try:
                    with open(ANNOTATIONS_FILE, "r") as f:
                        all_annotations = json.load(f)
                except Exception as e:
                    logger.warning(f"Error reading annotations: {e}")

            all_annotations[filename] = annotation_entry

            with open(ANNOTATIONS_FILE, "w") as f:
                json.dump(all_annotations, f, indent=2)
            cache_anno_path = CACHE_DIR / "training_annotations.json"
            with open(cache_anno_path, "w") as f:
                json.dump(all_annotations, f, indent=2)

            all_meta = load_metadata_safe()
            if filename not in all_meta:
                all_meta[filename] = {"filename": filename}
            all_meta[filename]["active_source"] = "human"
            all_meta[filename]["is_manually_verified"] = True
            all_meta[filename]["human_evaluation"] = {
                "source": "training_annotation",
                "verified": True,
                "bbox": annotation_entry["bbox_pixels"],
                "bbox_norm": annotation_entry["bbox_norm"],
                "station_height_pct": annotation_entry["station_height_pct"],
                "station_width_pct": annotation_entry["station_width_pct"],
                "distance_category": annotation_entry["distance_category"],
                "recommended_zoom": annotation_entry["recommended_zoom"],
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            save_metadata_safe(all_meta)

            # 1. Update detections.json so any future stage knows this station location
            new_det = {
                "status": "valid",
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "confidence": 1.0,
                "label": "station_head",
                "source": "human_annotation"
            }
            save_detection_entry_safe(filename, new_det)

            # 2. Reset manual adjustments to neutral 1.0x and 0 shift
            save_adjustment_entry_safe(filename, {"zoom": 1.0, "shift_x": 0.0, "shift_y": 0.0})

            # 3. Immediately re-normalize the 1080p frame centered at 40% height
            norm_data = {}
            try:
                from pipeline.normalizer import ImageNormalizer
                with open(CONFIG_PATH) as f:
                    config = yaml.safe_load(f)
                normalizer = ImageNormalizer(config)
                out_path = NORM_DIR / filename
                success, norm_data = normalizer._normalize_single_image(
                    img_path, new_det, out_path, {"zoom": 1.0, "shift_x": 0.0, "shift_y": 0.0}
                )
                if success:
                    save_registry_entry_safe(filename, norm_data)
                    all_meta = load_metadata_safe()
                    if filename in all_meta:
                        all_meta[filename]["is_too_high"] = bool(norm_data.get("is_too_high"))
                        save_metadata_safe(all_meta)
            except Exception as e:
                logger.error(f"Error re-normalizing annotated image {filename}: {e}")

            update_photo_metric(filename)

            logger.info(f"Saved annotation & re-normalized {filename} (Height: {h_pct:.1f}%, Cat: {dist_cat})")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "success",
                "annotation": annotation_entry,
                "norm_data": norm_data,
                "station_h_pct": norm_data.get("station_h_pct", 40.0) if norm_data else 40.0,
                "total_annotated": len(all_annotations)
            }).encode("utf-8"))
            return

        elif clean_path == "/api/delete_annotation":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            payload = json.loads(post_data.decode("utf-8"))

            filename = payload.get("filename")
            all_annotations = {}
            if ANNOTATIONS_FILE.exists():
                try:
                    with open(ANNOTATIONS_FILE, "r") as f:
                        all_annotations = json.load(f)
                except Exception as e:
                    pass
            elif (CACHE_DIR / "training_annotations.json").exists():
                try:
                    with open(CACHE_DIR / "training_annotations.json", "r") as f:
                        all_annotations = json.load(f)
                except Exception as e:
                    pass

            if filename in all_annotations:
                del all_annotations[filename]
                with open(ANNOTATIONS_FILE, "w") as f:
                    json.dump(all_annotations, f, indent=2)
                cache_anno_path = CACHE_DIR / "training_annotations.json"
                with open(cache_anno_path, "w") as f:
                    json.dump(all_annotations, f, indent=2)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success", "total_annotated": len(all_annotations)}).encode("utf-8"))
            return

        elif clean_path == "/api/export_dataset":
            from pipeline.train_detector import export_dataset
            res = export_dataset()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))
            return

        elif clean_path == "/api/train_detector":
            if training_status.get("running"):
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": "Training is already running!"}).encode("utf-8"))
                return

            t = threading.Thread(target=run_training_thread, daemon=True)
            t.start()

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success", "message": "Training started in background!"}).encode("utf-8"))
            return

        elif clean_path == "/api/open_folder":
            try:
                target = OUTPUT_DIR.resolve()
                content_len = int(self.headers.get("Content-Length", 0))
                if content_len > 0:
                    try:
                        p_data = json.loads(self.rfile.read(content_len).decode("utf-8"))
                        ver = p_data.get("version", "").strip().upper()
                        if ver:
                            ver_dir = target / f"Version_{ver}"
                            if ver_dir.exists():
                                target = ver_dir
                    except Exception:
                        pass

                target.mkdir(parents=True, exist_ok=True)
                if sys.platform == "darwin":
                    subprocess.Popen(["open", str(target)])
                elif sys.platform.startswith("win"):
                    subprocess.Popen(["explorer", str(target)])
                else:
                    subprocess.Popen(["xdg-open", str(target)])

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "success",
                    "folder": str(target)
                }).encode("utf-8"))
            except Exception as e:
                logger.error(f"Error opening folder: {e}")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode("utf-8"))
            return

        elif clean_path == "/api/render":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            payload = json.loads(post_data.decode("utf-8"))
            sequence = payload.get("sequence", [])
            sort_mode = payload.get("sort_mode", "custom")
            tag = payload.get("tag", "")
            resolution = payload.get("resolution", "both")
            fps = int(payload.get("fps", 15))
            min_res_mp = float(payload.get("min_res_mp", 0.0))
            treatments_req = payload.get("treatments")
            if isinstance(treatments_req, str):
                selected_treatments = [t.strip().upper() for t in treatments_req.split(",") if t.strip()]
            elif isinstance(treatments_req, list):
                selected_treatments = [str(t).strip().upper() for t in treatments_req if str(t).strip()]
            else:
                selected_treatments = ["A", "B", "C", "D", "E"]

            valid_treatments = ["A", "B", "C", "D", "E"]
            selected_treatments = [t for t in selected_treatments if t in valid_treatments]
            if not selected_treatments:
                selected_treatments = ["C"]

            parts_req = payload.get("parts", payload.get("split_parts", None))
            if parts_req is not None:
                try:
                    num_parts = int(parts_req)
                except (ValueError, TypeError):
                    num_parts = 3 if bool(parts_req) else 1
            else:
                num_parts = 3

            if not tag and sort_mode != "custom":
                tag_map = {
                    "original": "orig",
                    "similarity_greedy": "sim_greedy",
                    "similarity_tsp": "sim_tsp",
                    "brightness": "lum_l2d",
                    "brightness_reverse": "lum_d2l",
                    "sharpness": "sharp_h2l",
                    "sharpness_reverse": "sharp_l2h",
                    "zoom_far_to_near": "zoom_f2n",
                    "zoom_near_to_far": "zoom_n2f",
                    "color_light_to_dark": "col_l2d",
                    "color_dark_to_light": "col_d2l",
                    "conf_low_to_high": "conf_l2h",
                    "conf_high_to_low": "conf_h2l"
                }
                tag = tag_map.get(sort_mode, "")

            override_data = {
                "count": len(sequence),
                "sequence": sequence,
                "sort_mode": sort_mode,
                "tag": tag,
                "resolution": resolution,
                "fps": fps,
                "min_res_mp": min_res_mp,
                "treatments": selected_treatments,
                "parts": num_parts
            }

            override_path = CACHE_DIR / "sequence_override.json"
            meta_override_path = METADATA_DIR / "sequence_override.json"

            with open(override_path, "w") as f:
                json.dump(override_data, f, indent=2)
            with open(meta_override_path, "w") as f:
                json.dump(override_data, f, indent=2)

            res_info = f", min_res: {min_res_mp}MP" if min_res_mp > 0 else ""
            t_str = ",".join(selected_treatments)
            parts_info = f", parts: {num_parts}" if num_parts > 1 else ""
            logger.info(f"Rendering updated videos for {len(sequence)} photos (sort: {sort_mode}, tag: '{tag}', res: '{resolution}', fps: {fps}, treatments: [{t_str}]{parts_info}{res_info})...")

            # Initialize progress state for real-time frontend monitoring
            t_count = len(selected_treatments)
            # 1080p Master: MP4, WebM, WebP Poster = 3 steps per part
            # 720p Web: 5 web assets (Desktop MP4, WebM, Mobile MP4, WebM, Poster) = 5 steps per part
            steps_1080p = t_count * 3 * num_parts
            steps_720p = t_count * 5 * num_parts
            total_steps = (steps_1080p + steps_720p) if resolution == "both" else (steps_1080p if resolution == "1080p" else steps_720p)
            parts_label = f" ({num_parts} Parts)" if num_parts > 1 else ""
            res_label = f"1080p Master & 720p Web Bundles{parts_label}" if resolution == "both" else (f"1080p Master{parts_label}" if resolution == "1080p" else f"720p 5-Asset Web Bundles{parts_label}")
            init_prog = {
                "running": True,
                "completed": False,
                "percent": 0.0,
                "step": 0,
                "total_steps": total_steps,
                "treatment": "Init",
                "format": "",
                "label": f"Initializing {res_label} @ {fps}fps render pipeline for Version [{t_str}]...",
                "sub_label": f"Preparing {len(sequence)} photos...",
                "error": None
            }
            try:
                with open(CACHE_DIR / "render_progress.json", "w") as f:
                    json.dump(init_prog, f)
            except Exception:
                pass

            python_bin = sys.executable or str(BASE_DIR / ".venv" / "bin" / "python")
            cmd = [str(python_bin), "main.py", "--config", "config.yaml", "--stage", "normalize"]
            subprocess.run(cmd, cwd=str(BASE_DIR))

            cmd_render = [
                str(python_bin), "main.py", "--config", "config.yaml",
                "--stage", "render", "--resolution", resolution, "--fps", str(fps),
                "--treatments", ",".join(selected_treatments),
                "--parts", str(num_parts)
            ]
            if tag:
                cmd_render.extend(["--suffix", tag])

            try:
                res = subprocess.run(cmd_render, cwd=str(BASE_DIR), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if res.returncode == 0:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "status": "success",
                        "message": "Render completed successfully!",
                        "sort_mode": sort_mode,
                        "tag": tag,
                        "count": len(sequence)
                    }).encode("utf-8"))
                else:
                    err_prog = {
                        "running": False,
                        "completed": False,
                        "percent": 0.0,
                        "step": 0,
                        "total_steps": total_steps,
                        "treatment": "",
                        "format": "",
                        "label": "Render failed",
                        "sub_label": res.stderr[:200] if res.stderr else "FFmpeg error",
                        "error": res.stderr
                    }
                    try:
                        with open(CACHE_DIR / "render_progress.json", "w") as f:
                            json.dump(err_prog, f)
                    except Exception:
                        pass
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "error", "message": res.stderr}).encode("utf-8"))
            except Exception as e:
                err_prog = {
                    "running": False,
                    "completed": False,
                    "percent": 0.0,
                    "step": 0,
                    "total_steps": total_steps,
                    "treatment": "",
                    "format": "",
                    "label": "Render failed",
                    "sub_label": str(e),
                    "error": str(e)
                }
                try:
                    with open(CACHE_DIR / "render_progress.json", "w") as f:
                        json.dump(err_prog, f)
                except Exception:
                    pass
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode("utf-8"))
            return

        elif clean_path == "/api/redetect_photo":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            payload = json.loads(post_data.decode("utf-8"))
            filename = payload.get("filename")
            force_replace = bool(payload.get("force_replace", False))

            img_path = (PHOTOS_DIR / filename) if (PHOTOS_DIR / filename).exists() else (BASE_DIR / filename)
            if not img_path.exists():
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": f"Image {filename} not found"}).encode("utf-8"))
                return

            try:
                detector = get_detector()
                normalizer = get_normalizer()
                det = detector.detect_single(img_path)

                # Update cache/detections.json
                det_file = CACHE_DIR / "detections.json"
                detections = {}
                if det_file.exists():
                    try:
                        with open(det_file) as f:
                            detections = json.load(f)
                    except Exception:
                        pass
                detections[filename] = det
                with open(det_file, "w") as f:
                    json.dump(detections, f, indent=2)

                # Load dual metadata
                all_meta = load_metadata_safe()
                if filename not in all_meta:
                    all_meta[filename] = {"filename": filename}

                if det.get("status") == "valid":
                    all_meta[filename]["ai_evaluation"] = {
                        "model": "weatherxm_sensor_head_best.pt",
                        "confidence": round(det.get("confidence", 0.0), 4),
                        "confidence_pct": round(det.get("confidence", 0.0) * 100, 1),
                        "bbox": det.get("bbox"),
                        "updated_at": datetime.now(timezone.utc).isoformat()
                    }

                # Check if photo has human evaluation
                has_human = all_meta[filename].get("human_evaluation") is not None
                adj_file = CACHE_DIR / "manual_adjustments.json"
                all_adj = {}
                if adj_file.exists():
                    try:
                        with open(adj_file) as f:
                            all_adj = json.load(f)
                    except Exception:
                        pass

                was_human_protected = False
                norm_data = {}

                if has_human and not force_replace:
                    # Protect human evaluation! Keep manual adjustments and normalized crop intact
                    was_human_protected = True
                    reg_file = CACHE_DIR / "normalized_registry.json"
                    if reg_file.exists():
                        try:
                            with open(reg_file) as f:
                                registry = json.load(f)
                            norm_data = registry.get(filename, {})
                        except Exception:
                            pass
                else:
                    # Apply AI normalization as active
                    if filename in all_adj:
                        del all_adj[filename]
                        with open(adj_file, "w") as f:
                            json.dump(all_adj, f, indent=2)

                    all_meta[filename]["active_source"] = "ai"

                    if det.get("status") == "valid":
                        out_path = NORM_DIR / filename
                        success, norm_data = normalizer._normalize_single_image(img_path, det, out_path, {"zoom": 1.0, "shift_x": 0.0, "shift_y": 0.0})
                        if success:
                            save_registry_entry_safe(filename, norm_data)
                            all_meta[filename]["is_too_high"] = bool(norm_data.get("is_too_high"))

                save_metadata_safe(all_meta)
                update_photo_metric(filename)

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "success" if det.get("status") == "valid" else "not_found",
                    "filename": filename,
                    "detection": det,
                    "norm_data": norm_data,
                    "station_h_pct": norm_data.get("station_h_pct", 40.0) if norm_data else 40.0,
                    "was_human_protected": was_human_protected,
                    "active_source": all_meta[filename].get("active_source", "ai"),
                    "human_evaluation": all_meta[filename].get("human_evaluation"),
                    "ai_evaluation": all_meta[filename].get("ai_evaluation")
                }).encode("utf-8"))
            except Exception as e:
                logger.exception(f"Error re-detecting photo {filename}: {e}")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode("utf-8"))
            return

        elif clean_path == "/api/redetect_batch":
            if batch_redetect_status.get("running"):
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": "Batch re-evaluation is already running!"}).encode("utf-8"))
                return

            content_length = int(self.headers.get("Content-Length", 0))
            payload = {}
            if content_length > 0:
                post_data = self.rfile.read(content_length)
                try:
                    payload = json.loads(post_data.decode("utf-8"))
                except Exception:
                    pass

            file_list = payload.get("filenames")
            protect_human = payload.get("protect_human", True)
            t = threading.Thread(target=run_batch_redetect_thread, args=(file_list, protect_human), daemon=True)
            t.start()

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success", "message": "Batch re-detection started!"}).encode("utf-8"))
            return

        elif clean_path == "/api/toggle_active_source":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            payload = json.loads(post_data.decode("utf-8"))
            filename = payload.get("filename")
            new_source = payload.get("source", "human")

            all_meta = load_metadata_safe()
            if filename in all_meta:
                all_meta[filename]["active_source"] = new_source
                save_metadata_safe(all_meta)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success", "active_source": new_source}).encode("utf-8"))
            return

        elif clean_path == "/api/abort_batch":
            batch_redetect_status["abort"] = True
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success", "message": "Batch re-detection abort signal sent."}).encode("utf-8"))
            return

        elif clean_path == "/api/rebuild_previews":
            if batch_rebuild_status.get("running"):
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": "Preview rebuild is already running!"}).encode("utf-8"))
                return

            content_length = int(self.headers.get("Content-Length", 0))
            payload = {}
            if content_length > 0:
                post_data = self.rfile.read(content_length)
                try:
                    payload = json.loads(post_data.decode("utf-8"))
                except Exception:
                    pass

            force = bool(payload.get("force", False))
            t = threading.Thread(target=run_batch_rebuild_thread, args=(force,), daemon=True)
            t.start()

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success", "message": "Preview rebuild started!"}).encode("utf-8"))
            return

        elif clean_path == "/api/abort_rebuild_previews":
            batch_rebuild_status["abort"] = True
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success", "message": "Preview rebuild abort signal sent."}).encode("utf-8"))
            return

def run_server(port=8080):
    server_address = ("", port)
    httpd = ThreadingHTTPServer(server_address, MatchCutHandler)
    logger.info(f"Interactive Review & Annotation Server running at http://localhost:{port}/")
    httpd.serve_forever()

if __name__ == "__main__":
    port = 8080
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    run_server(port)
