import os
import json
import logging
from pathlib import Path
from PIL import Image
import torch
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class StationDetector:
    def __init__(self, config):
        self.config = config
        self.cache_dir = Path(config["paths"]["cache_dir"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.detections_path = self.cache_dir / "detections.json"
        self.conf_thresh = config["detection"]["confidence_threshold"]
        self.prompts = config["detection"]["prompts"]
        
        self.min_height_pct = config["alignment"]["min_height_pct"]
        self.max_height_pct = config["alignment"].get("max_height_pct", 0.65)
        self.max_width_pct = config["alignment"].get("max_width_pct", 0.55)
        
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"StationDetector initialized using device: {self.device}")
        
        self.model_type = None
        self.model = None
        self.processor = None

    def _init_model(self):
        if self.model is not None:
            return

        # Attempt 0: Custom Fine-Tuned WeatherXM Sensor Head YOLO model
        custom_weights = Path(self.config.get("paths", {}).get("weights_dir", "weights")) / "weatherxm_sensor_head_best.pt"
        if not custom_weights.exists():
            custom_weights = Path(__file__).resolve().parent.parent / "weights" / "weatherxm_sensor_head_best.pt"

        if custom_weights.exists():
            try:
                from ultralytics import YOLO
                logger.info(f"Loading custom fine-tuned WeatherXM Sensor Head model ({custom_weights})...")
                self.model = YOLO(str(custom_weights))
                self.model_type = "weatherxm_yolo"
                logger.info("Custom WeatherXM Sensor Head model loaded successfully.")
                return
            except Exception as e:
                logger.warning(f"Failed to load custom weights ({e}). Trying fallback...")

        # Attempt 1: Try Ultralytics YOLO-World v2 (High accuracy & tight sensor head boxes)
        try:
            from ultralytics import YOLO
            logger.info("Loading YOLO-World model (yolov8x-worldv2.pt)...")
            self.model = YOLO("yolov8x-worldv2.pt")
            self.model.set_classes(self.prompts)
            self.model_type = "yolo_world"
            logger.info("YOLO-World v2 loaded successfully.")
            return
        except Exception as e:
            logger.warning(f"Failed to load YOLO-World ({e}). Falling back to Grounding DINO...")

        # Attempt 2: Try HuggingFace Grounding DINO
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            logger.info("Loading Grounding DINO model (IDEA-Research/grounding-dino-tiny)...")
            model_id = "IDEA-Research/grounding-dino-tiny"
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
            self.model_type = "grounding_dino"
            logger.info("Grounding DINO loaded successfully.")
            return
        except Exception as e:
            logger.warning(f"Failed to load Grounding DINO ({e}).")

        self.model_type = "saliency"

    def detect_folder(self, input_dir, force_recompute=False):
        input_dir = Path(input_dir)
        detections = {}

        if self.detections_path.exists() and not force_recompute:
            try:
                with open(self.detections_path, "r") as f:
                    detections = json.load(f)
                logger.info(f"Loaded {len(detections)} cached detections from {self.detections_path}")
            except Exception as e:
                logger.warning(f"Could not load detection cache: {e}")

        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        image_paths = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in image_extensions])

        to_process = [p for p in image_paths if p.name not in detections or force_recompute]

        if not to_process:
            logger.info("All images already processed in detection cache.")
            return detections

        logger.info(f"Processing {len(to_process)} images for station detection...")
        self._init_model()

        valid_count = 0
        skipped_count = 0

        for idx, img_path in enumerate(to_process):
            res = self._detect_single_image(img_path)
            detections[img_path.name] = res
            
            if res["status"] == "valid":
                valid_count += 1
            else:
                skipped_count += 1

            if (idx + 1) % 50 == 0 or (idx + 1) == len(to_process):
                logger.info(f"Detection Progress: [{idx + 1}/{len(to_process)}] - Valid: {valid_count}, Skipped: {skipped_count}")
                with open(self.detections_path, "w") as f:
                    json.dump(detections, f, indent=2)

        with open(self.detections_path, "w") as f:
            json.dump(detections, f, indent=2)

        total_valid = sum(1 for v in detections.values() if v.get("status") == "valid")
        total_skipped = sum(1 for v in detections.values() if v.get("status") != "valid")
        logger.info(f"Detection complete. Total scanned: {len(detections)} | Valid: {total_valid} | Skipped: {total_skipped}")

        return detections

    def detect_single(self, img_path):
        self._init_model()
        return self._detect_single_image(Path(img_path))

    def _detect_single_image(self, img_path):
        try:
            image = Image.open(img_path).convert("RGB")
            img_w, img_h = image.size
        except Exception as e:
            return {"status": "skipped", "reason": f"corrupt_image: {e}"}

        res = None
        if self.model_type == "weatherxm_yolo":
            res = self._detect_custom_yolo(img_path, img_w, img_h)
        elif self.model_type == "yolo_world":
            res = self._detect_yolo_world(img_path, img_w, img_h)
        elif self.model_type == "grounding_dino":
            res = self._detect_grounding_dino(image, img_w, img_h)

        if res is None or res.get("status") != "valid":
            res = self._detect_saliency_station_head(img_path, img_w, img_h)

        return res

    def _detect_custom_yolo(self, img_path, img_w, img_h):
        try:
            results = self.model(str(img_path), conf=0.10, device=self.device, verbose=False)[0]
            boxes = results.boxes

            best_box = None
            best_score = -1.0

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    score = float(box.conf[0].cpu().numpy())
                    x1, y1, x2, y2 = [float(v) for v in xyxy]
                    box_w = x2 - x1
                    box_h = y2 - y1
                    w_pct = box_w / img_w
                    h_pct = box_h / img_h

                    if self.min_height_pct <= h_pct <= self.max_height_pct and w_pct <= self.max_width_pct:
                        if score > best_score:
                            best_score = score
                            best_box = [x1, y1, x2, y2]

                if best_box is None and len(boxes) > 0:
                    top_box = boxes[0]
                    best_score = float(top_box.conf[0].cpu().numpy())
                    best_box = [float(v) for v in top_box.xyxy[0].cpu().numpy()]

            if best_box is not None:
                return {
                    "status": "valid",
                    "bbox": [round(v, 1) for v in best_box],
                    "confidence": round(float(best_score), 4),
                    "model": "weatherxm_sensor_head_best.pt",
                    "img_width": img_w,
                    "img_height": img_h
                }
            else:
                return {"status": "skipped", "reason": "no_sensor_head_detected", "img_width": img_w, "img_height": img_h}

        except Exception as e:
            return {"status": "skipped", "reason": f"custom_yolo_error: {e}", "img_width": img_w, "img_height": img_h}

    def _detect_yolo_world(self, img_path, img_w, img_h):
        try:
            results = self.model(str(img_path), conf=self.conf_thresh, verbose=False)[0]
            boxes = results.boxes

            best_box = None
            best_score = -1.0

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    score = float(box.conf[0].cpu().numpy())
                    x1, y1, x2, y2 = [float(v) for v in xyxy]
                    box_w = x2 - x1
                    box_h = y2 - y1
                    w_pct = box_w / img_w
                    h_pct = box_h / img_h

                    if self.min_height_pct <= h_pct <= self.max_height_pct and w_pct <= self.max_width_pct:
                        if score > best_score:
                            best_score = score
                            best_box = [x1, y1, x2, y2]

            if best_box is not None:
                return {
                    "status": "valid",
                    "bbox": best_box,
                    "confidence": float(best_score),
                    "img_width": img_w,
                    "img_height": img_h
                }
            else:
                return {"status": "skipped", "reason": "no_tight_station_box_found", "img_width": img_w, "img_height": img_h}

        except Exception as e:
            return {"status": "skipped", "reason": f"yolo_error: {e}", "img_width": img_w, "img_height": img_h}

    def _detect_grounding_dino(self, image, img_w, img_h):
        try:
            text_prompt = ". ".join(self.prompts) + "."
            inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)

            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.conf_thresh,
                text_threshold=self.conf_thresh,
                target_sizes=[(img_h, img_w)]
            )[0]

            boxes = results["boxes"].cpu().numpy()
            scores = results["scores"].cpu().numpy()

            best_box = None
            best_score = -1.0

            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = [float(v) for v in box]
                box_w = x2 - x1
                box_h = y2 - y1
                w_pct = box_w / img_w
                h_pct = box_h / img_h

                if self.min_height_pct <= h_pct <= self.max_height_pct and w_pct <= self.max_width_pct:
                    if score > best_score:
                        best_score = score
                        best_box = [x1, y1, x2, y2]

            if best_box is not None:
                return {
                    "status": "valid",
                    "bbox": best_box,
                    "confidence": float(best_score),
                    "img_width": img_w,
                    "img_height": img_h
                }
            else:
                return {"status": "skipped", "reason": "no_tight_station_box_found", "img_width": img_w, "img_height": img_h}

        except Exception as e:
            logger.warning(f"Error running Grounding DINO on image: {e}")
            return {"status": "skipped", "reason": f"detection_error: {e}", "img_width": img_w, "img_height": img_h}

    def _detect_saliency_station_head(self, img_path, img_w, img_h):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                return {"status": "skipped", "reason": "read_error", "img_width": img_w, "img_height": img_h}

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            crop_top = int(img_h * 0.05)
            crop_bottom = int(img_h * 0.75)
            crop_left = int(img_w * 0.15)
            crop_right = int(img_w * 0.85)

            roi = gray[crop_top:crop_bottom, crop_left:crop_right]
            blurred = cv2.GaussianBlur(roi, (5, 5), 0)
            
            edges = cv2.Canny(blurred, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            best_contour_box = None
            max_area = 0.0

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                w_pct = w / img_w
                h_pct = h / img_h
                area = w * h

                if 0.05 <= h_pct <= self.max_height_pct and 0.05 <= w_pct <= self.max_width_pct:
                    if area > max_area:
                        max_area = area
                        gx1 = crop_left + x
                        gy1 = crop_top + y
                        gx2 = gx1 + w
                        gy2 = gy1 + h
                        best_contour_box = [float(gx1), float(gy1), float(gx2), float(gy2)]

            if best_contour_box is not None:
                return {
                    "status": "valid",
                    "bbox": best_contour_box,
                    "confidence": 0.40,
                    "saliency_fallback": True,
                    "img_width": img_w,
                    "img_height": img_h
                }
            else:
                return {"status": "skipped", "reason": "no_station_head_detected", "img_width": img_w, "img_height": img_h}

        except Exception as e:
            return {"status": "skipped", "reason": f"saliency_error: {e}", "img_width": img_w, "img_height": img_h}
