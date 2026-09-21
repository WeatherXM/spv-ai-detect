import os
import json
import logging
from pathlib import Path
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class ImageNormalizer:
    def __init__(self, config):
        self.config = config
        self.cache_dir = Path(config["paths"]["cache_dir"])
        self.norm_dir = self.cache_dir / "normalized_images"
        self.norm_dir.mkdir(parents=True, exist_ok=True)
        self.natural_dir = self.cache_dir / "natural_images"
        self.natural_dir.mkdir(parents=True, exist_ok=True)
        
        self.canvas_w = config["canvas"]["width"]
        self.canvas_h = config["canvas"]["height"]
        
        self.target_cx = config["alignment"].get("target_center_x", 0.50) * self.canvas_w
        self.target_cy = config["alignment"].get("target_center_y", 0.50) * self.canvas_h
        self.target_station_h = config["alignment"].get("target_height_pct", 0.40) * self.canvas_h
        self.max_auto_scale = float(config.get("alignment", {}).get("max_auto_scale", 2.5))
        self.max_total_scale = float(config.get("alignment", {}).get("max_total_scale", 4.0))
        
        self.metadata_dir = Path(config["paths"].get("metadata_dir", "metadata"))
        self.registry_path = self.cache_dir / "normalized_registry.json"
        self.adjustments_path = self.metadata_dir / "manual_adjustments.json"
        if not self.adjustments_path.exists() and (self.cache_dir / "manual_adjustments.json").exists():
            self.adjustments_path = self.cache_dir / "manual_adjustments.json"

    def normalize_folder(self, input_dir, detections, force_recompute=False):
        input_dir = Path(input_dir)
        registry = {}

        if self.registry_path.exists() and not force_recompute:
            try:
                with open(self.registry_path, "r") as f:
                    registry = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load normalized registry: {e}")

        # Load manual adjustments if present
        adjustments = {}
        if self.adjustments_path.exists():
            try:
                with open(self.adjustments_path, "r") as f:
                    adjustments = json.load(f)
                logger.info(f"Loaded {len(adjustments)} manual image adjustment overrides.")
            except Exception as e:
                logger.warning(f"Could not load manual adjustments: {e}")

        valid_detections = {
            filename: det for filename, det in detections.items()
            if det.get("status") == "valid"
        }

        logger.info(f"Normalizing {len(valid_detections)} valid detected photos...")

        for idx, (filename, det) in enumerate(valid_detections.items()):
            out_path = self.norm_dir / filename
            adj = adjustments.get(filename, {})

            has_adj = len(adj) > 0
            natural_out = self.natural_dir / filename
            reg_entry = registry.get(filename, {})
            missing_natural_geom = (
                reg_entry.get("natural_reg_bbox") is None
                or reg_entry.get("natural_scale") is None
                or reg_entry.get("natural_canvas_offset") is None
            )
            if not out_path.exists() or not natural_out.exists() or force_recompute or has_adj or filename not in registry or missing_natural_geom:
                img_path = input_dir / filename
                success, norm_data = self._normalize_single_image(img_path, det, out_path, adj)
                if success:
                    registry[filename] = norm_data
                else:
                    logger.warning(f"Failed normalization for {filename}")

            if (idx + 1) % 100 == 0 or (idx + 1) == len(valid_detections):
                logger.info(f"Normalization Progress: [{idx + 1}/{len(valid_detections)}]")
                with open(self.registry_path, "w") as f:
                    json.dump(registry, f, indent=2)

        with open(self.registry_path, "w") as f:
            json.dump(registry, f, indent=2)

        logger.info(f"Normalization complete. Total registered photos: {len(registry)}")
        return registry

    def _normalize_single_image(self, img_path, detection, out_path, adjustment=None):
        try:
            if adjustment is None:
                adjustment = {}

            img = cv2.imread(str(img_path))
            if img is None:
                return False, {"reason": "image_read_error"}

            orig_h, orig_w = img.shape[:2]
            bbox = detection["bbox"]  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox

            station_w = max(1.0, x2 - x1)
            station_h = max(1.0, y2 - y1)
            
            station_cx = (x1 + x2) / 2.0
            station_cy = (y1 + y2) / 2.0

            # Apply manual zoom multiplier if provided
            zoom_multiplier = float(adjustment.get("zoom", 1.0))
            shift_x = float(adjustment.get("shift_x", 0.0))
            shift_y = float(adjustment.get("shift_y", 0.0))

            # Base scale to achieve target height (432px on 1080p canvas)
            ideal_scale = self.target_station_h / station_h

            # Cut-off limit: If station is too far away, cap the digital zoom so it doesn't become blurry
            if self.max_auto_scale and ideal_scale > self.max_auto_scale:
                base_scale = self.max_auto_scale
                is_distant = True
            else:
                base_scale = ideal_scale
                is_distant = False

            scale = base_scale * zoom_multiplier
            if self.max_total_scale and scale > self.max_total_scale:
                scale = self.max_total_scale

            new_w = int(round(orig_w * scale))
            new_h = int(round(orig_h * scale))

            if new_w <= 0 or new_h <= 0:
                return False, {"reason": "invalid_scaled_dimensions"}

            scaled_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

            scaled_cx = station_cx * scale
            scaled_cy = station_cy * scale

            # Calculate top-left offset with manual position shift:
            # Keeps the detected station center exactly at the center of the photo (target_cx + shift_x, target_cy + shift_y)
            tx = int(round((self.target_cx - scaled_cx) + shift_x))
            ty = int(round((self.target_cy - scaled_cy) + shift_y))

            pad_top = max(0, ty)
            pad_bottom = max(0, self.canvas_h - (ty + new_h))
            pad_left = max(0, tx)
            pad_right = max(0, self.canvas_w - (tx + new_w))

            crop_y = max(0, -ty)
            crop_x = max(0, -tx)

            if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
                padded = cv2.copyMakeBorder(
                    scaled_img,
                    top=pad_top,
                    bottom=pad_bottom,
                    left=pad_left,
                    right=pad_right,
                    borderType=cv2.BORDER_CONSTANT,
                    value=[0, 0, 0]
                )
                canvas = padded[crop_y:crop_y + self.canvas_h, crop_x:crop_x + self.canvas_w]
            else:
                canvas = scaled_img[crop_y:crop_y + self.canvas_h, crop_x:crop_x + self.canvas_w]

            cv2.imwrite(str(out_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Also generate/update natural centered image (with manual zoom and shift applied)
            natural_out = self.natural_dir / out_path.name
            nat_success, nat_info = self._normalize_natural_single_image(img, orig_w, orig_h, bbox, natural_out, zoom=zoom_multiplier, shift_x=shift_x, shift_y=shift_y)

            reg_x1 = self.target_cx - (station_w * scale) / 2.0 + shift_x
            reg_x2 = self.target_cx + (station_w * scale) / 2.0 + shift_x
            reg_y1 = self.target_cy - (station_h * scale) / 2.0 + shift_y
            reg_y2 = self.target_cy + (station_h * scale) / 2.0 + shift_y

            # Station height percentage on canvas
            station_h_pct = ((reg_y2 - reg_y1) / self.canvas_h) * 100.0
            is_too_high = bool(ty > 0)

            norm_result = {
                "normalized_filename": out_path.name,
                "scale": float(scale),
                "reg_bbox": [float(reg_x1), float(reg_y1), float(reg_x2), float(reg_y2)],
                "station_h_pct": float(station_h_pct),
                "is_distant": bool(is_distant),
                "is_too_high": bool(is_too_high),
                "adjustment": adjustment,
                "canvas_offset": [int(tx), int(ty)],
                "orig_size": [int(orig_w), int(orig_h)]
            }
            if nat_success and isinstance(nat_info, dict):
                norm_result.update(nat_info)

            return True, norm_result

        except Exception as e:
            logger.error(f"Error normalizing {img_path}: {e}")
            return False, {"reason": str(e)}

    def _normalize_natural_single_image(self, img, orig_w, orig_h, bbox, out_path, zoom=1.0, shift_x=0.0, shift_y=0.0):
        """
        Normalizes photo without zoom correction: centers station head at (target_cx, target_cy)
        using only enough zoom to cover the 1080p canvas without empty margins.
        Stations far away appear naturally small and grow larger as distance shrinks.
        Manual zoom adjustment multiplier is applied if modified by user.
        """
        try:
            x1, y1, x2, y2 = bbox
            station_cx = (x1 + x2) / 2.0
            station_cy = (y1 + y2) / 2.0
            station_w = max(1.0, x2 - x1)
            station_h = max(1.0, y2 - y1)

            cover_scale = max(self.canvas_w / orig_w, self.canvas_h / orig_h)
            s_cx = self.target_cx / max(1.0, station_cx)
            s_w_cx = self.target_cx / max(1.0, orig_w - station_cx)
            s_cy = self.target_cy / max(1.0, station_cy)
            s_h_cy = self.target_cy / max(1.0, orig_h - station_cy)

            min_center_scale = max(cover_scale, s_cx, s_w_cx, s_cy, s_h_cy)
            scale = min(min_center_scale, cover_scale * 3.0) * float(zoom)

            new_w = int(round(orig_w * scale))
            new_h = int(round(orig_h * scale))
            scaled_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

            scaled_cx = station_cx * scale
            scaled_cy = station_cy * scale

            tx = int(round((self.target_cx - scaled_cx) + shift_x))
            ty = int(round((self.target_cy - scaled_cy) + shift_y))

            pad_top = max(0, ty)
            pad_bottom = max(0, self.canvas_h - (ty + new_h))
            pad_left = max(0, tx)
            pad_right = max(0, self.canvas_w - (tx + new_w))

            crop_y = max(0, -ty)
            crop_x = max(0, -tx)

            if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
                padded = cv2.copyMakeBorder(
                    scaled_img,
                    top=pad_top, bottom=pad_bottom, left=pad_left, right=pad_right,
                    borderType=cv2.BORDER_CONSTANT,
                    value=[0, 0, 0]
                )
                canvas = padded[crop_y:crop_y + self.canvas_h, crop_x:crop_x + self.canvas_w]
            else:
                canvas = scaled_img[crop_y:crop_y + self.canvas_h, crop_x:crop_x + self.canvas_w]

            cv2.imwrite(str(out_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])

            nat_x1 = self.target_cx - (station_w * scale) / 2.0 + shift_x
            nat_x2 = self.target_cx + (station_w * scale) / 2.0 + shift_x
            nat_y1 = self.target_cy - (station_h * scale) / 2.0 + shift_y
            nat_y2 = self.target_cy + (station_h * scale) / 2.0 + shift_y
            nat_station_h_pct = ((nat_y2 - nat_y1) / self.canvas_h) * 100.0

            return True, {
                "natural_scale": float(scale),
                "natural_reg_bbox": [float(nat_x1), float(nat_y1), float(nat_x2), float(nat_y2)],
                "natural_station_h_pct": float(nat_station_h_pct),
                "natural_canvas_offset": [int(tx), int(ty)]
            }
        except Exception as e:
            logger.warning(f"Error creating natural centered image for {out_path.name}: {e}")
            return False, {"reason": str(e)}
