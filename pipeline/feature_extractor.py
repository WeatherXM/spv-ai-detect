import os
import json
import logging
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self, config):
        self.config = config
        self.cache_dir = Path(config["paths"]["cache_dir"])
        self.norm_dir = self.cache_dir / "normalized_images"
        self.matrix_path = self.cache_dir / "distance_matrix.npy"
        self.features_path = self.cache_dir / "features.json"
        
        self.weights = config["features"]["weights"]
        self.clip_model_name = config["features"]["clip_model_name"]
        
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.clip_model = None
        self.clip_processor = None

    def _init_clip(self):
        if self.clip_model is not None:
            return
        try:
            from transformers import CLIPProcessor, CLIPModel
            logger.info(f"Loading CLIP model ({self.clip_model_name})...")
            self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
            self.clip_model = CLIPModel.from_pretrained(self.clip_model_name).to(self.device)
            self.clip_model.eval()
            logger.info("CLIP model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise e

    def extract_and_build_matrix(self, normalized_registry, force_recompute=False):
        filenames = sorted(list(normalized_registry.keys()))
        N = len(filenames)

        if N == 0:
            raise ValueError("No normalized images available for feature extraction!")

        if self.matrix_path.exists() and self.features_path.exists() and not force_recompute:
            try:
                dist_matrix = np.load(self.matrix_path)
                with open(self.features_path, "r") as f:
                    features_meta = json.load(f)
                if dist_matrix.shape == (N, N) and features_meta.get("filenames") == filenames:
                    logger.info(f"Loaded existing {N}x{N} distance matrix from cache.")
                    return filenames, dist_matrix
            except Exception as e:
                logger.warning(f"Could not load cached distance matrix ({e}). Recomputing...")

        logger.info(f"Extracting features for {N} normalized images...")
        self._init_clip()

        bg_histograms = []
        clip_embeddings = []
        brightness_saturations = []
        compositions = []

        for idx, fname in enumerate(filenames):
            img_path = self.norm_dir / fname
            norm_info = normalized_registry[fname]
            reg_bbox = norm_info.get("reg_bbox", [0, 0, 0, 0])

            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                logger.warning(f"Skipping unreadable normalized image: {fname}")
                continue

            # 1. CIELAB Background Color Histogram
            bg_hist = self._extract_bg_lab_hist(img_bgr, reg_bbox)
            bg_histograms.append(bg_hist)

            # 2. CLIP Embedding
            clip_emb = self._extract_clip_embedding(img_path)
            clip_embeddings.append(clip_emb)

            # 3. Brightness & Saturation
            bright_sat = self._extract_brightness_saturation(img_bgr, reg_bbox)
            brightness_saturations.append(bright_sat)

            # 4. Composition (Sky & Ground)
            comp = self._extract_composition(img_bgr)
            compositions.append(comp)

            if (idx + 1) % 50 == 0 or (idx + 1) == N:
                logger.info(f"Feature Extraction Progress: [{idx + 1}/{N}]")

        bg_histograms = np.array(bg_histograms)
        clip_embeddings = np.array(clip_embeddings)
        brightness_saturations = np.array(brightness_saturations)
        compositions = np.array(compositions)

        logger.info("Computing distance matrices...")

        # Distance 1: Lab Background Color Histogram
        D_bg = cdist(bg_histograms, bg_histograms, metric="euclidean")
        D_bg = D_bg / (np.max(D_bg) + 1e-8)

        # Distance 2: CLIP Cosine Distance
        clip_embeddings_norm = clip_embeddings / (np.linalg.norm(clip_embeddings, axis=1, keepdims=True) + 1e-8)
        D_clip = cdist(clip_embeddings_norm, clip_embeddings_norm, metric="cosine")
        D_clip = np.clip(D_clip, 0.0, 1.0)

        # Distance 3: Brightness & Saturation Difference
        D_bright = cdist(brightness_saturations, brightness_saturations, metric="cityblock")
        D_bright = D_bright / (np.max(D_bright) + 1e-8)

        # Distance 4: Composition Distance
        D_comp = cdist(compositions, compositions, metric="euclidean")
        D_comp = D_comp / (np.max(D_comp) + 1e-8)

        # Weighted Combination
        w_bg = self.weights["background_color"]
        w_clip = self.weights["clip_embedding"]
        w_bright = self.weights["brightness_saturation"]
        w_comp = self.weights["composition"]

        dist_matrix = w_bg * D_bg + w_clip * D_clip + w_bright * D_bright + w_comp * D_comp
        np.fill_diagonal(dist_matrix, 0.0)

        np.save(self.matrix_path, dist_matrix)

        features_meta = {
            "filenames": filenames,
            "count": N,
            "weights": self.weights
        }
        with open(self.features_path, "w") as f:
            json.dump(features_meta, f, indent=2)

        logger.info(f"Successfully computed {N}x{N} weighted distance matrix.")
        return filenames, dist_matrix

    def _extract_bg_lab_hist(self, img_bgr, reg_bbox):
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        h, w = lab.shape[:2]

        mask = np.ones((h, w), dtype=np.uint8) * 255
        rx1, ry1, rx2, ry2 = [int(round(v)) for v in reg_bbox]
        rx1, ry1 = max(0, rx1), max(0, ry1)
        rx2, ry2 = min(w, rx2), min(h, ry2)

        if rx2 > rx1 and ry2 > ry1:
            mask[ry1:ry2, rx1:rx2] = 0

        hist = cv2.calcHist([lab], [0, 1, 2], mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = hist.flatten()
        hist = hist / (np.sum(hist) + 1e-8)
        return hist

    def _extract_clip_embedding(self, img_path):
        image = Image.open(img_path).convert("RGB")
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            res = self.clip_model.get_image_features(**inputs)
            if hasattr(res, "pooler_output"):
                emb_tensor = res.pooler_output
            elif hasattr(res, "image_embeds"):
                emb_tensor = res.image_embeds
            elif isinstance(res, torch.Tensor):
                emb_tensor = res
            else:
                emb_tensor = res[0]
        emb = emb_tensor.cpu().numpy().flatten()
        return emb

    def _extract_brightness_saturation(self, img_bgr, reg_bbox):
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, w = lab.shape[:2]

        mask = np.ones((h, w), dtype=bool)
        rx1, ry1, rx2, ry2 = [int(round(v)) for v in reg_bbox]
        rx1, ry1 = max(0, rx1), max(0, ry1)
        rx2, ry2 = min(w, rx2), min(h, ry2)
        if rx2 > rx1 and ry2 > ry1:
            mask[ry1:ry2, rx1:rx2] = False

        bg_L = lab[:, :, 0][mask]
        bg_S = hsv[:, :, 1][mask]

        mean_L = np.mean(bg_L) if len(bg_L) > 0 else 128.0
        mean_S = np.mean(bg_S) if len(bg_S) > 0 else 128.0

        return [mean_L / 255.0, mean_S / 255.0]

    def _extract_composition(self, img_bgr):
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        h, w = lab.shape[:2]

        sky_region = lab[:int(h * 0.3), :, :]
        ground_region = lab[int(h * 0.6):, :, :]

        sky_mean = np.mean(sky_region, axis=(0, 1)) if sky_region.size > 0 else np.array([128, 128, 128])
        ground_mean = np.mean(ground_region, axis=(0, 1)) if ground_region.size > 0 else np.array([128, 128, 128])

        return np.concatenate([sky_mean / 255.0, ground_mean / 255.0])
