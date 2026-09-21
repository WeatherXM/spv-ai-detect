import os
import sys
import json
import argparse
import logging
import yaml
from pathlib import Path

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("MatchCutPipeline")

def load_config(config_path):
    cfg_path = Path(config_path).resolve()
    base_dir = cfg_path.parent
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)
    if "paths" in config:
        for k, v in config["paths"].items():
            p = Path(v)
            if not p.is_absolute():
                config["paths"][k] = str(base_dir / p)
    return config

def main():
    parser = argparse.ArgumentParser(description="Weather Station Match-Cut Video Production Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--stage", type=str, default="all", choices=["detect", "normalize", "feature", "optimize", "render", "all"], help="Stage to execute")
    parser.add_argument("--force", action="store_true", help="Force recomputation of stage outputs")
    parser.add_argument("--suffix", type=str, default="", help="Optional suffix to append to output video filenames")
    parser.add_argument("--resolution", type=str, default="both", choices=["both", "1080p", "720p"], help="Video resolution profile: 1080p, 720p, or both")
    parser.add_argument("--fps", type=int, default=None, help="Base video framerate/speed in frames per second (e.g. 15, 24, 30)")
    parser.add_argument("--treatments", type=str, default=None, help="Comma-separated treatments to render (e.g. C or C,D)")
    parser.add_argument("--parts", type=int, default=None, help="Split video sequence into N equal parts (e.g. 3)")
    
    args = parser.parse_args()

    config_file = Path(args.config)
    if not config_file.exists():
        logger.error(f"Config file not found: {config_file}")
        sys.exit(1)

    config = load_config(config_file)
    if args.fps is not None and args.fps > 0:
        if "rendering" not in config:
            config["rendering"] = {}
        config["rendering"]["fps"] = args.fps
        if "canvas" not in config:
            config["canvas"] = {}
        config["canvas"]["fps"] = args.fps
        logger.info(f"Custom Render Framerate / Speed Override: {args.fps} fps")
    if args.parts is not None and args.parts > 0:
        if "rendering" in config and "resolutions" in config["rendering"]:
            for r in config["rendering"]["resolutions"]:
                r["parts"] = args.parts
        logger.info(f"Custom Video Split Override: {args.parts} parts")
    logger.info("==================================================================")
    logger.info("       WEATHER STATION MATCH-CUT VIDEO PIPELINE LAUNCHED          ")
    logger.info("==================================================================")
    logger.info(f"Input Directory: {config['paths']['input_dir']}")
    logger.info(f"Canvas Resolution: {config['canvas']['width']}x{config['canvas']['height']} @ {config['canvas']['fps']} fps")
    logger.info(f"Target Alignment: Center ({config['alignment']['target_center_x']*100:.0f}%, {config['alignment']['target_center_y']*100:.0f}%), Height ({config['alignment']['target_height_pct']*100:.0f}%)")
    logger.info("==================================================================")

    from pipeline.detector import StationDetector
    from pipeline.normalizer import ImageNormalizer
    from pipeline.feature_extractor import FeatureExtractor
    from pipeline.sequence_optimizer import SequenceOptimizer
    from pipeline.video_renderer import VideoRenderer

    detector = StationDetector(config)
    normalizer = ImageNormalizer(config)
    feature_extractor = FeatureExtractor(config)
    optimizer = SequenceOptimizer(config)
    renderer = VideoRenderer(config)

    input_dir = config["paths"]["input_dir"]

    # FAST-PATH FOR STANDALONE RENDER STAGE
    if args.stage == "render":
        override_file = Path(config["paths"]["cache_dir"]) / "sequence_override.json"
        seq_file = Path(config["paths"]["cache_dir"]) / "sequence.json"
        sequence = None
        if override_file.exists():
            try:
                with open(override_file, "r") as f:
                    data = json.load(f)
                    sequence = data.get("sequence", [])
                    if sequence:
                        logger.info(f"Loaded sequence override with {len(sequence)} photos for rendering.")
            except Exception as e:
                logger.warning(f"Could not load sequence override: {e}")

        if not sequence and seq_file.exists():
            try:
                with open(seq_file, "r") as f:
                    data = json.load(f)
                    sequence = data.get("sequence", [])
                    if sequence:
                        logger.info(f"Loaded cached sequence with {len(sequence)} photos for rendering.")
            except Exception as e:
                logger.warning(f"Could not load cached sequence: {e}")

        if sequence:
            logger.info("\n--- STAGE 5: FFMPEG MULTI-TREATMENT VIDEO RENDERING ---")
            suffix = getattr(args, "suffix", "")
            resolution = getattr(args, "resolution", "both")
            treatments = getattr(args, "treatments", None)
            outputs = renderer.render_all_treatments(sequence, suffix=suffix, resolutions=resolution, treatments=treatments)
            logger.info("\n==================================================================")
            logger.info("                   SUCCESSFULLY GENERATED VIDEOS                 ")
            logger.info("==================================================================")
            for treatment, filepath in outputs.items():
                logger.info(f" Treatment {treatment}: {filepath}")
            logger.info("==================================================================")
            return
    if args.stage in ["detect", "all"]:
        logger.info("\n--- STAGE 1: STATION OBJECT DETECTION ---")
        detections = detector.detect_folder(input_dir, force_recompute=args.force)
    else:
        detections = detector.detect_folder(input_dir, force_recompute=False)

    # STAGE 2: Normalization
    if args.stage in ["normalize", "all"]:
        logger.info("\n--- STAGE 2: FRAMING NORMALIZATION & REGISTRATION ---")
        normalized_registry = normalizer.normalize_folder(input_dir, detections, force_recompute=args.force)
        if args.stage == "normalize":
            logger.info("Normalization complete.")
            return
    else:
        normalized_registry = normalizer.normalize_folder(input_dir, detections, force_recompute=False)

    # STAGE 3: Feature Extraction & Distance Matrix
    if args.stage in ["feature", "all"]:
        logger.info("\n--- STAGE 3: VISUAL FEATURE EXTRACTION & COST MATRIX ---")
        filenames, dist_matrix = feature_extractor.extract_and_build_matrix(normalized_registry, force_recompute=args.force)
        if args.stage == "feature":
            return
    else:
        filenames, dist_matrix = feature_extractor.extract_and_build_matrix(normalized_registry, force_recompute=False)

    # STAGE 4: TSP Sequence Optimization
    if args.stage in ["optimize", "all"]:
        logger.info("\n--- STAGE 4: MATCH-CUT SEQUENCE OPTIMIZATION (TSP) ---")
        sequence = optimizer.optimize_sequence(filenames, dist_matrix, force_recompute=args.force)
        if args.stage == "optimize":
            return
    else:
        sequence = optimizer.optimize_sequence(filenames, dist_matrix, force_recompute=False)

    # STAGE 5: FFmpeg Video Rendering
    if args.stage in ["render", "all"]:
        logger.info("\n--- STAGE 5: FFMPEG MULTI-TREATMENT VIDEO RENDERING ---")
        suffix = getattr(args, "suffix", "")
        resolution = getattr(args, "resolution", "both")
        outputs = renderer.render_all_treatments(sequence, suffix=suffix, resolutions=resolution)
        
        logger.info("\n==================================================================")
        logger.info("                   SUCCESSFULLY GENERATED VIDEOS                 ")
        logger.info("==================================================================")
        for treatment, filepath in outputs.items():
            logger.info(f" Treatment {treatment}: {filepath}")
        logger.info("==================================================================")

if __name__ == "__main__":
    main()
