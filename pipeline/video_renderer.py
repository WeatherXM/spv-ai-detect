import os
import time
import json
import subprocess
import logging
import threading
from pathlib import Path
import cv2

logger = logging.getLogger(__name__)

class VideoRenderer:
    def __init__(self, config):
        self.config = config
        self.cache_dir = Path(config["paths"]["cache_dir"])
        self.norm_dir = self.cache_dir / "normalized_images"
        self.natural_dir = self.cache_dir / "natural_images"
        self.output_dir = Path(config["paths"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Real-time progress files
        self.progress_file = self.cache_dir / "render_progress.json"
        self.ffmpeg_progress_file = self.cache_dir / "ffmpeg_progress.txt"
        
        # Resolution profiles: 1080p (Master / High Quality) and 720p (Web-Optimized)
        rend_conf = config.get("rendering", {})
        default_res = [
            {
                "name": "1080p",
                "width": 1920,
                "height": 1080,
                "bitrate": "10000k",
                "bitrate_max": "15000k",
                "bitrate_bufsize": "20000k",
                "webm_bitrate": "6500k",
                "webm_crf": "20",
                "label": "1080p Full HD Master (High Quality, ~10 Mbps)",
            },
            {
                "name": "720p",
                "width": 1280,
                "height": 720,
                "parts": 3,
                "bitrate": "2000k",
                "bitrate_max": "3200k",
                "bitrate_bufsize": "4500k",
                "webm_bitrate": "1400k",
                "webm_crf": "24",
                "label": "720p HD Web (Optimized Web Background, ~2.0 Mbps)",
            }
        ]
        conf_res = rend_conf.get("resolutions", default_res)
        self.resolution_profiles = {r["name"]: r for r in conf_res}
        
        self.fps = rend_conf.get("fps", 24)
        self.formats = rend_conf.get("formats", ["mp4", "webm"])
        self.h264_codec = rend_conf.get("h264_codec", rend_conf.get("codec", "h264_videotoolbox"))
        self.webm_codec = rend_conf.get("webm_codec", "libvpx-vp9")
        self.treatments = rend_conf.get("treatments", ["A", "B", "C", "D", "E"])

        # Default legacy fallback fields
        p720 = self.resolution_profiles.get("720p", default_res[1])
        self.width = p720.get("width", 1280)
        self.height = p720.get("height", 720)
        self.bitrate = p720.get("bitrate", "2000k")
        self.bitrate_max = p720.get("bitrate_max", "3200k")
        self.bitrate_bufsize = p720.get("bitrate_bufsize", "4500k")

    def _update_progress(self, step, total_steps, treatment, format_name, label, sub_label="", step_pct=0.0, error=None, completed=False):
        if completed:
            pct = 100.0
        elif error is not None:
            pct = 0.0
        else:
            pct = round(((max(0, step - 1) + step_pct) / max(1, total_steps)) * 100, 1)
        pct = max(0.0, min(100.0, pct))

        status = {
            "running": not completed and error is None,
            "completed": completed,
            "step": step,
            "total_steps": total_steps,
            "treatment": treatment,
            "format": format_name,
            "label": label,
            "sub_label": sub_label,
            "percent": pct,
            "error": error
        }
        try:
            tmp = self.progress_file.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(status, f)
            tmp.replace(self.progress_file)
        except Exception:
            pass

    def _parse_ffmpeg_progress(self):
        frame = 0
        speed = ""
        fps = ""
        if self.ffmpeg_progress_file.exists():
            try:
                with open(self.ffmpeg_progress_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("frame="):
                            val = line.split("=", 1)[1].strip()
                            if val.isdigit():
                                frame = int(val)
                        elif line.startswith("speed="):
                            speed = line.split("=", 1)[1].strip()
                        elif line.startswith("fps="):
                            fps = line.split("=", 1)[1].strip()
            except Exception:
                pass
        return frame, speed, fps

    def _run_ffmpeg_with_progress(self, cmd, expected_frames, step, total_steps, treatment, fmt, label):
        if self.ffmpeg_progress_file.exists():
            try:
                self.ffmpeg_progress_file.unlink()
            except Exception:
                pass

        cmd_with_prog = list(cmd[:-1]) + ["-progress", str(self.ffmpeg_progress_file)] + [cmd[-1]]
        self._update_progress(
            step=step,
            total_steps=total_steps,
            treatment=treatment,
            format_name=fmt,
            label=label,
            sub_label=f"Starting {fmt.upper()} encoder...",
            step_pct=0.0
        )

        stderr_file = self.cache_dir / "ffmpeg_last_stderr.log"
        with open(stderr_file, "w") as err_f:
            p = subprocess.Popen(cmd_with_prog, stdout=subprocess.DEVNULL, stderr=err_f)

            while p.poll() is None:
                time.sleep(0.15)
                frame, speed, fps = self._parse_ffmpeg_progress()
                step_pct = min(0.99, max(0.0, frame / expected_frames)) if expected_frames > 0 else 0.5
                speed_str = f" • Speed: {speed}" if speed and speed != "N/A" else ""
                sub = f"Frame {frame:,} / {expected_frames:,}{speed_str}"
                self._update_progress(
                    step=step,
                    total_steps=total_steps,
                    treatment=treatment,
                    format_name=fmt,
                    label=label,
                    sub_label=sub,
                    step_pct=step_pct
                )

        if p.returncode != 0:
            err_text = ""
            if stderr_file.exists():
                try:
                    err_text = stderr_file.read_text(errors="ignore")
                except Exception:
                    pass
            # Try libx264 fallback if h264_videotoolbox failed
            if self.h264_codec in cmd:
                logger.warning(f"FFmpeg returned code {p.returncode}. Retrying with fallback codec libx264...")
                cmd_fallback = [c if c != self.h264_codec else "libx264" for c in cmd]
                cmd_fallback_prog = list(cmd_fallback[:-1]) + ["-progress", str(self.ffmpeg_progress_file)] + [cmd_fallback[-1]]
                with open(stderr_file, "w") as err_f2:
                    p2 = subprocess.Popen(cmd_fallback_prog, stdout=subprocess.DEVNULL, stderr=err_f2)
                    while p2.poll() is None:
                        time.sleep(0.15)
                        frame, speed, fps = self._parse_ffmpeg_progress()
                        step_pct = min(0.99, max(0.0, frame / expected_frames)) if expected_frames > 0 else 0.5
                        sub = f"Frame {frame:,} / {expected_frames:,} (libx264 fallback)"
                        self._update_progress(
                            step=step,
                            total_steps=total_steps,
                            treatment=treatment,
                            format_name=fmt,
                            label=label,
                            sub_label=sub,
                            step_pct=step_pct
                        )
                if p2.returncode != 0:
                    fallback_err = ""
                    if stderr_file.exists():
                        try:
                            fallback_err = stderr_file.read_text(errors="ignore")
                        except Exception:
                            pass
                    raise RuntimeError(f"FFmpeg fallback failed with code {p2.returncode}: {fallback_err}")
            else:
                raise RuntimeError(f"FFmpeg failed with code {p.returncode}: {err_text}")

        self._update_progress(
            step=step,
            total_steps=total_steps,
            treatment=treatment,
            format_name=fmt,
            label=label,
            sub_label=f"Completed {fmt.upper()}",
            step_pct=1.0
        )

    @staticmethod
    def split_sequence_into_parts(seq, num_parts=3):
        """Splits a photo sequence into num_parts contiguous slices of approximately equal duration/photos."""
        n = len(seq)
        if num_parts <= 1 or n == 0:
            return [(1, seq)]
        if n <= num_parts:
            return [(i + 1, [seq[i]]) for i in range(n)]
        parts = []
        for i in range(num_parts):
            start = (i * n) // num_parts
            end = ((i + 1) * n) // num_parts
            sub_seq = seq[start:end]
            if sub_seq:
                parts.append((i + 1, sub_seq))
        return parts

    def render_all_treatments(self, sequence_filenames, suffix="", resolutions=None, treatments=None):
        N = len(sequence_filenames)
        if N == 0:
            logger.error("No images provided for video rendering!")
            self._update_progress(0, 10, "", "", "No images provided for video rendering!", error="Empty sequence", completed=False)
            return {}

        # Resolve requested resolutions
        if resolutions is None or resolutions == "both":
            active_res = ["1080p", "720p"]
        elif isinstance(resolutions, str):
            active_res = ["1080p", "720p"] if resolutions == "both" else [resolutions]
        else:
            active_res = list(resolutions)

        # Validate against known profiles
        active_res = [r for r in active_res if r in self.resolution_profiles]
        if not active_res:
            active_res = ["1080p", "720p"]

        # Format tag part if provided (e.g. '_zoom_f2n')
        tag_clean = f"_{suffix.lstrip('_')}" if suffix else ""

        # Use natural_images (uncorrected zoom centering) for far-to-near and near-to-far animations
        is_natural = suffix.lstrip('_') in ["zoom_f2n", "zoom_n2f"] if suffix else False
        active_img_dir = self.natural_dir if (is_natural and self.natural_dir.exists()) else self.norm_dir
        logger.info(f"Using source image directory: {active_img_dir} (is_natural={is_natural})")

        pace_A = round(self.fps / 3, 1)
        pace_B = self.fps
        pace_C = round(self.fps / 8, 1)
        pace_D = round(self.fps / 4, 1)
        pace_E = round(self.fps / 2, 1)
        treatments_meta = {
            "A": {"desc": f"3 frames/photo rhythmic beat ({pace_A} photos/sec @ {self.fps}fps)", "fpi": 3},
            "B": {"desc": f"1 frame/photo hyper-speed cut ({pace_B} photos/sec @ {self.fps}fps)", "fpi": 1},
            "C": {"desc": f"Cinematic slow push-in (Ken Burns @ {self.fps}fps, {pace_C} photos/sec [8 frames/photo])", "fpi": 8},
            "D": {"desc": f"Subtle push-in motion (Ken Burns @ {self.fps}fps, {pace_D} photos/sec [4 frames/photo])", "fpi": 4},
            "E": {"desc": f"Station fixed match-cut ({pace_E} photos/sec pace @ {self.fps}fps)", "fpi": 2},
        }

        # Resolve requested treatments
        if treatments:
            if isinstance(treatments, str):
                requested = [t.strip().upper() for t in treatments.split(",") if t.strip()]
            else:
                requested = [str(t).strip().upper() for t in treatments if str(t).strip()]
            active_treatments = [t for t in requested if t in treatments_meta]
        else:
            active_treatments = [t for t in self.treatments if t in treatments_meta]

        if not active_treatments:
            active_treatments = [t for t in self.treatments if t in treatments_meta] or ["C"]

        logger.info(
            f"Rendering match-cut video treatments {active_treatments} for {N} images across resolutions {active_res} "
            f"(@ {self.fps}fps, formats={self.formats}, suffix='{suffix}')..."
        )
        outputs = {}

        has_webm = "webm" in self.formats
        total_steps = 0
        for r_name in active_res:
            r_cfg = self.resolution_profiles[r_name]
            n_parts = r_cfg.get("parts", 1)
            if r_name == "720p":
                steps_per_part = (3 if not has_webm else 5)
            else:
                steps_per_part = (2 if not has_webm else 3)
            total_steps += len(active_treatments) * n_parts * steps_per_part
        current_step = 0

        for treatment in active_treatments:
            version_dir = self.output_dir / f"Version_{treatment}"
            version_dir.mkdir(parents=True, exist_ok=True)
            meta = treatments_meta.get(treatment, {"desc": "Match-cut", "fpi": 2})

            for res_name in active_res:
                res_cfg = self.resolution_profiles[res_name]
                w, h = res_cfg["width"], res_cfg["height"]
                br = res_cfg["bitrate"]
                num_parts = res_cfg.get("parts", 1)
                parts = self.split_sequence_into_parts(sequence_filenames, num_parts)

                if res_name == "720p":
                    for part_idx, part_seq in parts:
                        expected_frames = len(part_seq) * meta["fpi"]
                        part_dir = version_dir / f"Part_{part_idx}" if num_parts > 1 else version_dir
                        part_dir.mkdir(parents=True, exist_ok=True)
                        part_desc_suffix = f" (Part {part_idx}/{len(parts)}, {len(part_seq)} photos)" if num_parts > 1 else ""
                        out_key_part = f" Part {part_idx}" if num_parts > 1 else ""

                        # 1. Desktop 720p MP4
                        current_step += 1
                        out_mp4 = part_dir / "station_matchcut_720p.mp4"
                        mp4_label = f"Version {treatment}{out_key_part} (Desktop 720p MP4): {meta['desc']}{part_desc_suffix}"
                        logger.info(f"Rendering Treatment {treatment}{out_key_part} ({res_name}) -> {out_mp4.name} ({w}x{h} @ {self.fps}fps, ~{br})...")

                        if treatment == "E":
                            self._render_hardcuts(part_seq, out_mp4, frames_per_image=2, img_dir=active_img_dir,
                                                  expected_frames=expected_frames, step=current_step, total_steps=total_steps,
                                                  treatment=treatment, label=mp4_label, res_cfg=res_cfg)
                        elif treatment == "A":
                            self._render_hardcuts(part_seq, out_mp4, frames_per_image=3, img_dir=active_img_dir,
                                                  expected_frames=expected_frames, step=current_step, total_steps=total_steps,
                                                  treatment=treatment, label=mp4_label, res_cfg=res_cfg)
                        elif treatment == "B":
                            self._render_hardcuts(part_seq, out_mp4, frames_per_image=1, img_dir=active_img_dir,
                                                  expected_frames=expected_frames, step=current_step, total_steps=total_steps,
                                                  treatment=treatment, label=mp4_label, res_cfg=res_cfg)
                        elif treatment == "C":
                            self._render_push_in(part_seq, out_mp4, frame_duration=8, img_dir=active_img_dir,
                                                 expected_frames=expected_frames, step=current_step, total_steps=total_steps,
                                                 treatment=treatment, label=mp4_label, res_cfg=res_cfg)
                        elif treatment == "D":
                            self._render_push_in(part_seq, out_mp4, frame_duration=4, img_dir=active_img_dir,
                                                 expected_frames=expected_frames, step=current_step, total_steps=total_steps,
                                                 treatment=treatment, label=mp4_label, res_cfg=res_cfg)

                        outputs[f"{treatment}{out_key_part} (Desktop 720p MP4)"] = str(out_mp4)

                        # 2. Desktop 720p WebM
                        if has_webm:
                            current_step += 1
                            out_webm = part_dir / "station_matchcut_720p.webm"
                            webm_label = f"Version {treatment}{out_key_part} (Desktop 720p WebM VP9): High-efficiency browser delivery{part_desc_suffix}"
                            self._convert_to_webm(out_mp4, out_webm, expected_frames=expected_frames,
                                                  step=current_step, total_steps=total_steps,
                                                  treatment=treatment, label=webm_label, res_cfg=res_cfg)
                            outputs[f"{treatment}{out_key_part} (Desktop 720p WebM)"] = str(out_webm)

                        # 3. Mobile 480p MP4
                        current_step += 1
                        out_mob_mp4 = part_dir / "station_matchcut_mobile.mp4"
                        mob_mp4_label = f"Version {treatment}{out_key_part} (Mobile 480p MP4): Universal mobile stream{part_desc_suffix}"
                        self._render_mobile_mp4(out_mp4, out_mob_mp4, expected_frames=expected_frames,
                                                step=current_step, total_steps=total_steps,
                                                treatment=treatment, label=mob_mp4_label)
                        outputs[f"{treatment}{out_key_part} (Mobile 480p MP4)"] = str(out_mob_mp4)

                        # 4. Mobile 480p WebM
                        if has_webm:
                            current_step += 1
                            out_mob_webm = part_dir / "station_matchcut_mobile.webm"
                            mob_webm_label = f"Version {treatment}{out_key_part} (Mobile 480p WebM VP9): Cellular bandwidth saver{part_desc_suffix}"
                            self._render_mobile_webm(out_mp4, out_mob_webm, expected_frames=expected_frames,
                                                     step=current_step, total_steps=total_steps,
                                                     treatment=treatment, label=mob_webm_label)
                            outputs[f"{treatment}{out_key_part} (Mobile 480p WebM)"] = str(out_mob_webm)

                        # 5. Poster Image WebP
                        current_step += 1
                        out_poster = part_dir / "station_matchcut-poster.webp"
                        poster_label = f"Version {treatment}{out_key_part} (WebP Poster): First paint placeholder"
                        self._update_progress(step=current_step, total_steps=total_steps,
                                              treatment=treatment, format_name="webp",
                                              label=poster_label, sub_label="Extracting frame 0 to WebP (Q85)...",
                                              step_pct=0.5)
                        self._extract_poster_image(out_mp4, out_poster, quality=85)
                        self._update_progress(step=current_step, total_steps=total_steps,
                                              treatment=treatment, format_name="webp",
                                              label=poster_label, sub_label="WebP poster extracted successfully",
                                              step_pct=1.0)
                        outputs[f"{treatment}{out_key_part} (WebP Poster)"] = str(out_poster)

                        # 6. Embed Snippet in Part folder
                        self._create_embed_assets(part_dir, video_name="station_matchcut")

                elif res_name == "1080p":
                    num_parts = res_cfg.get("parts", 1)
                    parts = self.split_sequence_into_parts(sequence_filenames, num_parts)

                    for part_idx, part_seq in parts:
                        expected_frames = len(part_seq) * meta["fpi"]
                        part_dir = version_dir / f"Part_{part_idx}" if num_parts > 1 else version_dir
                        part_dir.mkdir(parents=True, exist_ok=True)
                        part_desc_suffix = f" (Part {part_idx}/{len(parts)}, {len(part_seq)} photos)" if num_parts > 1 else ""
                        out_key_part = f" Part {part_idx}" if num_parts > 1 else ""

                        # 1. Master 1080p MP4
                        current_step += 1
                        out_1080_mp4 = part_dir / "station_matchcut_master_1080p.mp4"
                        mp4_label = f"Version {treatment}{out_key_part} (1080p Master MP4): Full HD master archive ({w}x{h}){part_desc_suffix}"
                        if treatment == "E":
                            self._render_hardcuts(part_seq, out_1080_mp4, frames_per_image=2, img_dir=active_img_dir,
                                                  expected_frames=expected_frames, step=current_step, total_steps=total_steps,
                                                  treatment=treatment, label=mp4_label, res_cfg=res_cfg)
                        elif treatment == "A":
                            self._render_hardcuts(part_seq, out_1080_mp4, frames_per_image=3, img_dir=active_img_dir,
                                                  expected_frames=expected_frames, step=current_step, total_steps=total_steps,
                                                  treatment=treatment, label=mp4_label, res_cfg=res_cfg)
                        elif treatment == "B":
                            self._render_hardcuts(part_seq, out_1080_mp4, frames_per_image=1, img_dir=active_img_dir,
                                                  expected_frames=expected_frames, step=current_step, total_steps=total_steps,
                                                  treatment=treatment, label=mp4_label, res_cfg=res_cfg)
                        elif treatment == "C":
                            self._render_push_in(part_seq, out_1080_mp4, frame_duration=8, img_dir=active_img_dir,
                                                 expected_frames=expected_frames, step=current_step, total_steps=total_steps,
                                                 treatment=treatment, label=mp4_label, res_cfg=res_cfg)
                        elif treatment == "D":
                            self._render_push_in(part_seq, out_1080_mp4, frame_duration=4, img_dir=active_img_dir,
                                                 expected_frames=expected_frames, step=current_step, total_steps=total_steps,
                                                 treatment=treatment, label=mp4_label, res_cfg=res_cfg)
                        outputs[f"{treatment}{out_key_part} (1080p Master MP4)"] = str(out_1080_mp4)

                        # 2. Master 1080p WebM
                        if has_webm:
                            current_step += 1
                            out_1080_webm = part_dir / "station_matchcut_master_1080p.webm"
                            webm_label = f"Version {treatment}{out_key_part} (1080p Master WebM VP9): Universal 1080p delivery{part_desc_suffix}"
                            self._convert_to_webm(out_1080_mp4, out_1080_webm, expected_frames=expected_frames,
                                                  step=current_step, total_steps=total_steps,
                                                  treatment=treatment, label=webm_label, res_cfg=res_cfg)
                            outputs[f"{treatment}{out_key_part} (1080p Master WebM)"] = str(out_1080_webm)

                        # 3. Master 1080p Poster
                        current_step += 1
                        out_1080_poster = part_dir / "station_matchcut_master-poster.webp"
                        poster_label = f"Version {treatment}{out_key_part} (1080p Poster): High-res master poster"
                        self._update_progress(step=current_step, total_steps=total_steps,
                                              treatment=treatment, format_name="webp",
                                              label=poster_label, sub_label="Extracting frame 0 to WebP (Q85)...",
                                              step_pct=0.5)
                        self._extract_poster_image(out_1080_mp4, out_1080_poster, quality=85)
                        self._update_progress(step=current_step, total_steps=total_steps,
                                              treatment=treatment, format_name="webp",
                                              label=poster_label, sub_label="1080p poster extracted successfully",
                                              step_pct=1.0)
                        outputs[f"{treatment}{out_key_part} (1080p WebP Poster)"] = str(out_1080_poster)
                        self._create_embed_assets(part_dir, video_name="station_matchcut_master")

            # Write README & DEPLOY_TO_WEBSITE.sh in version_dir
            self._write_version_overview(version_dir, treatment, meta)

        self._update_progress(
            step=total_steps,
            total_steps=total_steps,
            treatment="Done",
            format_name="",
            label=f"All treatments rendered at {', '.join(active_res)} successfully!",
            sub_label=f"{len(outputs)} web asset files ready in version folders.",
            step_pct=1.0,
            completed=True
        )
        logger.info("Video rendering complete!")
        return outputs

    def _render_hardcuts(self, sequence_filenames, out_file, frames_per_image=2, img_dir=None,
                         expected_frames=0, step=0, total_steps=10, treatment="", label="", res_cfg=None):
        res_cfg = res_cfg or self.resolution_profiles.get("720p", {})
        width = res_cfg.get("width", self.width)
        height = res_cfg.get("height", self.height)
        bitrate = res_cfg.get("bitrate", self.bitrate)
        bitrate_max = res_cfg.get("bitrate_max", self.bitrate_max)
        bitrate_bufsize = res_cfg.get("bitrate_bufsize", self.bitrate_bufsize)

        target_dir = img_dir or self.norm_dir
        concat_file = self.cache_dir / f"concat_{out_file.stem}.txt"
        duration_sec = frames_per_image / self.fps

        with open(concat_file, "w") as f:
            for fname in sequence_filenames:
                img_path = target_dir / fname
                f.write(f"file '{img_path.resolve()}'\n")
                f.write(f"duration {duration_sec:.4f}\n")
            if sequence_filenames:
                f.write(f"file '{(target_dir / sequence_filenames[-1]).resolve()}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-vf", f"scale={width}:{height}:flags=bicubic,fps={self.fps},format=yuv420p",
            "-c:v", self.h264_codec,
        ]
        if self.h264_codec == "h264_videotoolbox":
            cmd.extend(["-profile:v", "high", "-spatial_aq", "1"])
        elif self.h264_codec == "libx264":
            cmd.extend(["-preset", "medium", "-profile:v", "high"])

        cmd.extend([
            "-g", str(self.fps * 2),
            "-flags", "+cgop",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
            "-b:v", bitrate,
            "-maxrate", bitrate_max,
            "-bufsize", bitrate_bufsize,
            "-movflags", "+faststart",
            "-an",
            str(out_file)
        ])

        self._run_ffmpeg_with_progress(cmd, expected_frames=expected_frames, step=step,
                                       total_steps=total_steps, treatment=treatment,
                                       fmt="mp4", label=label)

    def _render_push_in(self, sequence_filenames, out_file, frame_duration=4, img_dir=None,
                        expected_frames=0, step=0, total_steps=8, treatment="D", label="", res_cfg=None):
        res_cfg = res_cfg or self.resolution_profiles.get("720p", {})
        width = res_cfg.get("width", self.width)
        height = res_cfg.get("height", self.height)
        bitrate = res_cfg.get("bitrate", self.bitrate)
        bitrate_max = res_cfg.get("bitrate_max", self.bitrate_max)
        bitrate_bufsize = res_cfg.get("bitrate_bufsize", self.bitrate_bufsize)

        is_720 = res_cfg.get("name") == "720p" or width == 1280
        is_1080 = res_cfg.get("name") == "1080p" or width == 1920

        if treatment == "C":
            # For 8-frame slow push-in, high temporal redundancy between adjacent zoom frames allows aggressive compression.
            # 720p is specifically tuned for website background video (<10 MB/part) with smooth playback.
            if is_720:
                bitrate = "1000k"
                bitrate_max = "1600k"
                bitrate_bufsize = "2400k"
            elif is_1080:
                bitrate = "5500k"
                bitrate_max = "8500k"
                bitrate_bufsize = "12000k"
        elif treatment == "D":
            # For 4-frame push-in
            if is_720:
                bitrate = "1600k"
                bitrate_max = "2600k"
                bitrate_bufsize = "3600k"
            elif is_1080:
                bitrate = "7500k"
                bitrate_max = "11000k"
                bitrate_bufsize = "15000k"

        target_dir = img_dir or self.norm_dir
        fpi = max(1, int(frame_duration))
        total_expected_frames = len(sequence_filenames) * fpi

        if self.ffmpeg_progress_file.exists():
            try:
                self.ffmpeg_progress_file.unlink()
            except Exception:
                pass

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "bgr24",
            "-r", str(self.fps),
            "-i", "-",
            "-c:v", self.h264_codec,
        ]
        if self.h264_codec == "h264_videotoolbox":
            cmd.extend(["-profile:v", "high", "-spatial_aq", "1"])
        elif self.h264_codec == "libx264":
            cmd.extend(["-preset", "medium", "-profile:v", "high"])

        cmd.extend([
            "-g", str(self.fps * 2),
            "-flags", "+cgop",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
            "-b:v", bitrate,
            "-maxrate", bitrate_max,
            "-bufsize", bitrate_bufsize,
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",
            "-progress", str(self.ffmpeg_progress_file),
            str(out_file)
        ])

        def _feed_pushin_frames(proc_target):
            try:
                zoom_factor = 0.055 if fpi >= 8 else 0.045
                for fname in sequence_filenames:
                    img_path = target_dir / fname
                    if not img_path.exists():
                        continue
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    ih, iw = img.shape[:2]
                    for s in range(fpi):
                        prog = s / (fpi - 1) if fpi > 1 else 0.0
                        scale = 1.0 + prog * zoom_factor
                        crop_w = int(iw / scale)
                        crop_h = int(ih / scale)
                        x1 = (iw - crop_w) // 2
                        y1 = (ih - crop_h) // 2
                        cropped = img[y1:y1 + crop_h, x1:x1 + crop_w]
                        resized = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_CUBIC)
                        proc_target.stdin.write(resized.tobytes())
            except (BrokenPipeError, OSError):
                pass
            finally:
                try:
                    proc_target.stdin.close()
                except Exception:
                    pass

        self._update_progress(
            step=step,
            total_steps=total_steps,
            treatment=treatment,
            format_name="mp4",
            label=label,
            sub_label=f"Encoding frame-accurate Ken Burns push-in ({fpi} fpi @ {self.fps}fps)...",
            step_pct=0.0
        )

        stderr_file = self.cache_dir / "ffmpeg_last_stderr.log"
        with open(stderr_file, "w") as err_f:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=err_f)
            feeder = threading.Thread(target=_feed_pushin_frames, args=(proc,))
            feeder.start()

            while proc.poll() is None:
                time.sleep(0.15)
                frame, speed, _ = self._parse_ffmpeg_progress()
                step_pct = min(0.99, max(0.0, frame / total_expected_frames)) if total_expected_frames > 0 else 0.5
                speed_str = f" • Speed: {speed}" if speed and speed != "N/A" else ""
                sub = f"Frame {frame:,} / {total_expected_frames:,}{speed_str}"
                self._update_progress(
                    step=step,
                    total_steps=total_steps,
                    treatment=treatment,
                    format_name="mp4",
                    label=label,
                    sub_label=sub,
                    step_pct=step_pct
                )
            feeder.join()
            proc.wait()

        if proc.returncode != 0:
            logger.warning(f"Push-in hardware encoder failed (code {proc.returncode}). Retrying with libx264...")
            cmd_fallback = [c if c != self.h264_codec else "libx264" for c in cmd]
            with open(stderr_file, "w") as err_f2:
                proc2 = subprocess.Popen(cmd_fallback, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=err_f2)
                feeder2 = threading.Thread(target=_feed_pushin_frames, args=(proc2,))
                feeder2.start()
                while proc2.poll() is None:
                    time.sleep(0.15)
                    frame, speed, _ = self._parse_ffmpeg_progress()
                    step_pct = min(0.99, max(0.0, frame / total_expected_frames)) if total_expected_frames > 0 else 0.5
                    sub = f"Frame {frame:,} / {total_expected_frames:,} (libx264 fallback)"
                    self._update_progress(
                        step=step,
                        total_steps=total_steps,
                        treatment=treatment,
                        format_name="mp4",
                        label=label,
                        sub_label=sub,
                        step_pct=step_pct
                    )
                feeder2.join()
                proc2.wait()
                if proc2.returncode != 0:
                    err_msg = stderr_file.read_text(errors="ignore") if stderr_file.exists() else "Unknown error"
                    raise RuntimeError(f"FFmpeg push-in render failed: {err_msg[:300]}")

    def _convert_to_webm(self, mp4_file, webm_file, expected_frames=0, step=0, total_steps=10,
                         treatment="", label="", res_cfg=None):
        res_cfg = res_cfg or self.resolution_profiles.get("720p", {})
        webm_bitrate = res_cfg.get("webm_bitrate", "1400k")
        webm_crf = str(res_cfg.get("webm_crf", "24"))

        is_720 = res_cfg.get("name") == "720p" or res_cfg.get("width") == 1280
        is_1080 = res_cfg.get("name") == "1080p" or res_cfg.get("width") == 1920

        if treatment == "C":
            # Ultra-efficient VP9 parameters for website background video
            if is_720:
                webm_bitrate = "700k"
                webm_crf = "28"
            elif is_1080:
                webm_bitrate = "3500k"
                webm_crf = "22"
        elif treatment == "D":
            if is_720:
                webm_bitrate = "1100k"
                webm_crf = "26"
            elif is_1080:
                webm_bitrate = "5000k"
                webm_crf = "20"
        elif is_720:
            webm_bitrate = "1400k"
            webm_crf = "24"
        elif is_1080:
            webm_bitrate = "6500k"
            webm_crf = "20"

        logger.info(f"Encoding WebM (VP9/AV1) -> {webm_file.name} (bitrate={webm_bitrate}, crf={webm_crf})...")
        cmd_vp9 = [
            "ffmpeg", "-y",
            "-i", str(mp4_file),
            "-c:v", self.webm_codec,
            "-b:v", webm_bitrate,
            "-crf", webm_crf,
            "-deadline", "realtime",
            "-cpu-used", "4",
            "-row-mt", "1",
            "-g", str(self.fps * 2),
            "-flags", "+cgop",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
            "-aq-mode", "1",
            "-pix_fmt", "yuv420p",
            "-an",
            str(webm_file)
        ]
        try:
            self._run_ffmpeg_with_progress(cmd=cmd_vp9, expected_frames=expected_frames, step=step,
                                           total_steps=total_steps, treatment=treatment,
                                           fmt="webm", label=label)
        except Exception as e:
            logger.warning(f"VP9 encoding failed ({e}). Retrying with SVT-AV1 encoder...")
            cmd_av1 = [
                "ffmpeg", "-y",
                "-i", str(mp4_file),
                "-c:v", "libsvtav1",
                "-b:v", webm_bitrate,
                "-preset", "8",
                "-flags", "+cgop",
                "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
                "-pix_fmt", "yuv420p",
                "-an",
                str(webm_file)
            ]
            self._run_ffmpeg_with_progress(cmd=cmd_av1, expected_frames=expected_frames, step=step,
                                           total_steps=total_steps, treatment=treatment,
                                           fmt="webm", label=f"Version {treatment} (WebM AV1)")

    def _render_mobile_mp4(self, mp4_file, out_file, expected_frames=0, step=0, total_steps=10,
                           treatment="", label=""):
        logger.info(f"Encoding Mobile MP4 (480p) -> {out_file.name} (bitrate=500k, 854x480)...")
        cmd = [
            "ffmpeg", "-y",
            "-i", str(mp4_file),
            "-vf", "scale=854:480:flags=bicubic",
            "-c:v", self.h264_codec,
        ]
        if self.h264_codec == "h264_videotoolbox":
            cmd.extend(["-profile:v", "high", "-spatial_aq", "1"])
        elif self.h264_codec == "libx264":
            cmd.extend(["-preset", "fast", "-profile:v", "high", "-crf", "28"])

        cmd.extend([
            "-b:v", "500k",
            "-maxrate", "800k",
            "-bufsize", "1200k",
            "-g", str(self.fps * 2),
            "-flags", "+cgop",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",
            str(out_file)
        ])
        self._run_ffmpeg_with_progress(cmd=cmd, expected_frames=expected_frames, step=step,
                                       total_steps=total_steps, treatment=treatment,
                                       fmt="mp4", label=label)

    def _render_mobile_webm(self, mp4_file, out_file, expected_frames=0, step=0, total_steps=10,
                            treatment="", label=""):
        logger.info(f"Encoding Mobile WebM (480p VP9) -> {out_file.name} (bitrate=450k, 854x480)...")
        cmd_vp9 = [
            "ffmpeg", "-y",
            "-i", str(mp4_file),
            "-vf", "scale=854:480:flags=bicubic",
            "-c:v", self.webm_codec,
            "-b:v", "450k",
            "-crf", "36",
            "-deadline", "realtime",
            "-cpu-used", "4",
            "-row-mt", "1",
            "-g", str(self.fps * 2),
            "-flags", "+cgop",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
            "-aq-mode", "1",
            "-pix_fmt", "yuv420p",
            "-an",
            str(out_file)
        ]
        try:
            self._run_ffmpeg_with_progress(cmd=cmd_vp9, expected_frames=expected_frames, step=step,
                                           total_steps=total_steps, treatment=treatment,
                                           fmt="webm", label=label)
        except Exception as e:
            logger.warning(f"Mobile VP9 encoding failed ({e}). Retrying with SVT-AV1 encoder...")
            cmd_av1 = [
                "ffmpeg", "-y",
                "-i", str(mp4_file),
                "-vf", "scale=854:480:flags=bicubic",
                "-c:v", "libsvtav1",
                "-b:v", "450k",
                "-preset", "8",
                "-pix_fmt", "yuv420p",
                "-an",
                str(out_file)
            ]
            self._run_ffmpeg_with_progress(cmd=cmd_av1, expected_frames=expected_frames, step=step,
                                           total_steps=total_steps, treatment=treatment,
                                           fmt="webm", label=label)

    def _extract_poster_image(self, video_path, poster_path, quality=85):
        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                raise ValueError("Could not read frame 0 from video")
            saved = cv2.imwrite(str(poster_path), frame, [cv2.IMWRITE_WEBP_QUALITY, quality])
            if not saved:
                from PIL import Image
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                img.save(str(poster_path), "WEBP", quality=quality)
            if poster_path.name == "station_matchcut-poster.webp":
                import shutil
                net_poster = poster_path.parent / "network-matchcut-poster.webp"
                shutil.copy2(poster_path, net_poster)
            logger.info(f"Generated WebP poster: {poster_path.name} ({poster_path.stat().st_size / 1024:.1f} KB)")
            return True
        except Exception as e:
            logger.warning(f"Failed to extract poster image: {e}")
            return False

    def _create_embed_assets(self, target_dir, video_name="station_matchcut"):
        html_content = f"""<!-- ============================================================== -->
<!-- WeatherXM Video Embed Snippets (Generated per docs/video.md)   -->
<!-- Copy assets to: weatherxmcom-website/public/assets/videos/     -->
<!-- ============================================================== -->

<!-- 1. Responsive HTML5 Video Pattern -->
<video
  class="w-full h-full object-cover"
  autoplay
  loop
  muted
  playsinline
  webkit-playsinline="true"
  preload="metadata"
  poster="/assets/videos/{video_name}-poster.webp"
  aria-hidden="true"
>
  <!-- Mobile Tier (<= 640px) -->
  <source media="(max-width: 640px)" src="/assets/videos/{video_name}_mobile.webm" type="video/webm" />
  <source media="(max-width: 640px)" src="/assets/videos/{video_name}_mobile.mp4" type="video/mp4" />

  <!-- Desktop Tier (> 640px) -->
  <source src="/assets/videos/{video_name}_720p.webm" type="video/webm" />
  <source src="/assets/videos/{video_name}_720p.mp4" type="video/mp4" />
</video>

<!-- 2. Astro StationVideoMesh Component Pattern -->
<!-- (Used in src/pages/network.astro) -->
<StationVideoMesh
  videoSrc="/assets/videos/{video_name}_720p.mp4"
  videoSrcWebm="/assets/videos/{video_name}_720p.webm"
  videoSrcMobile="/assets/videos/{video_name}_mobile.mp4"
  videoSrcMobileWebm="/assets/videos/{video_name}_mobile.webm"
  posterSrc="/assets/videos/{video_name}-poster.webp"
/>
"""
        embed_file = target_dir / "embed_snippet.html"
        try:
            embed_file.write_text(html_content, encoding="utf-8")
        except Exception as e:
            logger.warning(f"Could not write embed_snippet.html: {e}")

    def _write_version_overview(self, version_dir, treatment, meta):
        readme_path = version_dir / "README.md"
        deploy_sh = version_dir / "DEPLOY_TO_WEBSITE.sh"

        content = f"""# WeatherXM Video Assets — Version {treatment}

Generated in accordance with the website video specifications (`docs/video.md`).

## Treatment Description
- **Version {treatment}**: {meta.get('desc', 'Match-cut animation')}
- **Pacing**: {meta.get('fpi', 4)} frames/photo @ {self.fps} fps

## Asset Bundle Structure
All required website video assets are placed directly in this folder for direct copy/pasting into `weatherxmcom-website/public/assets/videos/`:

| Asset File | Target Specs | Codec & Container | Purpose |
| :--- | :--- | :--- | :--- |
| `station_matchcut_720p.mp4` | 1280×720, ~1.0 Mbps | H.264 High Profile (+faststart, -an, BT.709) | Universal desktop background video |
| `station_matchcut_720p.webm` | 1280×720, ~900 kbps | VP9 (-an, Closed GOP, BT.709) | Modern high-efficiency desktop browsers |
| `station_matchcut_mobile.mp4` | 854×480, ~500 kbps | H.264 Baseline (+faststart, -an, BT.709) | Universal mobile stream (iOS Safari / Android) |
| `station_matchcut_mobile.webm` | 854×480, ~450 kbps | VP9 (-an, Closed GOP, BT.709) | Cellular bandwidth-saver stream |
| `station_matchcut-poster.webp` | 1280×720, ~60 KB | WebP (Q85, Frame 0) | Instant first-paint & Data-Saver fallback |
| `network-matchcut-poster.webp` | 1280×720, ~60 KB | WebP (Q85, Frame 0) | Alias for network.astro poster attribute |

## One-Click Deployment to Website
To deploy these assets directly into `weatherxmcom-website`:
```bash
./DEPLOY_TO_WEBSITE.sh
```
Or run manually from terminal:
```bash
cp station_matchcut* network-matchcut* path/to/weatherxmcom-website/public/assets/videos/
```

## Frontend Integration
See `embed_snippet.html` in this folder for ready-to-paste Astro `<StationVideoMesh>` and HTML5 `<video>` elements.
"""
        try:
            readme_path.write_text(content, encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to write README.md: {e}")

        script_content = f"""#!/usr/bin/env bash
# Deploys Version {treatment} assets directly to WeatherXM Website
set -e
TARGET="${{WEBSITE_DIR:-../weatherxmcom-website}}/public/assets/videos"
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"

SOURCE_DIR="$SCRIPT_DIR"
if [ ! -f "$SOURCE_DIR/station_matchcut_720p.mp4" ] && [ -d "$SCRIPT_DIR/Part_1" ]; then
    SOURCE_DIR="$SCRIPT_DIR/Part_1"
fi

if [ ! -d "$TARGET" ]; then
    echo "❌ Error: Website directory $TARGET not found."
    exit 1
fi

echo "🚀 Deploying Version {treatment} assets from $SOURCE_DIR to $TARGET..."
cp -v "$SOURCE_DIR"/station_matchcut* "$TARGET/"
if [ -f "$SOURCE_DIR/network-matchcut-poster.webp" ]; then
    cp -v "$SOURCE_DIR/network-matchcut-poster.webp" "$TARGET/"
fi
echo "✅ Successfully deployed video assets to $TARGET!"
"""
        try:
            deploy_sh.write_text(script_content, encoding="utf-8")
            import os
            os.chmod(deploy_sh, 0o755)
        except Exception as e:
            logger.warning(f"Failed to write DEPLOY_TO_WEBSITE.sh: {e}")

