import time
import json
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "cache"
WEIGHTS_DIR = BASE_DIR / "weights"
CSV_PATH = WEIGHTS_DIR / "weatherxm_sensor_head" / "results.csv"
TRAINING_LOG = WEIGHTS_DIR / "training.log"
PROGRESS_FILE = CACHE_DIR / "live_training_progress.json"
BEST_WEIGHTS = WEIGHTS_DIR / "weatherxm_sensor_head_best.pt"

def get_progress():
    data = {
        "running": True,
        "is_completed": False,
        "epoch": 1,
        "total_epochs": 35,
        "batch_str": "0/13",
        "batch_pct": 0.0,
        "overall_pct": 0.0,
        "eta_str": "Calculating...",
        "time_per_epoch": 35.0,
        "map50": 0.0,
        "map50_95": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "box_loss": 0.0,
        "cls_loss": 0.0,
        "recent_logs": []
    }

    # Check if final weights exist
    if BEST_WEIGHTS.exists():
        data["running"] = False
        data["is_completed"] = True
        data["overall_pct"] = 100.0
        data["epoch"] = 35
        data["eta_str"] = "Training Complete!"

    # Parse CSV for epoch-level metrics
    epoch_times = []
    if CSV_PATH.exists():
        try:
            lines = CSV_PATH.read_text().strip().split("\n")
            if len(lines) > 1:
                headers = [h.strip() for h in lines[0].split(",")]
                last_row = [v.strip() for v in lines[-1].split(",")]
                row_dict = dict(zip(headers, last_row))
                
                ep = int(row_dict.get("epoch", 1))
                data["epoch"] = ep
                data["map50"] = round(float(row_dict.get("metrics/mAP50(B)", 0)) * 100, 1)
                data["map50_95"] = round(float(row_dict.get("metrics/mAP50-95(B)", 0)) * 100, 1)
                data["precision"] = round(float(row_dict.get("metrics/precision(B)", 0)) * 100, 1)
                data["recall"] = round(float(row_dict.get("metrics/recall(B)", 0)) * 100, 1)
                data["box_loss"] = round(float(row_dict.get("train/box_loss", 0)), 3)
                data["cls_loss"] = round(float(row_dict.get("train/cls_loss", 0)), 3)

                for l in lines[1:]:
                    vals = [v.strip() for v in l.split(",")]
                    r = dict(zip(headers, vals))
                    if "time" in r:
                        epoch_times.append(float(r["time"]))
        except Exception:
            pass

    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 35.0
    data["time_per_epoch"] = round(avg_epoch_time, 1)

    # Check training logs
    log_file = None
    if TRAINING_LOG.exists() and TRAINING_LOG.stat().st_size > 0:
        log_file = TRAINING_LOG

    if log_file and log_file.exists():
        try:
            log_text = log_file.read_text()
            matches = list(re.finditer(r'(\d+)/35.*?(\d+)/(\d+).*?(\d+s<[\d:]+|\d+\.\d+s/it)', log_text))
            if matches:
                last_m = matches[-1]
                cur_epoch = int(last_m.group(1))
                cur_batch = int(last_m.group(2))
                total_batches = int(last_m.group(3))
                data["epoch"] = cur_epoch
                data["batch_str"] = f"{cur_batch}/{total_batches}"
                data["batch_pct"] = round((cur_batch / float(total_batches)) * 100, 1)
                
                completed_progress = (cur_epoch - 1) + (cur_batch / float(total_batches))
                overall_pct = min(100.0, max(0.0, (completed_progress / 35.0) * 100.0))
                data["overall_pct"] = round(overall_pct, 1)

                remaining_epochs = 35.0 - completed_progress
                remaining_seconds = remaining_epochs * avg_epoch_time
                if remaining_seconds > 60:
                    mins = int(remaining_seconds // 60)
                    secs = int(remaining_seconds % 60)
                    data["eta_str"] = f"~{mins}m {secs}s remaining"
                else:
                    data["eta_str"] = f"~{int(remaining_seconds)}s remaining"

            # Check if training finished in logs
            if "Model training complete!" in log_text or "Results saved to" in log_text and data["epoch"] >= 35:
                data["running"] = False
                data["is_completed"] = True
                data["overall_pct"] = 100.0
                data["eta_str"] = "Training Complete!"

            # Recent relevant log lines cleaned of ANSI escape codes
            ansi_escape = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')
            all_lines = [ansi_escape.sub('', l).strip() for l in log_text.split("\n") if l.strip()]
            relevant = [l for l in all_lines if "640:" in l or "1024:" in l or "mAP" in l or "Epoch" in l or "all" in l][-6:]
            data["recent_logs"] = relevant
        except Exception:
            pass

    return data

def main():
    CACHE_DIR.mkdir(exist_ok=True)
    while True:
        try:
            p = get_progress()
            with open(PROGRESS_FILE, "w") as f:
                json.dump(p, f, indent=2)
            if p["is_completed"]:
                break
        except Exception:
            pass
        time.sleep(2)

if __name__ == "__main__":
    main()
