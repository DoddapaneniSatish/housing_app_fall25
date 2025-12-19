from __future__ import annotations

import json
import shutil
from pathlib import Path

# --------------------------------------------------
# PATHS
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

METRICS_DIR = ROOT / "metrics"
MODELS_DIR = ROOT / "models"

SUMMARY_PATH = METRICS_DIR / "summary_16_experiments.json"

BEST_MODEL_PATH = MODELS_DIR / "telco_churn_best_model.pkl"
BEST_META_PATH = MODELS_DIR / "telco_churn_best_model_meta.json"


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing summary file: {SUMMARY_PATH}")

    runs = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    if len(runs) < 16:
        print(f"⚠️ Warning: expected 16 runs, found {len(runs)}")

    # Select run with highest F1
    best_run = max(runs, key=lambda r: r["f1"])
    run_id = best_run["run_id"]

    source_model = MODELS_DIR / f"{run_id}.pkl"
    if not source_model.exists():
        raise FileNotFoundError(f"Model file not found: {source_model}")

    # Copy model for deployment
    shutil.copyfile(source_model, BEST_MODEL_PATH)

    # Save metadata
    BEST_META_PATH.write_text(
        json.dumps(best_run, indent=2),
        encoding="utf-8",
    )

    print("✅ Best Telco Churn model selected!")
    print(f"Run ID   : {run_id}")
    print(f"Best F1  : {best_run['f1']:.4f}")
    print(f"Model →  : {BEST_MODEL_PATH}")
    print(f"Meta  →  : {BEST_META_PATH}")


if __name__ == "__main__":
    main()
