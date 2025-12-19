import json
import os
from pathlib import Path
import mlflow

# --------------------------------------------------
# DAGSHUB CONFIG
# --------------------------------------------------
DAGSHUB_USER = "DoddapaneniSatish"
DAGSHUB_REPO = "telco-churn-ml"
EXPERIMENT_NAME = "telco_churn_16_experiments"

# --------------------------------------------------
# AUTHENTICATION (REQUIRED)
# --------------------------------------------------
if "DAGSHUB_TOKEN" not in os.environ:
    raise RuntimeError(
        "❌ DAGSHUB_TOKEN not found.\n\n"
        "Please set it first:\n"
        "$env:DAGSHUB_TOKEN='412c323c30f6cf4535e92330412f0faf0ffb1727'\n"
    )

# --------------------------------------------------
# LOAD SUMMARY FILE
# --------------------------------------------------
def load_runs(summary_path: Path):
    data = json.loads(summary_path.read_text(encoding="utf-8"))

    # Accept either {"runs": [...]} or just [...]
    if isinstance(data, dict) and "runs" in data and isinstance(data["runs"], list):
        return data["runs"]

    if isinstance(data, list):
        return data

    raise ValueError(
        "Unknown summary JSON format. Expected a list or a dict with key 'runs'."
    )

# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    summary_path = Path("metrics") / "summary_16_experiments.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path.resolve()}")

    tracking_uri = f"https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.mlflow"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    runs = load_runs(summary_path)
    print(f"🚀 Logging {len(runs)} Telco runs to DagsHub MLflow...")
    print(f"🔗 Tracking URI: {tracking_uri}")

    for r in runs:
        run_name = r.get("run_id", "run")
        model_type = r.get("model", "unknown")
        use_pca = int(bool(r.get("use_pca", False)))
        use_tuning = int(bool(r.get("use_tuning", False)))
        f1 = float(r.get("f1", 0.0))

        params = r.get("best_params", {}) or {}

        with mlflow.start_run(run_name=run_name):
            # Tags
            mlflow.set_tag("model", model_type)
            mlflow.set_tag("use_pca", use_pca)
            mlflow.set_tag("use_tuning", use_tuning)

            # Params
            mlflow.log_param("model", model_type)
            mlflow.log_param("use_pca", use_pca)
            mlflow.log_param("use_tuning", use_tuning)

            # Tuned hyperparameters (if any)
            for k, v in params.items():
                mlflow.log_param(str(k), str(v))

            # Metric
            mlflow.log_metric("f1", f1)

    print("✅ SUCCESS: All 16 Telco experiments logged to DagsHub!")
    print("👉 Open: DagsHub → Experiments → MLflow to verify.")

if __name__ == "__main__":
    main()
