import sqlite3
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# PATHS
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
DB_PATH = ROOT / "data" / "db" / "telco_churn.sqlite3"
SCHEMA_PATH = ROOT / "src" / "db_schema.sql"

# --------------------------------------------------
# DIMENSION MAPPING
# --------------------------------------------------
DIMENSIONS = {
    "gender": ("dim_gender", "gender"),
    "Contract": ("dim_contract", "contract_type"),
    "PaymentMethod": ("dim_payment", "payment_method"),
    "InternetService": ("dim_internet", "internet_service"),

    "Partner": ("dim_yesno", "value"),
    "Dependents": ("dim_yesno", "value"),
    "PhoneService": ("dim_yesno", "value"),
    "PaperlessBilling": ("dim_yesno", "value"),
    "MultipleLines": ("dim_yesno", "value"),
    "OnlineSecurity": ("dim_yesno", "value"),
    "OnlineBackup": ("dim_yesno", "value"),
    "DeviceProtection": ("dim_yesno", "value"),
    "TechSupport": ("dim_yesno", "value"),
    "StreamingTV": ("dim_yesno", "value"),
    "StreamingMovies": ("dim_yesno", "value"),
}

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------
def insert_unique(conn, table, col, values):
    unique_vals = sorted(pd.Series(values).fillna("Unknown").astype(str).unique())
    conn.executemany(
        f"INSERT OR IGNORE INTO {table} ({col}) VALUES (?)",
        [(v,) for v in unique_vals],
    )

def fetch_mapping(conn, table, col):
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    id_col = df.columns[0]
    return dict(zip(df[col], df[id_col]))

# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load CSV
    df = pd.read_csv(CSV_PATH)

    # Clean / convert
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    # Build database
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")

        # Create schema
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))

        # Insert dimension values
        for col, (table, dim_col) in DIMENSIONS.items():
            insert_unique(conn, table, dim_col, df[col])

        # Build mappings
        maps = {
            col: fetch_mapping(conn, table, dim_col)
            for col, (table, dim_col) in DIMENSIONS.items()
        }

        # Build fact table
        fact = pd.DataFrame({
            "customer_id": df["customerID"],
            "tenure": df["tenure"],
            "monthly_charges": df["MonthlyCharges"],
            "total_charges": df["TotalCharges"],
            "senior_citizen": df["SeniorCitizen"],

            "gender_id": df["gender"].map(maps["gender"]),
            "contract_id": df["Contract"].map(maps["Contract"]),
            "payment_id": df["PaymentMethod"].map(maps["PaymentMethod"]),
            "internet_id": df["InternetService"].map(maps["InternetService"]),

            "partner_id": df["Partner"].map(maps["Partner"]),
            "dependents_id": df["Dependents"].map(maps["Dependents"]),
            "phone_service_id": df["PhoneService"].map(maps["PhoneService"]),
            "paperless_billing_id": df["PaperlessBilling"].map(maps["PaperlessBilling"]),

            "multiple_lines_id": df["MultipleLines"].map(maps["MultipleLines"]),
            "online_security_id": df["OnlineSecurity"].map(maps["OnlineSecurity"]),
            "online_backup_id": df["OnlineBackup"].map(maps["OnlineBackup"]),
            "device_protection_id": df["DeviceProtection"].map(maps["DeviceProtection"]),
            "tech_support_id": df["TechSupport"].map(maps["TechSupport"]),
            "streaming_tv_id": df["StreamingTV"].map(maps["StreamingTV"]),
            "streaming_movies_id": df["StreamingMovies"].map(maps["StreamingMovies"]),

            "churn": df["Churn"],
        })

        fact.to_sql("fact_churn", conn, if_exists="append", index=False)

        rows = conn.execute("SELECT COUNT(*) FROM fact_churn").fetchone()[0]

    print("✅ Telco Customer Churn DB created successfully")
    print(f"📦 Database: {DB_PATH}")
    print(f"📊 Rows inserted: {rows}")

if __name__ == "__main__":
    main()
