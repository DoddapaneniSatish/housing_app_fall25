from __future__ import annotations

import sqlite3
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# PATHS
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT / "data" / "db" / "telco_churn.sqlite3"

# --------------------------------------------------
# SQL JOIN QUERY
# --------------------------------------------------
JOIN_QUERY = """
SELECT
    f.customer_id,

    -- numeric
    f.tenure,
    f.monthly_charges,
    f.total_charges,
    f.senior_citizen,

    -- categorical (decoded from dimensions)
    g.gender,
    ct.contract_type,
    pm.payment_method,
    i.internet_service,

    yn_partner.value AS partner,
    yn_dependents.value AS dependents,
    yn_phone.value AS phone_service,
    yn_paperless.value AS paperless_billing,

    yn_multi.value AS multiple_lines,
    yn_sec.value AS online_security,
    yn_backup.value AS online_backup,
    yn_device.value AS device_protection,
    yn_tech.value AS tech_support,
    yn_tv.value AS streaming_tv,
    yn_movies.value AS streaming_movies,

    -- target
    f.churn
FROM fact_churn f
JOIN dim_gender g ON f.gender_id = g.gender_id
JOIN dim_contract ct ON f.contract_id = ct.contract_id
JOIN dim_payment pm ON f.payment_id = pm.payment_id
JOIN dim_internet i ON f.internet_id = i.internet_id

JOIN dim_yesno yn_partner ON f.partner_id = yn_partner.yesno_id
JOIN dim_yesno yn_dependents ON f.dependents_id = yn_dependents.yesno_id
JOIN dim_yesno yn_phone ON f.phone_service_id = yn_phone.yesno_id
JOIN dim_yesno yn_paperless ON f.paperless_billing_id = yn_paperless.yesno_id

JOIN dim_yesno yn_multi ON f.multiple_lines_id = yn_multi.yesno_id
JOIN dim_yesno yn_sec ON f.online_security_id = yn_sec.yesno_id
JOIN dim_yesno yn_backup ON f.online_backup_id = yn_backup.yesno_id
JOIN dim_yesno yn_device ON f.device_protection_id = yn_device.yesno_id
JOIN dim_yesno yn_tech ON f.tech_support_id = yn_tech.yesno_id
JOIN dim_yesno yn_tv ON f.streaming_tv_id = yn_tv.yesno_id
JOIN dim_yesno yn_movies ON f.streaming_movies_id = yn_movies.yesno_id
;
"""

# --------------------------------------------------
# LOAD FUNCTION
# --------------------------------------------------
def load_dataframe(db_path: str | Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    """
    Load full joined Telco Customer Churn dataset from SQLite.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite DB not found: {db_path}")

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(JOIN_QUERY, conn)

    return df

# --------------------------------------------------
# SPLIT X / y
# --------------------------------------------------
def split_X_y(df: pd.DataFrame):
    """
    Split dataframe into features (X) and target (y).
    """
    y = df["churn"].astype(int)
    X = df.drop(columns=["churn", "customer_id"])
    return X, y

# --------------------------------------------------
# QUICK TEST
# --------------------------------------------------
if __name__ == "__main__":
    df = load_dataframe()
    print("✅ Loaded joined Telco churn dataframe")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head(3))
