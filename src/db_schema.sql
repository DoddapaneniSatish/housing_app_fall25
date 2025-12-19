PRAGMA foreign_keys = ON;

-- Drop old tables if rerun
DROP TABLE IF EXISTS fact_churn;
DROP TABLE IF EXISTS dim_gender;
DROP TABLE IF EXISTS dim_contract;
DROP TABLE IF EXISTS dim_payment;
DROP TABLE IF EXISTS dim_internet;
DROP TABLE IF EXISTS dim_yesno;

-- =========================
-- Dimension Tables
-- =========================

CREATE TABLE dim_gender (
    gender_id INTEGER PRIMARY KEY AUTOINCREMENT,
    gender TEXT UNIQUE
);

CREATE TABLE dim_contract (
    contract_id INTEGER PRIMARY KEY AUTOINCREMENT,
    contract_type TEXT UNIQUE
);

CREATE TABLE dim_payment (
    payment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    payment_method TEXT UNIQUE
);

CREATE TABLE dim_internet (
    internet_id INTEGER PRIMARY KEY AUTOINCREMENT,
    internet_service TEXT UNIQUE
);

-- Shared Yes / No / No internet service dimension
CREATE TABLE dim_yesno (
    yesno_id INTEGER PRIMARY KEY AUTOINCREMENT,
    value TEXT UNIQUE
);

-- =========================
-- Fact Table
-- =========================

CREATE TABLE fact_churn (
    customer_id TEXT PRIMARY KEY,

    -- Numeric features
    tenure INTEGER,
    monthly_charges REAL,
    total_charges REAL,
    senior_citizen INTEGER,

    -- Categorical (FKs)
    gender_id INTEGER,
    contract_id INTEGER,
    payment_id INTEGER,
    internet_id INTEGER,

    partner_id INTEGER,
    dependents_id INTEGER,
    phone_service_id INTEGER,
    paperless_billing_id INTEGER,

    multiple_lines_id INTEGER,
    online_security_id INTEGER,
    online_backup_id INTEGER,
    device_protection_id INTEGER,
    tech_support_id INTEGER,
    streaming_tv_id INTEGER,
    streaming_movies_id INTEGER,

    -- Target
    churn INTEGER,

    FOREIGN KEY (gender_id) REFERENCES dim_gender(gender_id),
    FOREIGN KEY (contract_id) REFERENCES dim_contract(contract_id),
    FOREIGN KEY (payment_id) REFERENCES dim_payment(payment_id),
    FOREIGN KEY (internet_id) REFERENCES dim_internet(internet_id),

    FOREIGN KEY (partner_id) REFERENCES dim_yesno(yesno_id),
    FOREIGN KEY (dependents_id) REFERENCES dim_yesno(yesno_id),
    FOREIGN KEY (phone_service_id) REFERENCES dim_yesno(yesno_id),
    FOREIGN KEY (paperless_billing_id) REFERENCES dim_yesno(yesno_id),
    FOREIGN KEY (multiple_lines_id) REFERENCES dim_yesno(yesno_id),
    FOREIGN KEY (online_security_id) REFERENCES dim_yesno(yesno_id),
    FOREIGN KEY (online_backup_id) REFERENCES dim_yesno(yesno_id),
    FOREIGN KEY (device_protection_id) REFERENCES dim_yesno(yesno_id),
    FOREIGN KEY (tech_support_id) REFERENCES dim_yesno(yesno_id),
    FOREIGN KEY (streaming_tv_id) REFERENCES dim_yesno(yesno_id),
    FOREIGN KEY (streaming_movies_id) REFERENCES dim_yesno(yesno_id)
);
