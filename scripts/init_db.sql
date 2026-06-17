-- Prediction log table (Phase 5 monitoring)
CREATE TABLE IF NOT EXISTS prediction_log (
    id          SERIAL PRIMARY KEY,
    msisdn      BIGINT NOT NULL,
    date        DATE NOT NULL,
    predicted_at TIMESTAMPTZ DEFAULT NOW(),
    fraud_probability FLOAT NOT NULL,
    is_fraud    BOOLEAN NOT NULL,
    risk_tier   TEXT NOT NULL,
    model_version TEXT NOT NULL,
    threshold   FLOAT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pred_log_date ON prediction_log(date);
CREATE INDEX IF NOT EXISTS idx_pred_log_msisdn ON prediction_log(msisdn);
CREATE INDEX IF NOT EXISTS idx_pred_log_fraud ON prediction_log(is_fraud) WHERE is_fraud = true;

-- Drift report table (Phase 5)
CREATE TABLE IF NOT EXISTS drift_report (
    id          SERIAL PRIMARY KEY,
    report_date DATE NOT NULL,
    feature     TEXT NOT NULL,
    drift_score FLOAT,
    drift_detected BOOLEAN,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_drift_date ON drift_report(report_date);
