-- FedSnow: Create federation tables
-- Run after 01_create_schema.sql

USE DATABASE FEDSNOW_DB;
USE SCHEMA FEDERATION;

-- Client weight uploads per round
CREATE TABLE IF NOT EXISTS CLIENT_WEIGHTS (
    round_id         INT,
    client_id        VARCHAR,
    client_name      VARCHAR,
    num_samples      INT,
    weights          VARIANT,
    local_loss       FLOAT,
    local_accuracy   FLOAT,
    uploaded_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Aggregated global model per round
CREATE TABLE IF NOT EXISTS GLOBAL_MODEL (
    round_id         INT,
    weights          VARIANT,
    global_accuracy  FLOAT,
    avg_loss         FLOAT,
    num_clients      INT,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Per-round per-client metrics
CREATE TABLE IF NOT EXISTS ROUND_METRICS (
    round_id          INT,
    client_id         VARCHAR,
    client_name       VARCHAR,
    local_accuracy    FLOAT,
    local_loss        FLOAT,
    global_accuracy   FLOAT,
    weight_divergence FLOAT,
    num_samples       INT,
    round_timestamp   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Cortex drift analysis results
CREATE TABLE IF NOT EXISTS DRIFT_ANALYSIS (
    round_id      INT,
    client_id     VARCHAR,
    drift_score   FLOAT,
    cluster_id    INT,
    analysis_text VARCHAR,
    analyzed_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
