-- FedSnow: Create internal stage for model weight files
-- Run after 02_create_tables.sql

USE DATABASE FEDSNOW_DB;
USE SCHEMA FEDERATION;

CREATE STAGE IF NOT EXISTS MODEL_WEIGHTS_STAGE
    COMMENT = 'Internal stage for serialized PyTorch model weight files';

-- Verify stage was created
SHOW STAGES LIKE 'MODEL_WEIGHTS_STAGE';
