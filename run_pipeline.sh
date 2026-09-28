#!/usr/bin/env bash
# Rebuild everything from the raw LendingClub file: tables, figures, models, and the site's JSON.
# Expects data/raw/accepted_2007_to_2018Q4.csv.gz (download from Kaggle; see README). Takes ~40 minutes.
set -euo pipefail
PY=${PYTHON:-python}

$PY -m src.data                 # read only the needed columns from the raw CSV, cache as parquet
$PY -m src.eda                  # SQL base table + Phase 1 checks and charts
$PY -m src.split_check          # Phase 2: time split summary and training-window experiment
$PY -m src.train                # Phase 3: tune on 2014, train 4 models on 2007-2013
$PY -m src.evaluate             #          metrics, calibration, feature importance
$PY -m src.xgb_category_check   #          categorical-split experiment (validation only)
$PY -m src.strategy             # Phase 4: approve/decline policies in dollars
$PY -m src.state_check          # Phase 5: does the model need state? (pre-committed threshold)
$PY -m src.explain              #          decline reasons (SHAP)
$PY -m src.fairness             #          approval and default rates by group (SQL)
$PY -m src.monitoring           #          PSI drift
$PY -m src.export_site_data     # JSON for the static site
$PY -m pytest -q tests
