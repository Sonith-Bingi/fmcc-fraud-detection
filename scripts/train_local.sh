#!/bin/bash
# Quick-start: train the model locally using the raw data files.
# Run from repo root: bash scripts/train_local.sh
set -e

BIHAR="../MSCDATA_Bhiharcsv.csv"
RAJASTHAN="../MSCDATA_40459 Rajasthan CSV.csv"
SUSPECTS_BIHAR="../Suspectdata_01072025_to_15072025_Bihar CSV.csv"
SUSPECTS_RAJ="../Suspects_010725_to_15072025 Rajasthan CSV.csv"

echo "=== Installing dependencies ==="
pip install -r requirements.txt -q

echo "=== Training model (with MLflow) ==="
python -m pipeline.train_mlflow \
    --bihar            "$BIHAR" \
    --rajasthan        "$RAJASTHAN" \
    --suspects_bihar   "$SUSPECTS_BIHAR" \
    --suspects_rajasthan "$SUSPECTS_RAJ" \
    --output           models/

echo ""
echo "=== Symlinking latest model ==="
LATEST=$(ls -t models/*.pkl | head -1)
cp "$LATEST" models/latest.pkl
echo "Latest model: $LATEST"

echo ""
echo "=== Running tests ==="
pytest tests/ -v

echo ""
echo "Done. Start API with:"
echo "  uvicorn app.main:app --reload"
