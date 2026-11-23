"""Sri Lanka air quality forecasting pipeline.

Run in order:
    python -m src.prepare_data   # raw CSV -> data/zone_daily.csv.gz + data/dsd_zones.csv
    python -m src.train          # fit all models -> models/ + results/

Modules: data (paths, loading, features, split), models (model zoo), hybrids (hybrid models).
"""
