# Evaluation Results and Model Artifacts

This directory stores all empirical evaluation metrics, hold-out predictions, permutation importance rankings, and training metadata produced by the forecasting framework.

## Directory Structure

```
results/
├── metrics.csv          # Benchmark table for all evaluated models
├── family_summary.csv   # Best model per family
├── best.json            # Best ensemble name and its 2-day-ahead scores
├── predictions.csv.gz   # Hold-out predictions per model
├── importance.csv       # Hold-out permutation importance
└── meta.json            # Split cutoff, sample counts, feature list
```

---

## Artifact Descriptions

### 1. `metrics.csv`
Complete benchmark evaluation table for all models trained and evaluated on the chronological hold-out set (sizes and dates are in `meta.json`).
* `model`: Model name / variant identifier.
* `family`: Architectural classification (`Baseline`, `Bagging`, `Boosting`, `Stacking`, `Hybrid RF+ANN`, `Hybrid RF+ARIMA`, `Hybrid RF+Boosting`, `Time-series`, `Component`).
* `train_s`: Seconds spent on tuning and fitting (blank for baselines and hybrids, which are not timed).
* `RMSE`: Root Mean Squared Error on the hold-out test split.
* `MAE`: Mean Absolute Error on the hold-out test split.
* `R2`: Coefficient of determination ($R^2$) on the hold-out test split.
* `best_params`: JSON-serialized optimal hyperparameters discovered during `TimeSeriesSplit` cross-validation.

### 2. `family_summary.csv`
Distilled summary presenting the top-performing model for each distinct family sorted by hold-out RMSE.

### 3. `predictions.csv.gz`
Compressed CSV with every model's 1-day-ahead forecast for each zone and test day.
* `date`: Observation date ($t$); the forecasts are for the next day ($t+1$).
* `zone`: Spatial grid zone identifier (see `data/dsd_zones.csv`).
* `aqi`: Day $t$ observed US AQI.
* `y`: Observed US AQI on day $t+1$ (the value being forecast).
* Additional columns: Predicted values corresponding to each model.

### 4. `importance.csv`
Permutation feature importance on the unseen hold-out (5 repeats, random seed 42): the increase in RMSE when each feature's values are shuffled. Columns: `feature`, `Random Forest`, `LightGBM`.

### 5. `meta.json`
Metadata parameters capturing the experimental setup:
* `cutoff`: Temporal train/test split boundary date.
* `start` / `end`: Complete timeline span.
* `n_train` / `n_test`: Sample counts for train and test partitions.
* `n_zones`: Number of discrete ~45 km atmospheric grid zones.
* `features`: List of model input features.

### 6. `best.json`
* `best_model`: Best ensemble (Bagging, Boosting or Stacking family) by hold-out RMSE; saved as `models/best_today.joblib`.
* `day2`: The same model refitted to forecast two days ahead (`models/best_tomorrow.joblib`), with its hold-out `RMSE`, `MAE`, `R2`, the persistence baseline RMSE at that horizon, and `n_test`.
