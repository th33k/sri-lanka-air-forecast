"""Data loading, caching, and model artifact access utilities."""
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.data import FEATURES, MODELS_DIR, RESULTS_DIR, add_features, load_dsds, load_zones


@st.cache_data(show_spinner=False)
def get_zones():
    return load_zones()


@st.cache_data(show_spinner=False)
def get_dsds():
    return load_dsds()


@st.cache_data(show_spinner=False)
def get_dsd_daily():
    """Maps zone-level time series onto Divisional Secretariat Division boundaries."""
    return get_dsds().merge(get_zones().drop(columns=["lat", "lon"]), on="zone")


@st.cache_data(show_spinner=False)
def get_results():
    """Loads benchmark metrics, hold-out predictions, importance rankings, and metadata."""
    m_path = RESULTS_DIR / "metrics.csv"
    p_path = RESULTS_DIR / "predictions.csv.gz"
    i_path = RESULTS_DIR / "importance.csv"
    meta_path = RESULTS_DIR / "meta.json"
    best_path = RESULTS_DIR / "best.json"

    metrics = pd.read_csv(m_path) if m_path.exists() else None
    preds = pd.read_csv(p_path, parse_dates=["date"]) if p_path.exists() else None
    importance = pd.read_csv(i_path) if i_path.exists() else None
    meta = json.load(open(meta_path, encoding="utf-8")) if meta_path.exists() else {}
    best_model = json.load(open(best_path, encoding="utf-8")).get("best_model", "Ensemble") if best_path.exists() else "Ensemble"

    return metrics, preds, importance, meta, best_model


@st.cache_resource(show_spinner=False)
def get_best_model(horizon: int = 1):
    """Loads and caches the trained model for a 1-day or 2-day-ahead forecast."""
    return joblib.load(MODELS_DIR / ("best_today.joblib" if horizon == 1 else "best_tomorrow.joblib"))


def has_day2_model() -> bool:
    return (MODELS_DIR / "best_tomorrow.joblib").exists()


@st.cache_data(show_spinner=False)
def get_horizon_scores():
    """Hold-out RMSE/R2 per horizon, from results/best.json and metrics.csv."""
    best = json.load(open(RESULTS_DIR / "best.json", encoding="utf-8"))
    m = pd.read_csv(RESULTS_DIR / "metrics.csv").set_index("model")
    day1 = m.loc[best["best_model"], ["RMSE", "R2"]].to_dict() if best["best_model"] in m.index else {}
    return {1: day1, 2: best.get("day2", {})}


@st.cache_data(show_spinner=False)
def get_featured_zones():
    """Computes and caches zone-level feature matrices."""
    return add_features(get_zones())


@st.cache_data(show_spinner=False)
def get_forecast_for_date(selected_date=None, horizon: int = 1):
    """Forecast AQI `horizon` days after an observation date (t), for every DSD.

    The data ends at the last complete day, so horizon=2 from the latest date is tomorrow.
    Returns a DSD-level table with 'forecast', 'aqi' (observed on t), 'actual_next_day'
    (observed on t+horizon, when known), 'obs_date' and 'forecast_date'.
    """
    f = get_featured_zones()
    if selected_date is None:
        target_day = f.date.max().date()
    elif isinstance(selected_date, str):
        target_day = pd.to_datetime(selected_date).date()
    elif hasattr(selected_date, "date"):
        target_day = selected_date.date()
    else:
        target_day = selected_date

    day_rows = f[f.date == pd.Timestamp(target_day)]
    if day_rows.empty:
        day_rows = f[f.date == f.date.max()]
        target_day = f.date.max().date()

    model = get_best_model(horizon)
    out = day_rows[["zone", "date", "aqi"]].copy()
    out["forecast"] = np.clip(model.predict(day_rows[FEATURES]), 0, None)

    next_day = target_day + pd.Timedelta(days=horizon)
    next_actuals = f[f.date == pd.Timestamp(next_day)][["zone", "aqi"]].rename(columns={"aqi": "actual_next_day"})
    if not next_actuals.empty:
        out = out.merge(next_actuals, on="zone", how="left")
    else:
        out["actual_next_day"] = np.nan

    out["obs_date"] = target_day
    out["forecast_date"] = next_day
    out["date"] = pd.to_datetime(next_day)
    return get_dsds().merge(out, on="zone")



@st.cache_data(show_spinner=False)
def key_dates(cutoff=None):
    """Notable days computed from the data, so they stay correct as new data arrives."""
    z = get_zones()
    daily = z.groupby(z.date.dt.date)["aqi"].mean()
    dates = {
        f"{daily.index[-1]}: Latest record": daily.index[-1],
        f"{daily.idxmax()}: Worst day (island mean AQI {daily.max():.0f})": daily.idxmax(),
        f"{daily.idxmin()}: Cleanest day (island mean AQI {daily.min():.0f})": daily.idxmin(),
    }
    if cutoff:
        c = pd.to_datetime(cutoff).date()
        dates[f"{c}: Train / test split"] = c
    dates[f"{daily.index[0]}: First record"] = daily.index[0]
    return dates


@st.cache_data(show_spinner=False)
def get_district_daily():
    """District daily means (one row per district per day): ~14x smaller than the DSD table."""
    cols = ["aqi", "aqi_max", "pm25", "pm10", "no2", "so2", "o3", "co", "aod", "dust", "uv_max"]
    return get_dsd_daily().groupby(["district", "date"], as_index=False)[cols].mean()
