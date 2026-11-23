"""Shared paths, data loading, feature engineering, and the chronological split.

Builds leak-free features for forecasting each zone's daily mean US AQI:
- Current pollutant concentrations (PM2.5, PM10, NO₂, SO₂, O₃, CO, AOD, Dust, UV Index)
- Temporal lags (1, 2, 3, 7 days)
- Rolling statistics (3-day mean, 7-day mean, 7-day standard deviation)
- Calendar cyclical variations (monsoon season, day of week, annual harmonics)
- Targets: AQI one day ahead (y, t+1) and two days ahead (y2, t+2); because the
  dataset ends at the last complete day, t+1 is "today" and t+2 is "tomorrow"
- Strict chronological 70/30 train/test hold-out partitioning
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR, MODELS_DIR, RESULTS_DIR = ROOT / "data", ROOT / "models", ROOT / "results"

SEED = 42
TEST_FRACTION = 0.30  # Final 30% of timeline forms the chronological hold-out
SEASONS = {
    0: "North-East monsoon (Dec-Feb)",
    1: "1st inter-monsoon (Mar-Apr)",
    2: "South-West monsoon (May-Sep)",
    3: "2nd inter-monsoon (Oct-Nov)",
}

TODAY = [
    "aqi", "aqi_max", "pm25", "pm25_max", "pm10", "no2", "so2", "o3", "co",
    "aod", "dust", "uv_max",
]
FEATURES = (
    TODAY
    + ["aqi_lag1", "aqi_lag2", "aqi_lag3", "aqi_lag7", "pm25_lag1", "aqi_mean3", "aqi_mean7", "aqi_std7",
       "dow", "month_sin", "month_cos", "season", "is_new_year_week", "lat", "lon"]
)


def season_of(m: int) -> int:
    """Categorizes month into one of four Sri Lankan monsoon seasons."""
    return 0 if m in (12, 1, 2) else 1 if m in (3, 4) else 2 if m <= 9 else 3


def load_zones() -> pd.DataFrame:
    """Loads daily aggregated time series for all discrete ~45 km atmospheric grid zones."""
    return pd.read_csv(DATA_DIR / "zone_daily.csv.gz", parse_dates=["date"]).sort_values(["zone", "date"]).reset_index(drop=True)


def load_dsds() -> pd.DataFrame:
    """Loads spatial lookup table mapping Divisional Secretariat Divisions to grid zones."""
    return pd.read_csv(DATA_DIR / "dsd_zones.csv")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds temporal lags, rolling aggregations, calendar features, and targets y (t+1) and y2 (t+2)."""
    df = df.copy()
    g = df.groupby("zone", sort=False)
    for k in (1, 2, 3, 7):
        df[f"aqi_lag{k}"] = g["aqi"].shift(k)
    df["pm25_lag1"] = g["pm25"].shift(1)
    df["aqi_mean3"] = g["aqi"].transform(lambda s: s.rolling(3).mean())
    df["aqi_mean7"] = g["aqi"].transform(lambda s: s.rolling(7).mean())
    df["aqi_std7"] = g["aqi"].transform(lambda s: s.rolling(7).std())

    d = df["date"]
    df["dow"] = d.dt.day_of_week
    df["month_sin"] = np.sin(2 * np.pi * d.dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * d.dt.month / 12)
    df["season"] = d.dt.month.map(season_of)
    df["is_new_year_week"] = ((d.dt.month == 4) & d.dt.day.between(10, 17)).astype(int)
    df["y"] = g["aqi"].shift(-1)   # next day (t+1)
    df["y2"] = g["aqi"].shift(-2)  # two days ahead (t+2): "tomorrow" when t is the latest full day
    return df.dropna(subset=["aqi_lag7", "aqi_mean7", "aqi_std7"]).reset_index(drop=True)


def chrono_split(df: pd.DataFrame, frac: float = TEST_FRACTION):
    """Executes a chronological train/test split without shuffling."""
    df = df.dropna(subset=["y"])
    dates = np.sort(df["date"].unique())
    cut = pd.Timestamp(dates[int(len(dates) * (1 - frac))])
    # Training rows must have their target (t+1) before the cutoff, so the last
    # pre-cutoff day is dropped: its target is the first test day.
    train = df[df["date"] + pd.Timedelta(days=1) < cut].sort_values(["date", "zone"]).reset_index(drop=True)
    test = df[df["date"] >= cut].sort_values(["date", "zone"]).reset_index(drop=True)
    return train, test, cut


def aqi_category(v: float) -> str:
    """Categorizes continuous US AQI values into EPA health standard tiers."""
    for hi, name in [(50, "Good"), (100, "Moderate"), (150, "Unhealthy for Sensitive Groups"),
                     (200, "Unhealthy"), (300, "Very Unhealthy")]:
        if v <= hi:
            return name
    return "Hazardous"
