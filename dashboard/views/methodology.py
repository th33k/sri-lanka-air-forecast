"""Methodology View: data source, spatial zones, forecast targets and evaluation."""
import streamlit as st


def render(meta, dsds):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
#### Data Source
Daily values come from the
[dataset-sri-lanka-air-quality](https://github.com/th33k/dataset-sri-lanka-air-quality)
repository: Copernicus Atmosphere Monitoring Service (CAMS) global model data
accessed through the Open-Meteo API, aggregated to daily mean, minimum and maximum
for every DSD.

CAMS values are **model estimates on a ~45 km grid**, not readings from ground-level
monitoring stations. The dataset holds **completed days only**, so its latest day is
always yesterday.
""")

        st.markdown(f"""
#### Spatial Zones
The CAMS grid (~45 km) is coarser than most DSDs, so neighbouring divisions in the same
grid cell have identical series. Grouping them gives **{meta.get('n_zones', 'n/a')} zones**.
Models are trained on zones, so each distinct series counts once instead of being
repeated for every DSD in the cell; forecasts are then shown for every DSD in the zone.
""")

    with col2:
        st.markdown(f"""
#### Dataset Summary

| Attribute | Value |
|:---|:---|
| Spatial zones | {meta.get('n_zones', 'n/a')} |
| Administrative divisions (DSDs) | {len(dsds)} |
| Training data period | {meta.get('start', 'n/a')} to {meta.get('end', 'n/a')} |
| Training samples | {meta.get('n_train', 0):,} zone-days |
| Test samples | {meta.get('n_test', 0):,} zone-days |
| Train / test split date | {meta.get('cutoff', 'n/a')} |
| Number of features | {len(meta.get('features', []))} |
""")

        st.markdown("""
#### Forecast Targets and Evaluation

**Targets**: daily mean US AQI **one day ahead** (t+1) and **two days ahead** (t+2).
From the latest complete day (yesterday), t+1 is today and t+2 is tomorrow. The
2-day model is the best ensemble refitted on the t+2 target.

**Features**: the day's pollutant values (PM2.5, PM10, NO2, SO2, O3, CO, AOD, dust,
UV index), AQI lags (1, 2, 3, 7 days), rolling statistics (3- and 7-day mean, 7-day
standard deviation), calendar features (day of week, month harmonics, monsoon season,
Sinhala and Tamil New Year week) and zone coordinates.

**Evaluation**: strict chronological split (first 70% of days for training, last 30%
for testing, no shuffling; the last training day is dropped because its target is the first
test day). Hyperparameters are tuned on training data only, with the
same randomized-search budget for every tuned model (10 candidates x 4 `TimeSeriesSplit`
folds). All scores in this dashboard are on the unseen test period.
""")

    with st.expander("Complete Feature Catalog", expanded=False):
        features = meta.get("features", [])
        if features:
            n_cols = 4
            cols = st.columns(n_cols)
            for i, f in enumerate(features):
                cols[i % n_cols].markdown(f"- `{f}`")
        else:
            st.info("Feature list not available in metadata.")
