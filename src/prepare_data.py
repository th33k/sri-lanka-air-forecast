"""Build the modelling tables from the Sri Lanka DSD air quality dataset.

Reads the latest daily dataset and writes:
- data/dsd_zones.csv       DSD -> ~45 km CAMS grid zone lookup
- data/zone_daily.csv.gz   daily values per zone

DSDs inside the same CAMS grid cell have identical series, so they are merged
into one zone. The zone assignment is kept between runs so zone ids stay in
line with the trained model's results; pass --rezone to rebuild it (do this
before retraining).

The dataset is stored as one CSV per year. Source: the given folder (or CSV),
else a sibling clone of dataset-sri-lanka-air-quality, else GitHub (public repo).

Usage:
    python -m src.prepare_data                 # default source (see above)
    python -m src.prepare_data path/to/folder  # folder of yearly CSVs
    python -m src.prepare_data --rezone        # also rebuild zones
"""
from datetime import date
from pathlib import Path
from urllib.error import HTTPError
import sys

import pandas as pd

from .data import DATA_DIR, ROOT

DATASET_URL = (
    "https://raw.githubusercontent.com/th33k/dataset-sri-lanka-air-quality/"
    "main/data/processed/sri_lanka_dsd_air_quality_{year}.csv"
)
FIRST_YEAR = 2023
LOCAL_DATASET = ROOT.parent / "dataset-sri-lanka-air-quality" / "data" / "processed"
ZONES_CSV = DATA_DIR / "dsd_zones.csv"

RENAME = {
    "pm2_5_mean": "pm25", "pm2_5_max": "pm25_max", "pm10_mean": "pm10", "nitrogen_dioxide_mean": "no2",
    "sulphur_dioxide_mean": "so2", "ozone_mean": "o3", "carbon_monoxide_mean": "co",
    "aerosol_optical_depth_mean": "aod", "dust_mean": "dust", "uv_index_max": "uv_max",
    "us_aqi_mean": "aqi", "us_aqi_max": "aqi_max",
}


def build_zones(df: pd.DataFrame) -> pd.DataFrame:
    """Group DSDs whose AQI series are identical (same CAMS grid cell) into zones."""
    sig = df.pivot(index="dsd_pcode", columns="date", values="us_aqi_mean").round(2).astype(str).agg("|".join, axis=1)
    zone_of = pd.Series(sig.factorize()[0] + 1, index=sig.index)
    dsd = df.groupby("dsd_pcode").agg(
        dsd=("dsd_name", "first"), district=("district_name", "first"),
        province=("province_name", "first"), lat=("latitude", "first"), lon=("longitude", "first"),
    ).reset_index()
    dsd["zone"] = dsd["dsd_pcode"].map(zone_of)
    main_district = dsd.groupby("zone")["district"].agg(lambda s: s.mode()[0])
    dsd["zone_name"] = dsd["zone"].map(lambda z: f"Z{z:02d} {main_district[z]}")
    return dsd


def read_dataset(source=None) -> pd.DataFrame:
    """Read the yearly CSVs from a folder, a single CSV, or GitHub."""
    source = Path(source) if source else (LOCAL_DATASET if LOCAL_DATASET.exists() else None)
    if source and source.is_file():
        print(f"Reading {source}")
        return pd.read_csv(source)
    if source:
        files = sorted(source.glob("sri_lanka_dsd_air_quality_*.csv"))
        print(f"Reading {len(files)} yearly files from {source}")
        return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    frames = []
    for year in range(FIRST_YEAR, date.today().year + 1):
        try:
            frames.append(pd.read_csv(DATASET_URL.format(year=year)))
            print(f"Downloaded {year}")
        except HTTPError as err:
            print(f"Skipped {year}: {err}")
    if not frames:
        sys.exit("Could not download the dataset from GitHub.")
    return pd.concat(frames, ignore_index=True)


def main(source=None, rezone: bool = False):
    df = read_dataset(source)
    df = df.dropna(subset=["us_aqi_mean"]).drop_duplicates(["date", "dsd_pcode"])
    df["date"] = pd.to_datetime(df["date"])

    # Keep only the date range every DSD covers, and DSDs with no missing days in it.
    start = df.groupby("dsd_pcode")["date"].min().max()
    end = df.groupby("dsd_pcode")["date"].max().min()
    df = df[df["date"].between(start, end)]
    n_days = df["date"].nunique()
    complete = df.groupby("dsd_pcode").size().eq(n_days)
    df = df[df["dsd_pcode"].isin(complete[complete].index)]
    print(f"{df.dsd_pcode.nunique()} DSDs with complete records, {start:%Y-%m-%d} to {end:%Y-%m-%d} ({n_days} days)")

    if rezone or not ZONES_CSV.exists():
        dsd = build_zones(df)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        dsd.to_csv(ZONES_CSV, index=False, lineterminator="\n")
        print(f"Built {dsd.zone.nunique()} zones -> {ZONES_CSV.name}")
    else:
        dsd = pd.read_csv(ZONES_CSV)
        df = df[df["dsd_pcode"].isin(dsd["dsd_pcode"])]

    df = df.merge(dsd[["dsd_pcode", "zone"]], on="dsd_pcode")
    z = df.rename(columns=RENAME).groupby(["zone", "date"])[list(RENAME.values())].mean().reset_index()
    z = z.merge(dsd.groupby("zone").agg(lat=("lat", "mean"), lon=("lon", "mean")).reset_index(), on="zone")
    num_cols = z.select_dtypes(include="float").columns
    z[num_cols] = z[num_cols].round(3)
    z.to_csv(DATA_DIR / "zone_daily.csv.gz", index=False, lineterminator="\n",
             compression={"method": "gzip", "mtime": 0})  # byte-stable output
    print(f"Wrote {len(z):,} zone-days ({z.zone.nunique()} zones x {n_days} days) -> zone_daily.csv.gz")


if __name__ == "__main__":
    args = sys.argv[1:]
    paths = [a for a in args if not a.startswith("--")]
    main(paths[0] if paths else None, rezone="--rezone" in args)
