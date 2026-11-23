"""Train every model, evaluate on the hold-out, and save results.

Orchestrates:
1. Feature extraction and chronological 70/30 train/test hold-out splitting
2. Fitting and randomized hyperparameter search across all model families
3. Hybrid model formulation (RF+ARIMA, RF+ANN, residual boosting)
4. Standardized metrics (RMSE, MAE, R²) for every model on the same hold-out
5. Permutation feature importance analysis on the unseen hold-out split
6. Saving the best ensemble to `models/best_today.joblib` and a 2-day-ahead version to
   `models/best_tomorrow.joblib`
7. Exporting benchmark results to `results/` and the README results table

Usage:
    uv run python -m src.train
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance

from .data import (
    FEATURES,
    MODELS_DIR,
    ROOT,
    RESULTS_DIR,
    SEED,
    add_features,
    chrono_split,
    load_zones,
)
from .models import FAMILY, fit_zoo, reg_metrics
from .hybrids import family_of, hybrids


def prepare():
    """Loads zones, builds features and makes the chronological split."""
    raw = load_zones()
    feat = add_features(raw)
    train, test, cut = chrono_split(feat)
    return raw, train, test, cut


def main():
    raw, train, test, cut = prepare()
    X, y, Xte = train[FEATURES], train["y"], test[FEATURES]
    print(f"train {len(train):,} rows (< {cut:%Y-%m-%d}) | test {len(test):,} rows | {raw.zone.nunique()} zones", flush=True)

    preds, family, secs = baselines(test), dict(BASELINE_FAMILY), {}
    tuned, params, secs = fit_zoo(X, y)
    for n, m in tuned.items():
        preds[n], family[n] = m.predict(Xte), FAMILY[n]
    for n, p in hybrids(params, train, test, FEATURES, raw, cut).items():
        preds[n], family[n] = p, family_of(n)
    reg = save_results(raw, train, test, cut, tuned, params, preds, family, secs)
    print(reg.drop(columns="best_params").round(3).to_string(index=False))


BASELINE_FAMILY = {"Persistence (today = tomorrow)": "Baseline", "7-day mean": "Baseline"}


def baselines(test: pd.DataFrame) -> dict:
    """Naive forecasts: tomorrow equals today, and tomorrow equals the last 7-day mean."""
    return {"Persistence (today = tomorrow)": test["aqi"].values, "7-day mean": test["aqi_mean7"].values}


def score(yte, preds: dict, family: dict, secs: dict) -> pd.DataFrame:
    """Hold-out RMSE, MAE and R² for every model, best first."""
    rows = [dict(model=n, family=family[n], train_s=secs.get(n, np.nan),
                 **reg_metrics(yte, np.clip(np.asarray(p, dtype=float).ravel(), 0, None)))
            for n, p in preds.items()]
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


def save_results(raw, train, test, cut, tuned, params, preds, family, secs) -> pd.DataFrame:
    """Writes everything the dashboard reads: metrics, predictions, best models, importance, README table."""
    for d in (RESULTS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    Xte, yte = test[FEATURES], test["y"]
    meta_dict = dict(
        cutoff=str(cut.date()),
        start=str(raw.date.min().date()),
        end=str(raw.date.max().date()),
        n_train=len(train),
        n_test=len(test),
        n_zones=int(raw.zone.nunique()),
        features=FEATURES,
    )
    json.dump(meta_dict, open(RESULTS_DIR / "meta.json", "w"), indent=2)

    reg = score(yte, preds, family, secs)
    reg["best_params"] = reg["model"].map(lambda m: json.dumps(params.get(m, {}), default=float))
    reg.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    fam_summary = reg.groupby("family", as_index=False).first()[["family", "model", "RMSE", "MAE", "R2", "train_s"]]
    fam_summary.to_csv(RESULTS_DIR / "family_summary.csv", index=False)

    out = test[["date", "zone", "aqi", "y"]].copy()
    for k, v in preds.items():
        out[k] = np.clip(np.asarray(v, dtype=float).ravel(), 0, None)
    out.round(3).to_csv(RESULTS_DIR / "predictions.csv.gz", index=False)

    best = reg[reg.family.isin(["Bagging", "Boosting", "Stacking"])].iloc[0]["model"]
    joblib.dump(tuned[best], MODELS_DIR / "best_today.joblib", compress=3)
    day2 = fit_day2(tuned[best], train, test, cut)
    json.dump({"best_model": best, "day2": day2}, open(RESULTS_DIR / "best.json", "w"), indent=2)

    imp = {}  # permutation importance on the hold-out (not in-sample)
    for n in ("Random Forest", "LightGBM"):
        imp[n] = permutation_importance(tuned[n], Xte, yte, n_repeats=5, random_state=SEED,
                                        scoring="neg_root_mean_squared_error").importances_mean
    imp_df = pd.DataFrame(imp, index=FEATURES).sort_values("Random Forest", ascending=False).reset_index(names="feature")
    imp_df.to_csv(RESULTS_DIR / "importance.csv", index=False)
    print(f"Best ensemble: {best}")
    update_readme(reg, meta_dict, test)
    return reg


def fit_day2(best_model, train: pd.DataFrame, test: pd.DataFrame, cut: pd.Timestamp) -> dict:
    """Fit the best model to predict AQI two days ahead (t+2) and score it on the hold-out.

    The data always ends at the last complete day, so forecasting *tomorrow* needs t+2.
    Training rows whose t+2 target falls on or after the cutoff are left out (no peeking).
    """
    tr = train[train["y2"].notna() & (train["date"] + pd.Timedelta(days=2) < cut)]  # target before cutoff
    te = test[test["y2"].notna()]
    model = clone(best_model).fit(tr[FEATURES], tr["y2"])
    joblib.dump(model, MODELS_DIR / "best_tomorrow.joblib", compress=3)
    pred = np.clip(model.predict(te[FEATURES]), 0, None)
    scores = {k: round(v, 3) for k, v in reg_metrics(te["y2"], pred).items()}
    persistence = {k: round(v, 3) for k, v in reg_metrics(te["y2"], te["aqi"]).items()}
    print(f"  2-day-ahead best model: RMSE={scores['RMSE']:.2f} R2={scores['R2']:.3f} "
          f"(persistence RMSE={persistence['RMSE']:.2f})", flush=True)
    return {**scores, "persistence_RMSE": persistence["RMSE"], "n_test": int(len(te))}


def update_readme(reg: pd.DataFrame, meta: dict, test: pd.DataFrame):
    """Rewrite the results table in README.md (between the results markers)."""
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    start, end = "<!-- results:start -->", "<!-- results:end -->"
    if start not in text or end not in text:
        return
    rows = ["| Model | Family | RMSE | MAE | R² |", "|---|---|---|---|---|"]
    for i, r in reg.iterrows():
        b = (lambda v: f"**{v}**") if i == 0 else (lambda v: v)
        rows.append(f"| {r.model} | {r.family} | {b(f'{r.RMSE:.3f}')} | {b(f'{r.MAE:.3f}')} | {b(f'{r.R2:.3f}')} |")
    intro = (
        f"Evaluation on {meta['n_test']:,} zone-days hold-out (strict chronological split, final 30% of timeline, "
        f"test period: {test.date.min():%Y-%m-%d} to {test.date.max():%Y-%m-%d}; trained on {meta['start']} to "
        f"{meta['cutoff']}, {meta['n_zones']} zones covering all {count_dsds()} DSDs):"
    )
    block = f"{start}\n{intro}\n\n" + "\n".join(rows) + f"\n{end}"
    text = text[:text.index(start)] + block + text[text.index(end) + len(end):]
    readme.write_text(text, encoding="utf-8")
    print("Updated README results table.")


def count_dsds() -> int:
    return len(pd.read_csv(ROOT / "data" / "dsd_zones.csv"))


if __name__ == "__main__":
    main()
