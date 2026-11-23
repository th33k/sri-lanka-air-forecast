"""Hybrid architectures combining RF with ARIMA, ANN, and boosting.

Integrates complementary model paradigms to evaluate error reduction:
1. RF + ARIMA:
   - arima_feature: Uses out-of-fold ARIMA(2,0,1) forecast as an exogenous input feature for Random Forest
   - arima_residuals: Trains Random Forest on the residual errors of the fitted ARIMA baseline
2. RF + ANN:
   - ann_stacking: Stacking regressor with a Ridge meta-learner blending Random Forest and MLP neural network
   - ann_average: Simple unweighted linear blend of RF and ANN predictions
3. RF + Boosting:
   - boost_residuals: Trains a secondary LightGBM regressor on Random Forest out-of-bag residuals
   - equal_vote: Equal-weight ensemble averaging RF, XGBoost, LightGBM, and CatBoost
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA

from .data import SEED
from .models import base_models

warnings.filterwarnings("ignore")
ARIMA_ORDER = (2, 0, 1)
FAMILY = {
    "RF + ANN": "Hybrid RF+ANN",
    "RF + boosting": "Hybrid RF+Boosting",
    "RF + LightGBM": "Hybrid RF+Boosting",
    "ARIMA alone": "Time-series",
    "RF + ARIMA": "Hybrid RF+ARIMA",
    "ARIMA + RF": "Hybrid RF+ARIMA",
}


def family_of(name: str) -> str:
    """Classifies model name into its respective family category."""
    if name.endswith("(component)"):
        return "Component"
    return next((v for k, v in FAMILY.items() if name.startswith(k)), "Other")


def build(name: str, params: dict):
    """Instantiates a base model with tuned hyperparameters."""
    return base_models()[name].set_params(**params.get(name, {}))


def ann(size=(64, 32)):
    """Creates a standardized Multi-Layer Perceptron neural network pipeline."""
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=size, early_stopping=True, max_iter=300, random_state=SEED),
    )


def arima_next_day(full: pd.DataFrame, cut: pd.Timestamp) -> pd.DataFrame:
    """Fits ARIMA(2,0,1) per zone on the training days, then gives each day t its forecast of t+1.

    The parameters are estimated on training days only; `apply` then runs the fitted model
    over the whole series, so each one-step forecast uses only data up to day t.
    fittedvalues[i] is the forecast of y[i] made at i-1, so the forecast for t+1 made at t
    is fittedvalues[t+1] (and forecast(1) for the last day).
    """
    out = []
    for zone, g in full.groupby("zone"):
        g = g.sort_values("date")
        y = g["aqi"].values.astype(float)
        n_train = int((g["date"] < cut).sum())
        model = ARIMA(y[:n_train], order=ARIMA_ORDER).fit()
        applied = model.apply(y)
        f_next = np.concatenate([applied.fittedvalues[1:], applied.forecast(1)])
        out.append(pd.DataFrame({"zone": zone, "date": g["date"].values, "arima_next": f_next}))
    return pd.concat(out, ignore_index=True)


def hybrids(params: dict, train: pd.DataFrame, test: pd.DataFrame, features: list, full: pd.DataFrame, cut: pd.Timestamp) -> dict:
    """Fits all hybrid architectures and generates hold-out predictions."""
    Xtr, ytr = train[features], train["y"]
    Xte, yte = test[features], test["y"]
    preds = {}

    # RF + ARIMA integration
    arima_df = arima_next_day(full, cut)
    tr_ar = train.merge(arima_df, on=["zone", "date"]).dropna(subset=["arima_next"])
    te_ar = test.merge(arima_df, on=["zone", "date"]).dropna(subset=["arima_next"])

    rf_ar = build("Random Forest", params).fit(tr_ar[features + ["arima_next"]], tr_ar["y"])
    preds["RF + ARIMA (forecast as feature)"] = rf_ar.predict(te_ar[features + ["arima_next"]])
    preds["ARIMA alone"] = te_ar["arima_next"].values

    # ARIMA residuals fitted by Random Forest
    rf_res = build("Random Forest", params).fit(tr_ar[features], tr_ar["y"] - tr_ar["arima_next"])
    preds["ARIMA + RF (RF on ARIMA residuals)"] = te_ar["arima_next"].values + rf_res.predict(te_ar[features])

    # RF + ANN blending and stacking
    ann_model = ann().fit(Xtr, ytr)
    rf_model = build("Random Forest", params).fit(Xtr, ytr)
    p_rf, p_ann = rf_model.predict(Xte), ann_model.predict(Xte)
    preds["ANN (component)"] = p_ann
    preds["RF + ANN (simple average)"] = 0.5 * (p_rf + p_ann)

    stack_ann = StackingRegressor(
        estimators=[("rf", build("Random Forest", params)), ("ann", ann())],
        final_estimator=RidgeCV(), cv=KFold(5, shuffle=False), n_jobs=-1
    ).fit(Xtr, ytr)
    preds["RF + ANN (ridge stacking)"] = stack_ann.predict(Xte)

    # RF + Boosting integration
    boosters = {k: build(k, params).fit(Xtr, ytr) for k in ("Random Forest", "XGBoost", "LightGBM", "CatBoost")}
    preds["RF + boosting (vote: RF+XGB+LGBM+Cat)"] = np.mean([m.predict(Xte) for m in boosters.values()], axis=0)

    # Boosting RF residuals with LightGBM
    rf_oob = build("Random Forest", params).set_params(oob_score=True).fit(Xtr, ytr)
    lgb_res = build("LightGBM", params).fit(Xtr, ytr - rf_oob.oob_prediction_)
    preds["RF + LightGBM (boost RF residuals)"] = rf_oob.predict(Xte) + lgb_res.predict(Xte)

    return preds
