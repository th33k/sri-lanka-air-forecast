"""Model zoo: base models, hyperparameter search spaces, tuning, and metrics.

Defines:
- Baseline models: Persistence, 7-day mean, Ridge regression, Decision Tree
- Bagging ensemble models: Bagging (trees), Random Forest, Extra Trees
- Boosting ensemble models: AdaBoost, XGBoost, LightGBM, CatBoost
- Stacking ensemble: Ridge meta-learner over RF + XGBoost + LightGBM + CatBoost
- Randomized search (10 candidates x 4 TimeSeriesSplit folds) for every tuned model
- Regression evaluation metrics (RMSE, MAE, R²)
"""
import time
import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy.stats import randint, uniform
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from .data import SEED

TUNE_ITER, CV_SPLITS = 10, 4  # Uniform tuning budget across every tuned model
STACK = "Stacking (RF+XGB+LGBM+Cat)"
FAMILY = {
    "Ridge": "Baseline",
    "Decision Tree": "Baseline",
    "Bagging (trees)": "Bagging",
    "Random Forest": "Bagging",
    "Extra Trees": "Bagging",
    "AdaBoost": "Boosting",
    "XGBoost": "Boosting",
    "LightGBM": "Boosting",
    "CatBoost": "Boosting",
    STACK: "Stacking",
}

SPACE = {
    "Random Forest": dict(
        n_estimators=randint(100, 300),
        max_depth=randint(6, 20),
        min_samples_leaf=randint(2, 15),
        max_features=uniform(0.3, 0.6),
    ),
    "Extra Trees": dict(
        n_estimators=randint(100, 300),
        max_depth=randint(6, 20),
        min_samples_leaf=randint(2, 15),
        max_features=uniform(0.3, 0.7),
    ),
    "Bagging (trees)": dict(
        n_estimators=randint(20, 80),
        max_samples=uniform(0.5, 0.5),
        max_features=uniform(0.5, 0.5),
    ),
    "AdaBoost": dict(
        n_estimators=randint(40, 150),
        learning_rate=uniform(0.05, 0.95),
    ),
    "XGBoost": dict(
        n_estimators=randint(150, 500),
        max_depth=randint(2, 7),
        learning_rate=uniform(0.02, 0.15),
        subsample=uniform(0.6, 0.4),
        colsample_bytree=uniform(0.5, 0.5),
        min_child_weight=randint(1, 10),
    ),
    "LightGBM": dict(
        n_estimators=randint(150, 500),
        num_leaves=randint(7, 50),
        learning_rate=uniform(0.02, 0.15),
        subsample=uniform(0.6, 0.4),
        colsample_bytree=uniform(0.5, 0.5),
        min_child_samples=randint(10, 60),
    ),
    "CatBoost": dict(
        iterations=randint(200, 500),
        depth=randint(3, 8),
        learning_rate=uniform(0.03, 0.15),
        l2_leaf_reg=uniform(1, 9),
    ),
}


def base_models():
    """Initializes un-tuned base estimators with pinned random seed."""
    rs = dict(random_state=SEED)
    return {
        "Decision Tree": DecisionTreeRegressor(max_depth=6, min_samples_leaf=20, **rs),
        "Ridge": make_pipeline(StandardScaler(), Ridge()),
        "Bagging (trees)": BaggingRegressor(DecisionTreeRegressor(min_samples_leaf=3, **rs), n_jobs=-1, **rs),
        "Random Forest": RandomForestRegressor(n_jobs=-1, **rs),
        "Extra Trees": ExtraTreesRegressor(n_jobs=-1, **rs),
        "AdaBoost": AdaBoostRegressor(**rs),
        "XGBoost": XGBRegressor(n_jobs=-1, **rs),
        "LightGBM": LGBMRegressor(n_jobs=-1, verbose=-1, **rs),
        "CatBoost": CatBoostRegressor(verbose=0, allow_writing_files=False, **rs),
    }


def reg_metrics(y_true, y_pred):
    """Calculates standardized regression metrics: RMSE, MAE, R²."""
    yt, yp = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return dict(
        RMSE=float(np.sqrt(mean_squared_error(yt, yp))),
        MAE=float(mean_absolute_error(yt, yp)),
        R2=float(r2_score(yt, yp)),
    )


def fit_models(X, y, names):
    """Tunes (randomized search, TimeSeriesSplit on training data) and fits the named models."""
    models = base_models()
    cv = TimeSeriesSplit(n_splits=CV_SPLITS)
    tuned, best_params, train_time = {}, {}, {}
    for name in names:
        m = models[name]
        t0 = time.perf_counter()
        if name in SPACE:
            search = RandomizedSearchCV(
                m, SPACE[name], n_iter=TUNE_ITER, cv=cv, scoring="neg_root_mean_squared_error",
                random_state=SEED, n_jobs=-1, refit=True
            )
            search.fit(X, y)
            tuned[name] = search.best_estimator_
            best_params[name] = search.best_params_
        else:
            m.fit(X, y)
            tuned[name] = m
            best_params[name] = {}
        train_time[name] = time.perf_counter() - t0
    return tuned, best_params, train_time


def fit_stack(tuned, X, y):
    """Stacking: Ridge meta-learner on out-of-fold predictions of the tuned RF and boosters."""
    t0 = time.perf_counter()
    stack = StackingRegressor(
        estimators=[(k.lower().replace(" ", "_"), clone(tuned[k])) for k in ("Random Forest", "XGBoost", "LightGBM", "CatBoost")],
        final_estimator=RidgeCV(), cv=KFold(n_splits=5, shuffle=False), n_jobs=-1, passthrough=False
    )
    stack.fit(X, y)
    return stack, time.perf_counter() - t0


def fit_zoo(X, y):
    """Tunes and fits every single model, then the stacking ensemble."""
    tuned, best_params, train_time = fit_models(X, y, list(FAMILY)[:-1])
    tuned[STACK], train_time[STACK] = fit_stack(tuned, X, y)
    best_params[STACK] = {}
    return tuned, best_params, train_time
