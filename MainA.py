#!/usr/bin/env python3
"""
Energy Efficiency Linear Regression
___________________________________
Train and evaluate linear models that predict a building's heating (Y1) or
cooling (Y2) load from eight design parameters contained in the UCI *Energy
Efficiency* data_set.

Features
~~~~~~~~
* Relative Compactness
* Surface Area (m²)
* Wall Area (m²)
* Roof Area (m²)
* Overall Height (m)
* Orientation (categorical, 2_5)
* Glazing Area (%)
* Glazing Area Distribution (categorical, 0_5)

Targets
~~~~~~~
* Y1 : Heating Load (kWh·m⁻²)
* Y2 : Cooling Load (kWh . m⁻²)

The script offers:
* reproducible train/test split
* 5_fold cross_validation
* optional hyper_parameter tuning for Ridge, Lasso and ElasticNet
* standardised coefficients for easy interpretation

Usage
~~~~~
    python energy_regression.py --data data.xlsx --sheet 1 --target Y1 --tune

Dependencies
~~~~~~~~~~~~
* pandas
* numpy
* scikit_learn 1.4+

"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
CV_FOLDS: int = 5

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Data utilities
# -----------------------------------------------------------------------------

def load_data(path: str | Path, sheet: str | int = 0) -> pd.DataFrame:
    """Load the Excel file and return a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_excel(path, sheet_name=sheet)
    return df

# -----------------------------------------------------------------------------
# Modelling helpers
# -----------------------------------------------------------------------------

def make_pipeline(regressor) -> Pipeline:
    """Return a (scaler ▸ regressor) pipeline with categorical passthrough."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("reg", regressor),
        ]
    )


def evaluate(model: Pipeline, X_train, X_test, y_train, y_test) -> None:
    """Print hold_out and cross_validated metrics."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    import numpy as np
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    r2 = r2_score(y_test, y_pred)
    logger.info("Hold_out RMSE: %.3f", rmse)
    logger.info("Hold_out R²  : %.3f", r2)
    X_full = pd.concat([X_train, X_test])
    y_full = pd.concat([y_train, y_test])

    cv_scores = cross_val_score(model, X_full, y_full, cv=5, scoring="neg_mean_squared_error")
    cv_rmse = np.sqrt(-cv_scores)

    cv_r2 = cross_val_score(model, X_full, y_full,cv=CV_FOLDS, scoring="r2")
    logger.info("CV RMSE      : %.3f ± %.3f", cv_rmse.mean(), cv_rmse.std())
    logger.info("CV R²        : %.3f", cv_r2.mean())

    if hasattr(model["reg"], "coef_"):
        # Standardised coefficients (already scaled)
        coef = pd.Series(model["reg"].coef_, index=X_train.columns).sort_values(key=np.abs, ascending=False)
        logger.info("\nStandardised coefficients:\n%s", coef.to_string())


def hyperparameter_tuning(X, y) -> None:
    """Grid_search Ridge, Lasso and ElasticNet; print best CV RMSE."""
    gridspec = {
        "ridge": {
            "model": Ridge(random_state=RANDOM_STATE),
            "params": {"reg__alpha": np.logspace(-3, 3, 13)},
        },
        "lasso": {
            "model": Lasso(random_state=RANDOM_STATE, max_iter=5000),
            "params": {"reg__alpha": np.logspace(-3, 3, 13)},
        },
        "elastic": {
            "model": ElasticNet(random_state=RANDOM_STATE, max_iter=5000),
            "params": {
                "reg__alpha": np.logspace(-3, 3, 13),
                "reg__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
            },
        },
    }

    for name, spec in gridspec.items():
        pipe = make_pipeline(spec["model"])
        grid = GridSearchCV(
            pipe,
            param_grid=spec["params"],
            cv=CV_FOLDS,
            scoring="neg_root_mean_squared_error",
        )
        grid.fit(X, y)
        logger.info("%s best CV RMSE: %.3f | params: %s",
                    name.capitalize(), -grid.best_score_, grid.best_params_)

# -----------------------------------------------------------------------------
# Main CLI
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Linear regression on the Energy Efficiency dataset.")
    parser.add_argument("--data", type=str, default="data.xlsx", help="Path to Excel file (or CSV).")
    parser.add_argument("--sheet", default="1", help="Worksheet name or index if using Excel.")
    parser.add_argument("--target", choices=["Y1", "Y2"], default="Y1", help="Target variable to predict.")
    parser.add_argument("--test_size", type=float, default=TEST_SIZE, help="Proportion of test set.")
    parser.add_argument("--tune", action="store_true", help="Run hyper‑parameter tuning for regularised models.")
    return parser.parse_args()


def main():
    args = parse_args()

    df = load_data(args.data, args.sheet)
    feature_cols = [c for c in df.columns if c not in ("Y1", "Y2")]
    X = df[feature_cols]
    y = df[args.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE, stratify=None
    )

    base_model = make_pipeline(LinearRegression())
    logger.info("\n========== Baseline Linear Regression ==========")
    evaluate(base_model, X_train, X_test, y_train, y_test)

    if args.tune:
        logger.info("\n========== Hyper_parameter Search (Ridge/Lasso/ElasticNet) ==========")
        hyperparameter_tuning(X, y)


if __name__ == "__main__":
    main()
