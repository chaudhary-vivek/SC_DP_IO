# demand_model.py
"""
Production-quality demand forecasting (mean prediction) WITHOUT argparse.

Design patterns used
- Strategy: swap forecasting algorithm without changing orchestration code
- Template Method: consistent fit/predict flow with overridable pieces
- Facade: DemandForecaster provides a simple API for the rest of the system
- Persistence: save/load a single artifact via joblib

Expected columns (minimum):
- day (int-like)
- store_id (str-like)
- promo_flag (bool/int-like)

Training additionally requires:
- demand (numeric)

Optional extra columns (if present):
- warehouse_id (categorical)
- any numeric exogenous features (price, discount, etc.)

Outputs:
- predicted_demand (int, clipped >= 0)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LOGGER = logging.getLogger("demand_model")


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class DemandModelConfig:
    col_day: str = "day"
    col_store: str = "store_id"
    col_promo: str = "promo_flag"
    col_target: str = "demand"
    pred_col: str = "predicted_demand"

    # Seasonality periods in "days"
    season_periods: Tuple[int, ...] = (7, 30)

    # Preprocessing
    scale_numeric: bool = True
    onehot_min_frequency: Optional[int] = None  # e.g., 10 to bucket rare categories

    # Model defaults (reasonable baseline)
    random_state: int = 42


# -----------------------------
# Validation / coercion
# -----------------------------
def _ensure_columns(df: pd.DataFrame, cols: Sequence[str], df_name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def _coerce_day(x: pd.Series, col: str) -> pd.Series:
    s = pd.to_numeric(x, errors="coerce")
    if s.isna().any():
        bad = x[s.isna()].head(5).tolist()
        raise ValueError(f"Column '{col}' has non-numeric values (sample): {bad}")
    return s.astype(int)


def _coerce_promo(x: pd.Series, col: str) -> pd.Series:
    if x.dtype == bool:
        return x.astype(int)

    if x.dtype == object:
        lx = x.astype(str).str.strip().str.lower()
        mapped = lx.map({"true": 1, "false": 0, "1": 1, "0": 0})
        if mapped.isna().any():
            mapped = pd.to_numeric(lx, errors="coerce")
        mapped = mapped.fillna(0)
        return mapped.clip(0, 1).astype(int)

    s = pd.to_numeric(x, errors="coerce").fillna(0)
    return s.clip(0, 1).astype(int)


def _coerce_target(x: pd.Series, col: str) -> pd.Series:
    s = pd.to_numeric(x, errors="coerce")
    if s.isna().any():
        bad = x[s.isna()].head(5).tolist()
        raise ValueError(f"Target column '{col}' has non-numeric values (sample): {bad}")
    return s.clip(lower=0).astype(float)


def _validate_train_df(df: pd.DataFrame, cfg: DemandModelConfig) -> pd.DataFrame:
    _ensure_columns(df, [cfg.col_day, cfg.col_store, cfg.col_promo, cfg.col_target], "train df")
    out = df.copy()
    out[cfg.col_day] = _coerce_day(out[cfg.col_day], cfg.col_day)
    out[cfg.col_store] = out[cfg.col_store].astype(str)
    out[cfg.col_promo] = _coerce_promo(out[cfg.col_promo], cfg.col_promo)
    out[cfg.col_target] = _coerce_target(out[cfg.col_target], cfg.col_target)
    return out


def _validate_predict_df(df: pd.DataFrame, cfg: DemandModelConfig) -> pd.DataFrame:
    _ensure_columns(df, [cfg.col_day, cfg.col_store, cfg.col_promo], "predict df")
    out = df.copy()
    out[cfg.col_day] = _coerce_day(out[cfg.col_day], cfg.col_day)
    out[cfg.col_store] = out[cfg.col_store].astype(str)
    out[cfg.col_promo] = _coerce_promo(out[cfg.col_promo], cfg.col_promo)
    return out


# -----------------------------
# Feature Engineering
# -----------------------------
class FeatureEngineer:
    """Pure feature engineering (stateless)."""

    def __init__(self, cfg: DemandModelConfig):
        self.cfg = cfg

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        day = out[self.cfg.col_day].astype(float)

        # Seasonality sin/cos for each configured period
        for p in self.cfg.season_periods:
            angle = 2.0 * math.pi * (day / float(p))
            out[f"sin_{p}"] = np.sin(angle)
            out[f"cos_{p}"] = np.cos(angle)

        # day-of-week signals derived from "day" index
        dow = (out[self.cfg.col_day] % 7).astype(int)
        out["dow"] = dow
        angle_dow = 2.0 * math.pi * (dow / 7.0)
        out["sin_dow"] = np.sin(angle_dow)
        out["cos_dow"] = np.cos(angle_dow)

        return out


def _infer_feature_columns(df: pd.DataFrame, cfg: DemandModelConfig) -> Tuple[List[str], List[str]]:
    """
    Feature selection policy:
    - Always numeric: day, promo_flag + engineered
    - Always categorical: store_id
    - Include extra numeric columns automatically
    - Include extra categorical columns automatically (e.g., warehouse_id)
    """
    engineered = ["sin_dow", "cos_dow", "dow"] + \
                [f"sin_{p}" for p in cfg.season_periods] + \
                [f"cos_{p}" for p in cfg.season_periods]

    numeric_cols = [cfg.col_day, cfg.col_promo] + engineered
    categorical_cols = [cfg.col_store]

    excluded = {cfg.col_target, cfg.pred_col}

    for c in df.columns:
        if c in excluded or c in numeric_cols or c in categorical_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)

    # de-dupe
    def _dedupe(xs: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return _dedupe(numeric_cols), _dedupe(categorical_cols)


# -----------------------------
# Strategy Pattern: model choice
# -----------------------------
class ModelStrategy(Protocol):
    """Algorithm strategy interface."""

    def build(
        self,
        numeric_features: Sequence[str],
        categorical_features: Sequence[str],
        cfg: DemandModelConfig,
    ) -> Pipeline: ...


class HGBRStrategy:
    """Default strategy: HistGradientBoostingRegressor + one-hot for categoricals."""

    def __init__(
        self,
        *,
        max_depth: int = 6,
        learning_rate: float = 0.08,
        max_iter: int = 400,
        min_samples_leaf: int = 30,
        l2_regularization: float = 0.0,
    ):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization

    def build(
        self,
        numeric_features: Sequence[str],
        categorical_features: Sequence[str],
        cfg: DemandModelConfig,
    ) -> Pipeline:
        numeric_steps = []
        if cfg.scale_numeric:
            numeric_steps.append(("scaler", StandardScaler(with_mean=True, with_std=True)))
        numeric_transformer = Pipeline(steps=numeric_steps) if numeric_steps else "passthrough"

        # OneHotEncoder: bucket unknown categories safely
        ohe_kwargs = dict(handle_unknown="ignore", sparse_output=False)
        if cfg.onehot_min_frequency is not None:
            ohe_kwargs["min_frequency"] = cfg.onehot_min_frequency

        pre = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, list(numeric_features)),
                ("cat", OneHotEncoder(**ohe_kwargs), list(categorical_features)),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

        model = HistGradientBoostingRegressor(
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            random_state=cfg.random_state,
        )

        return Pipeline([("preprocess", pre), ("model", model)])


# -----------------------------
# Facade + Template Method
# -----------------------------
class DemandForecaster:
    """
    Facade for training/predicting demand in a production-friendly way.

    Template Method:
      fit() -> validate -> feature engineer -> build pipeline -> train
      predict() -> validate -> feature engineer -> align columns -> predict
    """

    def __init__(
        self,
        cfg: Optional[DemandModelConfig] = None,
        strategy: Optional[ModelStrategy] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.cfg = cfg or DemandModelConfig()
        self.strategy = strategy or HGBRStrategy()
        self.fe = FeatureEngineer(self.cfg)
        self.log = logger or LOGGER

        self.pipeline: Optional[Pipeline] = None
        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = []

        self.is_fitted: bool = False
        self.train_day_min: Optional[int] = None
        self.train_day_max: Optional[int] = None

    # --- Template steps (internal) ---
    def _prepare_train(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        dfv = _validate_train_df(df, self.cfg)
        dff = self.fe.transform(dfv)

        self.numeric_features, self.categorical_features = _infer_feature_columns(dff, self.cfg)

        X = dff.drop(columns=[self.cfg.col_target])
        y = dff[self.cfg.col_target].to_numpy(dtype=float)

        self.train_day_min = int(dff[self.cfg.col_day].min())
        self.train_day_max = int(dff[self.cfg.col_day].max())

        return X, y

    def _prepare_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        dfv = _validate_predict_df(df, self.cfg)
        dff = self.fe.transform(dfv)

        # Ensure any missing columns expected by the pipeline exist
        needed = set(self.numeric_features + self.categorical_features)
        for c in needed:
            if c not in dff.columns:
                if c in self.numeric_features:
                    dff[c] = 0.0
                else:
                    dff[c] = "__MISSING__"
        return dff

    # --- Public API ---
    def fit(self, df: pd.DataFrame) -> "DemandForecaster":
        X, y = self._prepare_train(df)

        self.pipeline = self.strategy.build(self.numeric_features, self.categorical_features, self.cfg)

        self.log.info(
            "Training demand model | rows=%d | day_range=[%s,%s] | num=%d | cat=%d",
            len(X),
            self.train_day_min,
            self.train_day_max,
            len(self.numeric_features),
            len(self.categorical_features),
        )
        self.pipeline.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted or self.pipeline is None:
            raise RuntimeError("Model is not fitted. Call fit() or load() first.")

        dff = self._prepare_predict(df)
        yhat = self.pipeline.predict(dff)

        out = df.copy()
        out[self.cfg.pred_col] = np.maximum(0, np.rint(yhat)).astype(int)
        return out

    def evaluate_time_split(self, df: pd.DataFrame, train_end_day: int) -> Dict[str, float]:
        """
        Train on day < train_end_day, test on day >= train_end_day.
        Returns MAE, RMSE and row counts.
        """
        dfv = _validate_train_df(df, self.cfg)

        train_df = dfv[dfv[self.cfg.col_day] < train_end_day].copy()
        test_df = dfv[dfv[self.cfg.col_day] >= train_end_day].copy()

        if train_df.empty or test_df.empty:
            raise ValueError("Time split produced empty train or test set. Adjust train_end_day.")

        self.fit(train_df)

        preds = self.predict(test_df[[self.cfg.col_day, self.cfg.col_store, self.cfg.col_promo] + [
            c for c in test_df.columns
            if c not in {self.cfg.col_target} and c not in {self.cfg.col_day, self.cfg.col_store, self.cfg.col_promo}
        ]])

        merged = test_df[[self.cfg.col_day, self.cfg.col_store, self.cfg.col_target]].merge(
            preds[[self.cfg.col_day, self.cfg.col_store, self.cfg.pred_col]],
            on=[self.cfg.col_day, self.cfg.col_store],
            how="left",
        )

        y_true = merged[self.cfg.col_target].to_numpy(dtype=float)
        y_pred = merged[self.cfg.pred_col].to_numpy(dtype=float)

        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)

        return {
            "train_end_day": float(train_end_day),
            "mae": mae,
            "rmse": rmse,
            "n_train": float(len(train_df)),
            "n_test": float(len(test_df)),
        }

    def save(self, path: str | Path) -> None:
        if not self.is_fitted or self.pipeline is None:
            raise RuntimeError("Model is not fitted; nothing to save.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        artifact = {
            "config": asdict(self.cfg),
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "train_day_min": self.train_day_min,
            "train_day_max": self.train_day_max,
            "strategy": self.strategy.__class__.__name__,  # informational
            "pipeline": self.pipeline,
        }
        joblib.dump(artifact, path)
        self.log.info("Saved demand model artifact to %s", str(path))

    @classmethod
    def load(cls, path: str | Path, logger: Optional[logging.Logger] = None) -> "DemandForecaster":
        path = Path(path)
        artifact = joblib.load(path)

        cfg = DemandModelConfig(**artifact["config"])
        obj = cls(cfg=cfg, strategy=HGBRStrategy(), logger=logger)  # default strategy for metadata
        obj.numeric_features = artifact["numeric_features"]
        obj.categorical_features = artifact["categorical_features"]
        obj.train_day_min = artifact.get("train_day_min")
        obj.train_day_max = artifact.get("train_day_max")
        obj.pipeline = artifact["pipeline"]
        obj.is_fitted = True

        (logger or LOGGER).info("Loaded demand model artifact from %s", str(path))
        return obj


# -----------------------------
# Optional: lightweight "factory"
# -----------------------------
def build_default_forecaster(cfg: Optional[DemandModelConfig] = None) -> DemandForecaster:
    """
    Factory function: one place to standardize default choices.
    """
    return DemandForecaster(cfg=cfg or DemandModelConfig(), strategy=HGBRStrategy())


# -----------------------------
# Minimal usage example
# -----------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    df = pd.read_csv("data/demand_history.csv")

    forecaster = build_default_forecaster()
    metrics = forecaster.evaluate_time_split(df, train_end_day=150)
    print("metrics:", metrics)

    # Train full (example) and save
    forecaster.fit(df[df["day"] < 150])
    forecaster.save("outputs/demand_model.joblib")

    # Predict a single day
    day = 170
    to_pred = df[df["day"] == day][["day", "store_id", "promo_flag", "warehouse_id"]].copy()
    preds = forecaster.predict(to_pred)
    print(preds.head())
