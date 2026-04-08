"""
XGBoost propensity score model for covariate imbalance correction.

The propensity score P(treatment | covariates) is estimated by training a
binary classifier on user features. If CV AUC >> 0.5, the groups are imbalanced
and IPW adjustment is warranted.

Ideal AUC ≈ 0.5 (model cannot distinguish treatment from control).
AUC > 0.6 → covariate imbalance detected, log a warning.
"""
import os
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from src.utils.logger import get_logger

logger = get_logger(__name__)

MODELS_PATH = Path(os.getenv("MODELS_PATH", "models"))

NUMERIC_FEATURES = [
    "total_sessions",
    "total_revenue",
    "days_active",
    "avg_price_viewed",
    "total_views",
    "total_carts",
    "total_purchases",
    "avg_session_length",
]
CATEGORICAL_FEATURES = ["favorite_category"]


def _build_pipeline() -> Pipeline:
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline([("preprocessor", preprocessor), ("classifier", clf)])


class PropensityModel:

    def __init__(self):
        self._pipeline: Pipeline | None = None
        self._feature_names: list[str] | None = None
        self._cv_auc: float | None = None

    def train(
        self,
        user_features_df: pd.DataFrame,
        assignments_df: pd.DataFrame,
        experiment_id: str,
    ) -> float:
        """
        Train XGBoost propensity model.

        Target: 1 = treatment, 0 = control.
        Evaluates with 5-fold stratified CV AUC.
        Saves model to models/propensity_{experiment_id}.pkl.
        Registers in SageMaker if AWS credentials are available.

        Returns CV AUC score.
        """
        # Merge features with treatment labels
        label_map = assignments_df.copy()
        label_map["treatment"] = (label_map["variant"] != "control").astype(int)
        merged = user_features_df.merge(
            label_map[["user_id", "treatment"]], on="user_id", how="inner"
        )

        # Fill missing categoricals
        merged["favorite_category"] = merged["favorite_category"].fillna("unknown")
        for col in NUMERIC_FEATURES:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0.0)

        X = merged[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        y = merged["treatment"].values
        self._feature_names = NUMERIC_FEATURES + CATEGORICAL_FEATURES

        pipeline = _build_pipeline()

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        auc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        cv_auc = float(np.mean(auc_scores))
        self._cv_auc = cv_auc

        if cv_auc > 0.6:
            logger.warning(
                f"[{experiment_id}] Covariate imbalance detected! "
                f"CV AUC = {cv_auc:.3f} (threshold: 0.6). "
                "IPW adjustment is strongly recommended."
            )
        else:
            logger.info(
                f"[{experiment_id}] Groups appear balanced. CV AUC = {cv_auc:.3f}"
            )

        # Fit on full dataset
        pipeline.fit(X, y)
        self._pipeline = pipeline

        # Save locally
        MODELS_PATH.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_PATH / f"propensity_{experiment_id}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Model saved to {model_path}")

        # Register in SageMaker (best-effort)
        try:
            from scripts.deploy_model import register_model
            register_model(
                model_path=model_path,
                experiment_id=experiment_id,
                cv_auc=cv_auc,
                feature_names=self._feature_names,
            )
        except Exception as e:
            logger.warning(f"SageMaker registration skipped: {e}")

        return cv_auc

    def predict_propensity(self, user_features_df: pd.DataFrame) -> np.ndarray:
        """
        Return P(treatment | covariates) for each user.

        Returns array of shape (n_users,) with values in (0, 1).
        """
        if self._pipeline is None:
            raise RuntimeError("Model not trained. Call train() first.")

        df = user_features_df.copy()
        df["favorite_category"] = df["favorite_category"].fillna("unknown")
        for col in NUMERIC_FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)

        X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        return self._pipeline.predict_proba(X)[:, 1]

    def plot_covariate_balance(
        self,
        user_features_df: pd.DataFrame,
        assignments_df: pd.DataFrame,
    ):
        """
        Standardised Mean Difference (SMD) plot before and after IPW weighting.

        SMD = (mean_treatment - mean_control) / pooled_std

        Values < 0.1 (dashed line) indicate acceptable balance.
        Returns a Plotly figure.
        """
        import plotly.graph_objects as go
        from src.causal.ipw import compute_weights

        label_map = assignments_df.copy()
        label_map["treatment"] = (label_map["variant"] != "control").astype(int)
        merged = user_features_df.merge(
            label_map[["user_id", "treatment"]], on="user_id", how="inner"
        )
        for col in NUMERIC_FEATURES:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0.0)

        propensity = self.predict_propensity(merged)
        weights = compute_weights(propensity, merged["treatment"].values)

        smds_before, smds_after = [], []
        for col in NUMERIC_FEATURES:
            t_vals = merged.loc[merged["treatment"] == 1, col].values
            c_vals = merged.loc[merged["treatment"] == 0, col].values
            t_w = weights[merged["treatment"].values == 1]
            c_w = weights[merged["treatment"].values == 0]

            pooled_std = np.sqrt(
                (np.var(t_vals, ddof=1) + np.var(c_vals, ddof=1)) / 2
            )
            if pooled_std == 0:
                smds_before.append(0.0)
                smds_after.append(0.0)
                continue

            smd_before = abs(np.mean(t_vals) - np.mean(c_vals)) / pooled_std
            smd_after = (
                abs(np.average(t_vals, weights=t_w) - np.average(c_vals, weights=c_w))
                / pooled_std
            )
            smds_before.append(smd_before)
            smds_after.append(smd_after)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=smds_before, y=NUMERIC_FEATURES, mode="markers",
            name="Before IPW", marker=dict(color="tomato", size=10, symbol="circle"),
        ))
        fig.add_trace(go.Scatter(
            x=smds_after, y=NUMERIC_FEATURES, mode="markers",
            name="After IPW", marker=dict(color="steelblue", size=10, symbol="diamond"),
        ))
        fig.add_vline(x=0.1, line_dash="dash", line_color="gray",
                      annotation_text="SMD = 0.1 threshold")
        fig.update_layout(
            title="Covariate Balance: Standardised Mean Difference (SMD)",
            xaxis_title="Absolute SMD",
            yaxis_title="Covariate",
            legend=dict(orientation="h"),
        )
        return fig
