"""
FastAPI application for the A/B Testing Platform.

Endpoints:
    POST /experiments/{experiment_id}/analyze  — run full analysis pipeline
    GET  /experiments                          — list all experiments
    GET  /experiments/{experiment_id}/results  — latest cached results
    GET  /experiments/{experiment_id}/power-analysis — sample size / MDE
    POST /experiments/{experiment_id}/stop     — mark experiment stopped
    GET  /health                               — health check
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from sqlalchemy import create_engine, text

load_dotenv()

from src.api.schemas import (
    BayesianResult,
    ExperimentRequest,
    ExperimentResult,
    ExperimentSummary,
    FrequentistTestResult,
    NoveltyResult,
    PropensityResult,
    SampleSizeAnalysis,
    SRMResult,
    VariantMetrics,
)
from src.experiments.assignment import ExperimentAssigner
from src.experiments.metrics import MetricComputer
from src.stats.frequentist import (
    minimum_detectable_effect,
    required_sample_size,
    two_proportion_z_test,
    two_sample_t_test,
)
from src.stats.corrections import apply_corrections
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="A/B Testing Platform",
    description="End-to-end experimentation platform with frequentist, Bayesian, and propensity-adjusted analysis.",
    version="1.0.0",
)

DB_URL = os.getenv("DB_URL", "postgresql://postgres:postgres@localhost:5432/ab_testing")


def _get_engine():
    return create_engine(DB_URL)


def _load_experiment_meta(experiment_id: str, engine) -> dict:
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM experiments WHERE experiment_id = :eid"),
            {"eid": experiment_id},
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found")
    return dict(row._mapping)


def _load_assignments(experiment_id: str, engine) -> pd.DataFrame:
    with engine.connect() as conn:
        df = pd.read_sql(
            "SELECT ua.user_id, ua.experiment_id, v.name AS variant "
            "FROM user_assignments ua "
            "JOIN variants v ON ua.variant_id = v.variant_id "
            "WHERE ua.experiment_id = :eid",
            conn,
            params={"eid": experiment_id},
        )
    return df


def _load_events(engine) -> pd.DataFrame:
    processed_path = os.getenv("PROCESSED_DATA_PATH", "data/processed")
    parquet_path = f"{processed_path}/events.parquet"
    try:
        return pd.read_parquet(parquet_path)
    except FileNotFoundError:
        logger.warning(f"events.parquet not found at {parquet_path}, using empty DataFrame")
        return pd.DataFrame(columns=["user_id", "event_type", "price", "user_session", "date"])


def _persist_results(experiment_id: str, variant_id: str, results: list[dict], engine) -> None:
    with engine.begin() as conn:
        for r in results:
            conn.execute(
                text("""
                    INSERT INTO experiment_results
                        (experiment_id, variant_id, metric_name, value, ci_lower, ci_upper, p_value)
                    VALUES (:eid, :vid, :metric, :value, :ci_lower, :ci_upper, :pval)
                """),
                {
                    "eid": experiment_id,
                    "vid": variant_id,
                    "metric": r.get("metric_name"),
                    "value": r.get("value"),
                    "ci_lower": r.get("ci_lower"),
                    "ci_upper": r.get("ci_upper"),
                    "pval": r.get("p_value"),
                },
            )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/experiments", response_model=list[ExperimentSummary])
def list_experiments():
    engine = _get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT e.experiment_id, e.name, e.status, e.start_date, e.end_date, "
            "COUNT(DISTINCT ua.user_id) AS n_users "
            "FROM experiments e "
            "LEFT JOIN user_assignments ua ON e.experiment_id = ua.experiment_id "
            "GROUP BY e.experiment_id, e.name, e.status, e.start_date, e.end_date"
        )).fetchall()
    engine.dispose()
    return [ExperimentSummary(**dict(r._mapping)) for r in rows]


@app.get("/experiments/{experiment_id}/results", response_model=ExperimentResult)
def get_results(experiment_id: str):
    engine = _get_engine()
    meta = _load_experiment_meta(experiment_id, engine)

    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT * FROM experiment_results WHERE experiment_id = :eid ORDER BY computed_at DESC"),
            {"eid": experiment_id},
        ).fetchall()
    engine.dispose()

    if not rows:
        raise HTTPException(status_code=404, detail="No results found. Run /analyze first.")

    return ExperimentResult(
        experiment_id=experiment_id,
        name=meta["name"],
        status=meta["status"],
        computed_at=rows[0]._mapping["computed_at"],
    )


@app.post("/experiments/{experiment_id}/analyze", response_model=ExperimentResult)
def analyze_experiment(experiment_id: str, request: ExperimentRequest):
    engine = _get_engine()
    meta = _load_experiment_meta(experiment_id, engine)
    assignments_df = _load_assignments(experiment_id, engine)
    events_df = _load_events(engine)

    if assignments_df.empty:
        raise HTTPException(status_code=400, detail="No user assignments found for this experiment.")

    assigner = ExperimentAssigner()
    mc = MetricComputer()

    # --- SRM check ---
    srm = assigner.check_srm(assignments_df)
    srm_result = SRMResult(**srm)

    # --- Metrics ---
    all_metrics = mc.compute_all_metrics(events_df, assignments_df, experiment_id)
    conv_df = all_metrics["conversion_rate"]
    rev_df = all_metrics["revenue_per_user"]
    cart_df = all_metrics["add_to_cart_rate"]

    variant_metrics = []
    for _, row in conv_df.iterrows():
        rev_row = rev_df[rev_df["variant"] == row["variant"]].iloc[0]
        cart_row = cart_df[cart_df["variant"] == row["variant"]].iloc[0]
        variant_metrics.append(VariantMetrics(
            variant=row["variant"],
            n_users=int(row["n_users"]),
            conversion_rate=float(row["conversion_rate"]),
            mean_revenue=float(rev_row["mean_revenue"]),
            add_to_cart_rate=float(cart_row["add_to_cart_rate"]),
        ))

    # --- Frequentist tests ---
    variants = assignments_df["variant"].unique().tolist()
    control_variant = next((v for v in variants if v == "control"), variants[0])
    treatment_variant = next((v for v in variants if v != control_variant), variants[-1])

    ctrl_conv = conv_df[conv_df["variant"] == control_variant].iloc[0]
    trt_conv = conv_df[conv_df["variant"] == treatment_variant].iloc[0]
    ctrl_rev = rev_df[rev_df["variant"] == control_variant].iloc[0]
    trt_rev = rev_df[rev_df["variant"] == treatment_variant].iloc[0]

    z_result = two_proportion_z_test(
        n_control=int(ctrl_conv["n_users"]),
        conv_control=int(ctrl_conv["n_converters"]),
        n_treatment=int(trt_conv["n_users"]),
        conv_treatment=int(trt_conv["n_converters"]),
    )

    ctrl_users = assignments_df[assignments_df["variant"] == control_variant]["user_id"]
    trt_users = assignments_df[assignments_df["variant"] == treatment_variant]["user_id"]
    ctrl_revenues = events_df[
        (events_df["event_type"] == "purchase") & events_df["user_id"].isin(ctrl_users)
    ].groupby("user_id")["price"].sum().reindex(ctrl_users, fill_value=0).values
    trt_revenues = events_df[
        (events_df["event_type"] == "purchase") & events_df["user_id"].isin(trt_users)
    ].groupby("user_id")["price"].sum().reindex(trt_users, fill_value=0).values

    t_result = two_sample_t_test(ctrl_revenues, trt_revenues)

    raw_results = {
        "conversion_rate": {**z_result, "p_value": z_result["p_value"]},
        "revenue_per_user": {**t_result, "p_value": t_result["p_value"]},
    }
    corrected = apply_corrections(raw_results, method=request.correction_method)

    freq_results = [
        FrequentistTestResult(
            metric="conversion_rate",
            p_value=z_result["p_value"],
            significant=z_result["significant"],
            significant_corrected=corrected["conversion_rate"]["significant_corrected"],
            absolute_lift=z_result["absolute_lift"],
            relative_lift=z_result["relative_lift"],
            ci_lower=z_result["ci_lower"],
            ci_upper=z_result["ci_upper"],
        ),
        FrequentistTestResult(
            metric="revenue_per_user",
            p_value=t_result["p_value"],
            significant=t_result["significant"],
            significant_corrected=corrected["revenue_per_user"]["significant_corrected"],
            ci_lower=t_result["ci_lower"],
            ci_upper=t_result["ci_upper"],
        ),
    ]

    # --- Bayesian ---
    bayesian_result = None
    try:
        from src.stats.bayesian import BayesianABTest
        bayes = BayesianABTest()
        bayes.fit_conversion(
            n_control=int(ctrl_conv["n_users"]),
            conv_control=int(ctrl_conv["n_converters"]),
            n_treatment=int(trt_conv["n_users"]),
            conv_treatment=int(trt_conv["n_converters"]),
            tune=300, draws=500, chains=2,
        )
        ci_lo, ci_hi = bayes.credible_interval()
        bayesian_result = BayesianResult(
            probability_treatment_better=bayes.probability_treatment_better(),
            expected_loss=bayes.expected_loss(),
            credible_interval_lower=ci_lo,
            credible_interval_upper=ci_hi,
        )
    except Exception as e:
        logger.warning(f"Bayesian analysis failed: {e}")

    # --- Propensity ---
    propensity_result = None
    if request.use_propensity:
        try:
            from src.causal.propensity import PropensityModel
            from src.causal.ipw import compute_weights, adjusted_conversion_rate

            user_features_path = os.getenv("PROCESSED_DATA_PATH", "data/processed") + "/user_features.parquet"
            user_features = pd.read_parquet(user_features_path)

            prop_model = PropensityModel()
            cv_auc = prop_model.train(user_features, assignments_df, experiment_id)

            assignments_df["treatment"] = (assignments_df["variant"] != control_variant).astype(int)
            merged = user_features.merge(assignments_df[["user_id", "treatment"]], on="user_id", how="inner")

            propensity_scores = prop_model.predict_propensity(merged)
            weights = compute_weights(propensity_scores, merged["treatment"].values)

            purchase_map = (
                events_df[events_df["event_type"] == "purchase"]
                .groupby("user_id").size().gt(0).astype(int)
            )
            conversions = merged["user_id"].map(purchase_map).fillna(0).values

            adj = adjusted_conversion_rate(conversions, weights, merged["treatment"].values)
            propensity_result = PropensityResult(
                cv_auc=cv_auc,
                imbalance_detected=cv_auc > 0.6,
                adjustment_magnitude=adj["adjustment_magnitude"],
                unadjusted_lift=adj["unadjusted_lift"],
                adjusted_lift=adj["adjusted_lift"],
            )
        except Exception as e:
            logger.warning(f"Propensity analysis failed: {e}")

    # --- Sample size analysis ---
    baseline_rate = float(ctrl_conv["conversion_rate"]) if float(ctrl_conv["conversion_rate"]) > 0 else 0.05
    n_per_variant = int((int(ctrl_conv["n_users"]) + int(trt_conv["n_users"])) / 2)
    mde = minimum_detectable_effect(
        n_control=int(ctrl_conv["n_users"]),
        n_treatment=int(trt_conv["n_users"]),
        baseline_rate=baseline_rate,
    )
    req_n = required_sample_size(baseline_rate=baseline_rate, mde=max(mde, 0.001))
    sample_size_result = SampleSizeAnalysis(
        baseline_rate=baseline_rate,
        current_n_per_variant=n_per_variant,
        minimum_detectable_effect=mde,
        required_n_for_80pct_power=req_n,
    )

    # --- Persist results ---
    try:
        ctrl_variant_id = _get_variant_id(experiment_id, control_variant, engine)
        trt_variant_id = _get_variant_id(experiment_id, treatment_variant, engine)
        _persist_results(experiment_id, ctrl_variant_id, [
            {"metric_name": "conversion_rate", "value": float(ctrl_conv["conversion_rate"]),
             "p_value": z_result["p_value"], "ci_lower": z_result["ci_lower"], "ci_upper": z_result["ci_upper"]},
        ], engine)
        _persist_results(experiment_id, trt_variant_id, [
            {"metric_name": "conversion_rate", "value": float(trt_conv["conversion_rate"]),
             "p_value": z_result["p_value"], "ci_lower": z_result["ci_lower"], "ci_upper": z_result["ci_upper"]},
        ], engine)
    except Exception as e:
        logger.warning(f"Failed to persist results: {e}")

    engine.dispose()

    return ExperimentResult(
        experiment_id=experiment_id,
        name=meta["name"],
        status=meta["status"],
        computed_at=datetime.utcnow(),
        srm_check=srm_result,
        variants=variant_metrics,
        frequentist=freq_results,
        bayesian=bayesian_result,
        propensity=propensity_result,
        novelty_check=None,
        sample_size_analysis=sample_size_result,
    )


@app.get("/experiments/{experiment_id}/power-analysis")
def power_analysis(
    experiment_id: str,
    metric: str = Query(default="conversion_rate"),
    mde: float = Query(..., gt=0, lt=1),
):
    engine = _get_engine()
    _load_experiment_meta(experiment_id, engine)
    assignments_df = _load_assignments(experiment_id, engine)
    events_df = _load_events(engine)
    engine.dispose()

    mc = MetricComputer()
    conv_df = mc.compute_conversion_rate(events_df, assignments_df, experiment_id)

    variants = assignments_df["variant"].unique().tolist()
    control_variant = next((v for v in variants if v == "control"), variants[0])
    ctrl_row = conv_df[conv_df["variant"] == control_variant]

    baseline_rate = float(ctrl_row["conversion_rate"].iloc[0]) if not ctrl_row.empty else 0.05
    req_n = required_sample_size(baseline_rate=baseline_rate, mde=mde)

    return {
        "experiment_id": experiment_id,
        "metric": metric,
        "mde": mde,
        "baseline_rate": baseline_rate,
        "required_n_per_variant": req_n,
        "current_n_per_variant": int(assignments_df.shape[0] / max(len(variants), 1)),
    }


@app.post("/experiments/{experiment_id}/stop")
def stop_experiment(experiment_id: str):
    engine = _get_engine()
    _load_experiment_meta(experiment_id, engine)
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE experiments SET status = 'stopped', end_date = NOW() WHERE experiment_id = :eid"),
            {"eid": experiment_id},
        )
    engine.dispose()
    return {"experiment_id": experiment_id, "status": "stopped"}


def _get_variant_id(experiment_id: str, variant_name: str, engine) -> str:
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT variant_id FROM variants WHERE experiment_id = :eid AND name = :name"),
            {"eid": experiment_id, "name": variant_name},
        ).fetchone()
    return row[0] if row else variant_name
