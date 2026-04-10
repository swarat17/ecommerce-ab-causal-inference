"""
Streamlit dashboard for the A/B Testing Platform.

Four pages:
  1. Experiment Overview  — list all experiments, create new ones
  2. Results & Tests      — statistical results, Bayesian, corrections
  3. Propensity Diagnostics — covariate balance, IPW-adjusted vs raw lift
  4. Power Analysis       — sample size planning, stop experiment

Run with:
    streamlit run frontend/app.py
"""
from __future__ import annotations

import os
import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

from frontend.helpers import (
    days_to_significance,
    format_ci,
    format_lift,
    format_p_value,
    significance_label,
    srm_banner,
)

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="A/B Testing Platform",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

def _sidebar():
    st.sidebar.title("🧪 A/B Platform")
    page = st.sidebar.radio(
        "Navigate",
        [
            "Experiment Overview",
            "Results & Tests",
            "Propensity Diagnostics",
            "Power Analysis",
        ],
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(f"API: `{API_BASE}`")
    return page


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _get(path: str) -> dict | list | None:
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to API at {API_BASE}. Make sure FastAPI is running.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API error: {e}")
        return None


def _post(path: str, payload: dict) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=60)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to API at {API_BASE}.")
        return None
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        st.error(f"API error: {detail}")
        return None


# ---------------------------------------------------------------------------
# Page 1: Experiment Overview
# ---------------------------------------------------------------------------

def page_overview():
    st.title("Experiment Overview")

    data = _get("/experiments")
    if data is None:
        return

    if not data:
        st.info("No experiments found. Create one below.")
    else:
        df = pd.DataFrame(data)[
            ["experiment_id", "name", "status", "start_date", "n_users"]
        ]
        df.columns = ["ID", "Name", "Status", "Start Date", "Users"]
        st.dataframe(df, use_container_width=True)

    # ---------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Create New Experiment")
    with st.form("new_experiment"):
        exp_id = st.text_input("Experiment ID (e.g., exp_C)", placeholder="exp_C")
        exp_name = st.text_input("Name", placeholder="Button Color Test")
        exp_desc = st.text_area("Description", placeholder="Test whether a red CTA button increases conversion")
        col1, col2 = st.columns(2)
        with col1:
            control_name = st.text_input("Control variant name", value="control")
        with col2:
            treatment_name = st.text_input("Treatment variant name", value="treatment")
        submitted = st.form_submit_button("Create Experiment")

    if submitted:
        if not exp_id or not exp_name:
            st.warning("Experiment ID and Name are required.")
        else:
            # Write directly to PostgreSQL via setup script pattern
            # (API does not have a create endpoint — use a lightweight direct insert)
            try:
                from sqlalchemy import create_engine, text
                db_url = os.getenv("DB_URL", "postgresql://postgres:postgres@localhost:5432/ab_testing")
                engine = create_engine(db_url)
                with engine.begin() as conn:
                    conn.execute(text("""
                        INSERT INTO experiments (experiment_id, name, description, start_date, status)
                        VALUES (:eid, :name, :desc, NOW(), 'running')
                        ON CONFLICT (experiment_id) DO NOTHING
                    """), {"eid": exp_id, "name": exp_name, "desc": exp_desc})
                    conn.execute(text("""
                        INSERT INTO variants (variant_id, experiment_id, name, is_control)
                        VALUES (:cid, :eid, :cname, TRUE)
                        ON CONFLICT (variant_id) DO NOTHING
                    """), {"cid": f"{exp_id}_control", "eid": exp_id, "cname": control_name})
                    conn.execute(text("""
                        INSERT INTO variants (variant_id, experiment_id, name, is_control)
                        VALUES (:tid, :eid, :tname, FALSE)
                        ON CONFLICT (variant_id) DO NOTHING
                    """), {"tid": f"{exp_id}_treatment", "eid": exp_id, "tname": treatment_name})
                engine.dispose()
                st.success(f"Experiment '{exp_id}' created! Refresh the page to see it.")
            except Exception as e:
                st.error(f"Failed to create experiment: {e}")


# ---------------------------------------------------------------------------
# Page 2: Results & Tests
# ---------------------------------------------------------------------------

def page_results():
    st.title("Results & Statistical Tests")

    experiments = _get("/experiments")
    if not experiments:
        st.info("No experiments found.")
        return

    exp_ids = [e["experiment_id"] for e in experiments]
    selected = st.selectbox("Select Experiment", exp_ids)

    col1, col2 = st.columns(2)
    with col1:
        correction_method = st.selectbox(
            "Multiple Testing Correction",
            ["benjamini_hochberg", "bonferroni"],
            format_func=lambda x: "Benjamini-Hochberg (FDR)" if x == "benjamini_hochberg" else "Bonferroni (FWER)",
        )
    with col2:
        use_cache = st.checkbox("Use cached results (faster)", value=True)

    run_col, stop_col = st.columns([3, 1])
    with run_col:
        run_btn = st.button("Run Analysis", type="primary")
    with stop_col:
        stop_btn = st.button("Stop Experiment", type="secondary")

    if stop_btn:
        result = _post(f"/experiments/{selected}/stop", {})
        if result:
            st.warning(f"Experiment '{selected}' has been stopped.")

    result = None
    if run_btn:
        with st.spinner("Running full analysis pipeline..."):
            result = _post(
                f"/experiments/{selected}/analyze",
                {
                    "experiment_id": selected,
                    "correction_method": correction_method,
                    "use_propensity": False,
                },
            )
    elif use_cache:
        result = _get(f"/experiments/{selected}/results")

    if result is None:
        st.info("Click 'Run Analysis' to compute results, or enable 'Use cached results'.")
        return

    _render_results(result, correction_method)


def _render_results(result: dict, correction_method: str):
    # --- SRM Banner ---
    srm = result.get("srm_check")
    if srm:
        banner = srm_banner(srm["srm_detected"])
        color = banner["color"]
        msg = banner["message"]
        if color == "red":
            st.error(msg)
        else:
            st.success(msg)

    st.markdown(f"**Experiment:** `{result['experiment_id']}` — {result['name']} ({result['status']})")
    if result.get("computed_at"):
        st.caption(f"Computed at: {result['computed_at']}")

    # --- Variant Metric Cards ---
    variants = result.get("variants", [])
    if variants:
        st.subheader("Variant Metrics")
        cols = st.columns(len(variants))
        for i, v in enumerate(variants):
            with cols[i]:
                st.metric(label=f"Variant: **{v['variant']}**", value="")
                st.write(f"Users: **{v.get('n_users', 'N/A'):,}**")
                cr = v.get("conversion_rate")
                if cr is not None:
                    st.write(f"Conversion Rate: **{cr:.2%}**")
                rev = v.get("mean_revenue")
                if rev is not None:
                    st.write(f"Mean Revenue: **${rev:.2f}**")
                cart = v.get("add_to_cart_rate")
                if cart is not None:
                    st.write(f"Add-to-Cart: **{cart:.2%}**")

    # --- Frequentist Results ---
    freq = result.get("frequentist", [])
    if freq:
        st.subheader("Frequentist Tests")
        method_label = "BH" if correction_method == "benjamini_hochberg" else "Bonferroni"
        rows = []
        for f in freq:
            rows.append({
                "Metric": f["metric"],
                "p-value": format_p_value(f["p_value"]),
                "Absolute Lift": format_lift(f["absolute_lift"]) if f.get("absolute_lift") is not None else "N/A",
                "Relative Lift": format_lift(f["relative_lift"]) if f.get("relative_lift") is not None else "N/A",
                "95% CI": format_ci(f.get("ci_lower"), f.get("ci_upper")),
                f"Significant ({method_label})": significance_label(f["significant_corrected"], method_label),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # --- Bayesian Results ---
    bayes = result.get("bayesian")
    if bayes:
        st.subheader("Bayesian Analysis (Beta-Binomial)")
        b_col1, b_col2, b_col3 = st.columns(3)
        with b_col1:
            prob = bayes["probability_treatment_better"]
            st.metric("P(Treatment > Control)", f"{prob:.1%}")
        with b_col2:
            loss = bayes["expected_loss"]
            st.metric("Expected Loss", f"{loss:.4f}")
        with b_col3:
            ci_lo = bayes["credible_interval_lower"]
            ci_hi = bayes["credible_interval_upper"]
            st.metric("95% Credible Interval", format_ci(ci_lo, ci_hi))

        # Posterior plot
        fig = go.Figure()
        x = [ci_lo + i * (ci_hi - ci_lo) / 100 for i in range(101)]
        fig.add_trace(go.Scatter(x=x, y=[1.0] * 101, fill="tozeroy",
                                  fillcolor="rgba(99,110,250,0.2)",
                                  line=dict(color="rgba(99,110,250,0.8)"),
                                  name="Credible Interval"))
        fig.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="No effect")
        fig.update_layout(
            title="Credible Interval on Lift (Treatment - Control)",
            xaxis_title="Lift",
            yaxis_visible=False,
            height=250,
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Sample Size Analysis ---
    ssa = result.get("sample_size_analysis")
    if ssa:
        st.subheader("Sample Size Analysis")
        s_col1, s_col2, s_col3 = st.columns(3)
        with s_col1:
            st.metric("Baseline Conversion Rate", f"{ssa['baseline_rate']:.2%}")
        with s_col2:
            st.metric("Minimum Detectable Effect", f"{ssa['minimum_detectable_effect']:.2%}")
        with s_col3:
            current_n = ssa["current_n_per_variant"]
            required_n = ssa["required_n_for_80pct_power"]
            delta = current_n - required_n
            st.metric(
                "Current vs Required n",
                f"{current_n:,}",
                delta=f"{delta:+,} vs required {required_n:,}",
                delta_color="normal" if delta >= 0 else "inverse",
            )


# ---------------------------------------------------------------------------
# Page 3: Propensity Diagnostics
# ---------------------------------------------------------------------------

def page_propensity():
    st.title("Propensity Score Diagnostics")
    st.markdown(
        """
        The XGBoost propensity model estimates P(treatment | user features).
        If the CV AUC is > 0.6, high-value users are disproportionately in one variant —
        Inverse Probability Weighting (IPW) corrects for this.
        """
    )

    experiments = _get("/experiments")
    if not experiments:
        st.info("No experiments found.")
        return

    exp_ids = [e["experiment_id"] for e in experiments]
    selected = st.selectbox("Select Experiment", exp_ids, key="prop_exp")
    correction_method = st.selectbox(
        "Correction Method",
        ["benjamini_hochberg", "bonferroni"],
        key="prop_correction",
    )

    run_btn = st.button("Run Propensity Analysis", type="primary")

    result = None
    if run_btn:
        with st.spinner("Training XGBoost propensity model and running IPW... (may take 30–60 seconds)"):
            result = _post(
                f"/experiments/{selected}/analyze",
                {
                    "experiment_id": selected,
                    "correction_method": correction_method,
                    "use_propensity": True,
                },
            )

    if result is None:
        st.info("Click 'Run Propensity Analysis' to train the XGBoost model and compute IPW-adjusted results.")
        return

    prop = result.get("propensity")
    if prop is None:
        st.warning(
            "Propensity analysis was not returned. This usually means user_features.parquet "
            "is missing. Run `scripts/run_experiment.py` first to generate features."
        )
    else:
        # --- CV AUC & Imbalance Detection ---
        auc = prop["cv_auc"]
        imbalanced = prop["imbalance_detected"]
        if imbalanced:
            st.error(f"⚠️ Covariate imbalance detected! CV AUC = {auc:.3f} (threshold: 0.6). IPW adjustment applied.")
        else:
            st.success(f"✅ Groups appear balanced. CV AUC = {auc:.3f}")

        # --- Lift Comparison ---
        st.subheader("Unadjusted vs IPW-Adjusted Lift")
        adj_col1, adj_col2, adj_col3 = st.columns(3)
        with adj_col1:
            st.metric("Unadjusted Lift", format_lift(prop["unadjusted_lift"]))
        with adj_col2:
            st.metric("IPW-Adjusted Lift", format_lift(prop["adjusted_lift"]))
        with adj_col3:
            mag = prop["adjustment_magnitude"]
            st.metric(
                "Adjustment Magnitude",
                f"{mag:.4f}",
                help="How much IPW changed the raw result. Large values mean the raw result was misleading.",
            )

        if mag > 0.01:
            st.warning(
                f"IPW adjustment changed conversion lift by **{mag:.4f}** (>{0.01:.2f}). "
                "The raw result may be misleading due to covariate imbalance."
            )

        # SMD plot placeholder (actual plot requires user_features.parquet + fitted model)
        st.subheader("Covariate Balance — Standardised Mean Difference (SMD)")
        st.info(
            "The SMD plot is generated by `PropensityModel.plot_covariate_balance()`. "
            "To render it, run the propensity model directly and embed the returned Plotly figure. "
            "This requires user_features.parquet to be present in data/processed/."
        )

    # --- Frequentist results (with propensity context) ---
    freq = result.get("frequentist", [])
    if freq:
        st.subheader("Frequentist Test Results (for reference)")
        rows = []
        for f in freq:
            rows.append({
                "Metric": f["metric"],
                "p-value": format_p_value(f["p_value"]),
                "Absolute Lift": format_lift(f["absolute_lift"]) if f.get("absolute_lift") is not None else "N/A",
                "95% CI": format_ci(f.get("ci_lower"), f.get("ci_upper")),
                "Significant": significance_label(f["significant_corrected"]),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ---------------------------------------------------------------------------
# Page 4: Power Analysis & Planning
# ---------------------------------------------------------------------------

def page_power():
    st.title("Power Analysis & Experiment Planning")
    st.markdown(
        "Use this page to determine whether your experiment has sufficient statistical power "
        "and how many more days/users are needed."
    )

    experiments = _get("/experiments")
    if not experiments:
        st.info("No experiments found.")
        return

    exp_ids = [e["experiment_id"] for e in experiments]
    selected = st.selectbox("Select Experiment", exp_ids, key="power_exp")
    exp_meta = next((e for e in experiments if e["experiment_id"] == selected), {})

    st.subheader("Parameters")
    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        mde = st.number_input(
            "Minimum Detectable Effect (MDE)",
            min_value=0.001,
            max_value=0.999,
            value=0.01,
            step=0.001,
            format="%.3f",
            help="Smallest lift you care about detecting (absolute, e.g. 0.01 = 1pp)",
        )
    with p_col2:
        metric = st.selectbox("Metric", ["conversion_rate", "revenue_per_user", "add_to_cart_rate"])
    with p_col3:
        days_running = st.number_input("Days experiment has been running", min_value=1, value=14)

    fetch_btn = st.button("Calculate Power", type="primary")
    if fetch_btn:
        with st.spinner("Fetching power analysis..."):
            pa = _get(f"/experiments/{selected}/power-analysis?metric={metric}&mde={mde}")

        if pa:
            r_col1, r_col2, r_col3, r_col4 = st.columns(4)
            with r_col1:
                st.metric("Baseline Conversion Rate", f"{pa['baseline_rate']:.2%}")
            with r_col2:
                current_n = pa["current_n_per_variant"]
                st.metric("Current n / variant", f"{current_n:,}")
            with r_col3:
                required_n = pa["required_n_per_variant"]
                st.metric("Required n / variant (80% power)", f"{required_n:,}")
            with r_col4:
                days_left = days_to_significance(required_n, current_n, days_running)
                if days_left == 0:
                    st.metric("Days to significance", "Ready now ✅")
                else:
                    st.metric("Est. days remaining", f"~{days_left} days")

            if current_n >= required_n:
                st.success(
                    f"✅ Experiment has sufficient power at MDE = {mde:.1%}. "
                    f"You may analyze results and decide whether to ship."
                )
            else:
                pct_done = min(current_n / max(required_n, 1) * 100, 100)
                st.warning(
                    f"⚠️ Experiment is underpowered ({pct_done:.0f}% of required n reached). "
                    "Do not make a shipping decision yet."
                )
                st.progress(pct_done / 100)

            # Sample size curve
            st.subheader("Required Sample Size vs MDE")
            try:
                from src.stats.frequentist import required_sample_size
                mde_values = [0.005 + i * 0.005 for i in range(40)]
                baseline = pa["baseline_rate"] if pa["baseline_rate"] > 0 else 0.05
                n_values = [required_sample_size(baseline_rate=baseline, mde=m) for m in mde_values]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=[m * 100 for m in mde_values], y=n_values,
                                          mode="lines+markers", name="Required n"))
                fig.add_vline(x=mde * 100, line_dash="dash", line_color="red",
                              annotation_text=f"Current MDE ({mde:.1%})")
                fig.add_hline(y=current_n, line_dash="dot", line_color="blue",
                              annotation_text=f"Current n ({current_n:,})")
                fig.update_layout(
                    title="Sample Size Required for 80% Power",
                    xaxis_title="MDE (%)",
                    yaxis_title="Required n per variant",
                    height=350,
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.info("Install src package to render the sample size curve.")

    # --- Stop Experiment ---
    st.markdown("---")
    st.subheader("Stop Experiment")
    st.caption(
        "Stopping an experiment sets its status to 'stopped' and records the end date. "
        "Only stop when you have sufficient power or when SRM is detected."
    )
    if st.button("Stop Experiment", type="secondary"):
        confirm = st.checkbox("I confirm I want to stop this experiment")
        if confirm:
            result = _post(f"/experiments/{selected}/stop", {})
            if result:
                st.warning(f"Experiment '{selected}' has been stopped.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    page = _sidebar()

    if page == "Experiment Overview":
        page_overview()
    elif page == "Results & Tests":
        page_results()
    elif page == "Propensity Diagnostics":
        page_propensity()
    elif page == "Power Analysis":
        page_power()


if __name__ == "__main__":
    main()
