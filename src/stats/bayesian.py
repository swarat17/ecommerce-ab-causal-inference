"""
Bayesian A/B testing using PyMC Beta-Binomial model.

Model:
    p_control   ~ Beta(1, 1)   (uniform prior)
    p_treatment ~ Beta(1, 1)
    conversions ~ Binomial(n, p)

Posterior inference via MCMC (NUTS sampler).
"""
import numpy as np
import pymc as pm
import arviz as az

from src.utils.logger import get_logger

logger = get_logger(__name__)


class BayesianABTest:

    def __init__(self):
        self._trace = None
        self._n_control = None
        self._n_treatment = None

    def fit_conversion(
        self,
        n_control: int,
        conv_control: int,
        n_treatment: int,
        conv_treatment: int,
        tune: int = 500,
        draws: int = 1000,
        chains: int = 2,
        random_seed: int = 42,
    ) -> "BayesianABTest":
        """
        Fit Beta-Binomial model to conversion data.

        Stores posterior trace. Returns self for chaining.
        Logs R-hat diagnostic — values < 1.1 indicate convergence.
        """
        self._n_control = n_control
        self._n_treatment = n_treatment

        with pm.Model():
            p_control = pm.Beta("p_control", alpha=1, beta=1)
            p_treatment = pm.Beta("p_treatment", alpha=1, beta=1)

            pm.Binomial("obs_control", n=n_control, p=p_control, observed=conv_control)
            pm.Binomial("obs_treatment", n=n_treatment, p=p_treatment, observed=conv_treatment)

            _ = pm.Deterministic("delta", p_treatment - p_control)

            self._trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                random_seed=random_seed,
                progressbar=False,
            )

        # Log convergence diagnostics
        rhat = az.rhat(self._trace)
        max_rhat = float(max(
            rhat["p_control"].values.max(),
            rhat["p_treatment"].values.max(),
        ))
        if max_rhat > 1.1:
            logger.warning(f"Convergence warning: max R-hat = {max_rhat:.3f} (should be < 1.1)")
        else:
            logger.info(f"Model converged. Max R-hat = {max_rhat:.3f}")

        return self

    def _posterior_control(self) -> np.ndarray:
        return self._trace.posterior["p_control"].values.flatten()

    def _posterior_treatment(self) -> np.ndarray:
        return self._trace.posterior["p_treatment"].values.flatten()

    def probability_treatment_better(self) -> float:
        """
        P(treatment conversion rate > control conversion rate).

        The Bayesian analogue of a p-value — directly interpretable by stakeholders.
        """
        p_c = self._posterior_control()
        p_t = self._posterior_treatment()
        return float(np.mean(p_t > p_c))

    def expected_loss(self, threshold: float = 0.0) -> float:
        """
        Expected conversion rate lost if you choose the wrong variant.

        Decision rule: stop when expected_loss < 0.001 (0.1%).
        Uses the "opportunity loss" formulation:
            E[max(p_control - p_treatment, threshold)]
        """
        p_c = self._posterior_control()
        p_t = self._posterior_treatment()
        loss = np.maximum(p_c - p_t, threshold)
        return float(np.mean(loss))

    def credible_interval(self, alpha: float = 0.05) -> tuple[float, float]:
        """
        Highest Density Interval (HDI) on the delta = treatment - control.

        Returns (lower, upper) of the (1-alpha) HDI.
        """
        delta = self._posterior_treatment() - self._posterior_control()
        hdi = az.hdi(delta, hdi_prob=1 - alpha)
        return float(hdi[0]), float(hdi[1])

    def plot_posteriors(self):
        """
        Plotly figure showing posterior distributions for both variants.

        - KDE curves for p_control and p_treatment
        - Vertical line at delta=0 (no effect)
        - Annotation showing P(treatment > control)
        """
        import plotly.graph_objects as go
        from scipy.stats import gaussian_kde

        p_c = self._posterior_control()
        p_t = self._posterior_treatment()
        prob = self.probability_treatment_better()

        x_min = min(p_c.min(), p_t.min()) - 0.005
        x_max = max(p_c.max(), p_t.max()) + 0.005
        x = np.linspace(x_min, x_max, 300)

        kde_c = gaussian_kde(p_c)(x)
        kde_t = gaussian_kde(p_t)(x)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=kde_c, mode="lines", name="Control",
            line=dict(color="steelblue", width=2),
            fill="tozeroy", fillcolor="rgba(70,130,180,0.2)",
        ))
        fig.add_trace(go.Scatter(
            x=x, y=kde_t, mode="lines", name="Treatment",
            line=dict(color="darkorange", width=2),
            fill="tozeroy", fillcolor="rgba(255,140,0,0.2)",
        ))

        fig.add_vline(
            x=0, line_dash="dash", line_color="gray",
            annotation_text="No effect", annotation_position="top right",
        )

        ci_lo, ci_hi = self.credible_interval()
        fig.update_layout(
            title=f"Posterior Distributions — P(treatment > control) = {prob:.1%}",
            xaxis_title="Conversion Rate",
            yaxis_title="Density",
            legend=dict(orientation="h"),
            annotations=[dict(
                x=0.98, y=0.95, xref="paper", yref="paper",
                text=f"95% HDI on Δ: [{ci_lo:.4f}, {ci_hi:.4f}]",
                showarrow=False,
            )],
        )
        return fig
