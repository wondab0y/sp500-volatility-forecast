"""compare a 3-state volatility HMM with the saved walk-forward benchmarks"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from .build_stage1 import ANNUALIZE, DATA, score
from .compare_stage2 import REFIT_SESSIONS, TARGET, period_samples, scheduled_horizon_steps, scoring_mask


STATES = ("low", "medium", "high")
PROBABILITIES = [f"{state}_probability" for state in STATES]
VARIANCES = [f"{state}_daily_variance" for state in STATES]
FILTERED = "raw_hmm_forecast_variance"
STATIONARY = "stationary_hmm_forecast_variance"
SCALED_VIX = "scaled_vix_forecast_variance"
MODELS = {
    "raw_hmm": FILTERED,
    "raw_garch": "raw_garch_forecast_variance",
    "stationary_hmm": STATIONARY,
    "raw_har": "raw_har_forecast_variance",
    "raw_vix": "raw_vix_forecast_variance",
    "scaled_vix": SCALED_VIX,
}
MAX_ITERATIONS = 300
FIT_TOLERANCE = 1e-4


class ZeroMeanGaussianHMM(GaussianHMM):
    def _needs_sufficient_statistics_for_mean(self):
        # hmmlearn 0.3.3 also collects state occupancy here for the variance denominator.
        return True


def observed_sequences(returns):
    positions = np.flatnonzero(np.isfinite(returns))
    blocks = np.split(positions, np.flatnonzero(np.diff(positions) != 1) + 1)
    return returns[positions, None], [len(block) for block in blocks if len(block)]


def fit_hmm(returns, previous):
    observations, lengths = observed_sequences(returns)
    if len(observations) < len(STATES):
        raise ValueError("Too few observed returns to fit the HMM.")
    model = ZeroMeanGaussianHMM(
        n_components=3, covariance_type="diag", random_state=42,
        params="tc", init_params="", covars_prior=0.0, covars_weight=1.0,
        implementation="scaling", n_iter=MAX_ITERATIONS, tol=FIT_TOLERANCE,
    )
    model.startprob_ = np.full(3, 1 / 3)
    model.means_ = np.zeros((3, 1))
    if previous is None:
        transition = np.full((3, 3), 0.025)
        np.fill_diagonal(transition, 0.95)
        variances = np.quantile(observations**2, [0.30, 0.70, 0.95])
        model.transmat_ = transition
        model.covars_ = np.maximum(variances[:, None], 1e-10)
    else:
        model.transmat_ = previous.transmat_.copy()
        model.covars_ = previous.covars_[:, :, 0].copy()
    model.fit(observations, lengths)
    order = np.argsort(model.covars_[:, 0, 0])
    model.transmat_ = model.transmat_[np.ix_(order, order)]
    model.covars_ = model.covars_[order, :, 0]
    return model


def observe(prior, value, variances):
    if not np.isfinite(value):
        return prior
    log_likelihood = -0.5 * (np.log(variances) + value**2 / variances)
    weights = prior * np.exp(log_likelihood - log_likelihood.max())
    return weights / weights.sum()


def filter_history(returns, start, transition, variances):
    probabilities = observe(start, returns[0], variances)
    for value in returns[1:]:
        probabilities = observe(probabilities @ transition, value, variances)
    return probabilities


def forward_variance(probabilities, transition, variances, steps):
    # each row starts from a different time-t distribution under the same model
    total = np.zeros(len(probabilities))
    for _ in range(steps):
        probabilities = probabilities @ transition
        total += probabilities @ variances
    return ANNUALIZE * total


def build_forecasts(returns, first_origin):
    dates = returns.index
    steps = scheduled_horizon_steps(dates)
    first = dates.get_loc(first_origin)
    if first == 0:
        raise ValueError("Expected daily training history before the first Stage 2 origin.")
    values = returns.to_numpy()
    model = None
    rows = []
    for pos in range(first, len(dates)):
        if (pos - first) % REFIT_SESSIONS == 0:
            fit_date = dates[pos]
            model = fit_hmm(values[: pos + 1], model)
            transition = model.transmat_
            variances = model.covars_[:, 0, 0]
            stationary = model.get_stationary_distribution()
            np.testing.assert_allclose(stationary @ transition, stationary, atol=1e-10, rtol=0)
            np.testing.assert_allclose(transition.sum(axis=1), 1, atol=1e-10, rtol=0)
            if (stationary < -1e-10).any() or not (np.isfinite(variances) & (variances > 0)).all():
                raise ValueError(f"Invalid HMM parameters at {fit_date.date()}.")
            stationary = stationary.clip(min=0)
            stationary /= stationary.sum()
            probability = filter_history(values[: pos + 1], model.startprob_, transition, variances)
            last_gain = np.diff(model.monitor_.history)[-1]
            hit_limit = model.monitor_.iter == MAX_ITERATIONS and last_gain >= FIT_TOLERANCE
            likelihood_decreased = last_gain < -1e-8
        else:
            # missing close-to-close returns leave the transition-only prior intact
            probability = observe(probability @ transition, values[pos], variances)

        filtered, control = forward_variance(np.vstack([probability, stationary]), transition, variances, steps[pos])
        rows.append((
            fit_date, model.monitor_.iter, hit_limit, likelihood_decreased,
            filtered, control, *probability, *variances,
        ))
    forecasts = pd.DataFrame(
        rows, index=dates[first:],
        columns=[
            "fit_date", "fit_iterations", "fit_hit_iteration_limit", "fit_likelihood_decreased",
            FILTERED, STATIONARY, *PROBABILITIES, *VARIANCES,
        ],
    )
    probabilities = forecasts[PROBABILITIES]
    np.testing.assert_allclose(probabilities.sum(axis=1, skipna=False), 1, atol=1e-10, rtol=0)
    if not (np.isfinite(probabilities) & probabilities.ge(0)).all(axis=None):
        raise ValueError("Filtered state probabilities must be finite and nonnegative.")
    variances = forecasts[[FILTERED, STATIONARY]]
    if not (np.isfinite(variances) & variances.gt(0)).all(axis=None):
        raise ValueError("HMM forecasts must be positive and finite.")
    return forecasts


def compare(shared, cutoff):
    tables = []
    for period, sample in [("Overall", shared), *period_samples(shared)]:
        for model, column in MODELS.items():
            row = score(sample, cutoff, forecast_column=column)
            if row.at[0, "scored_origins"] != len(sample):
                raise ValueError(f"{model} has different scoring dates in {period}.")
            tables.append(row.assign(period=period, model=model))
    scores = pd.concat(tables, ignore_index=True)
    return scores[["period", "model", *scores.columns[:-2]]]


def state_summary(shared):
    levels = 100 * np.sqrt(252 * shared[VARIANCES])
    probabilities = shared[PROBABILITIES]
    most_likely = probabilities.to_numpy().argmax(axis=1)
    return pd.DataFrame({
        "state": STATES,
        "median_annualized_volatility_pct": levels.median().to_numpy(),
        "last_scored_annualized_volatility_pct": levels.iloc[-1].to_numpy(),
        "mean_filtered_probability": probabilities.mean().to_numpy(),
        "most_likely_share": np.bincount(most_likely, minlength=len(STATES)) / len(shared),
    })


def plot_comparison(shared, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    for column, label, color, width in (
        (TARGET, "Forward realized", "#35383d", 0.95),
        (MODELS["raw_garch"], "Raw GARCH(1,1)", "#5943c2", 0.85),
        (FILTERED, "Raw HMM", "#2166ac", 0.85),
    ):
        ax.plot(shared.index, 100 * np.sqrt(shared[column]), color=color, linewidth=width, label=label)
    ax.set_title("S&P 500 · 30-calendar-day volatility", loc="left", fontweight="bold")
    ax.set(xlabel="Forecast origin", ylabel="Annualized volatility (%)", ylim=(0, None))
    ax.grid(axis="y", color="#e3e6e9")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, ncols=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    processed = DATA / "processed"
    daily = pd.read_csv(processed / "stage1_daily.csv", index_col="date", parse_dates=["date"])
    stage2 = pd.read_csv(processed / "stage2_forecasts.csv", index_col="date", parse_dates=["date"])
    stage3 = pd.read_csv(processed / "stage3_forecasts.csv", index_col="date", parse_dates=["date"])
    cutoff = pd.Timestamp(pd.read_csv(processed / "stage2_metrics.csv")["snapshot_cutoff"].iloc[0])
    if not scoring_mask(stage2).equals(stage2["scored"]):
        raise ValueError("The saved Stage 2 scoring mask does not match its forecasts.")

    hmm = build_forecasts(daily["daily_log_return"], stage2.index[0])
    forecasts = stage2.join(stage3[[SCALED_VIX]]).join(hmm)
    forecasts["scored"] &= np.isfinite(forecasts[SCALED_VIX]) & forecasts[SCALED_VIX].gt(0)
    shared = forecasts.loc[forecasts["scored"]]
    if not shared.index.equals(stage3.index[stage3["scored"]]):
        raise ValueError("The common scoring dates do not match the saved Stage 3 dates.")
    scores = compare(shared, cutoff)
    states = state_summary(shared)
    forecasts.to_csv(processed / "stage4_forecasts.csv")
    scores.to_csv(processed / "stage4_scores.csv", index=False)
    states.to_csv(processed / "stage4_states.csv", index=False)
    plot_comparison(shared, processed / "stage4_comparison.png")
    fits = hmm.drop_duplicates("fit_date")
    print(
        f"{len(fits)} fits. Iteration limit: {fits['fit_hit_iteration_limit'].sum()}. "
        f"Final likelihood decreased: {fits['fit_likelihood_decreased'].sum()}."
    )
    print(scores.to_string(index=False))
    print(states.to_string(index=False))


if __name__ == "__main__":
    main()
