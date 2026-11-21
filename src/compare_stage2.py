"""compare raw GARCH, HAR, and VIX forecasts on the frozen stage 1 dataset"""

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from arch import arch_model

from .build_stage1 import ANNUALIZE, DATA, HORIZON, score


FIRST_YEAR = 2007
REFIT_SESSIONS = 21
VARIANCE_FLOOR = 1e-12
TARGET = "realized_target_variance"
MODELS = {
    "raw_garch": "Raw GARCH(1,1)",
    "raw_har": "Raw HAR",
    "raw_vix": "Raw VIX",
}
FORECAST_COLUMNS = [f"{model}_forecast_variance" for model in MODELS]
PERIODS = {
    "2007–09": (2007, 2009),
    "2010–19": (2010, 2019),
    "2020–21": (2020, 2021),
    "2022–end": (2022, None),
}


def fit_garch(history):
    observed = history.dropna()
    fit = arch_model(observed, mean="Zero", p=1, q=1, vol="GARCH", dist="normal").fit(
        disp="off", show_warning=False
    )
    if fit.convergence_flag:
        raise RuntimeError(f"GARCH fit failed on {history.index[-1].date()}: {fit.optimization_result.message}")
    omega, alpha, beta = fit.params[["omega", "alpha[1]", "beta[1]"]]
    next_variance = fit.forecast(horizon=1).variance.iloc[-1, 0]
    # advance the state through any unobserved sessions after the last fitted return
    trailing_missing = len(history) - history.index.get_loc(observed.index[-1]) - 1
    for _ in range(trailing_missing):
        next_variance = omega + (alpha + beta) * next_variance
    return omega, alpha, beta, next_variance


def garch_forecast(next_variance, omega, persistence, steps):
    total = 0.0
    for _ in range(steps):
        total += next_variance
        next_variance = omega + persistence * next_variance
    return ANNUALIZE * total / 10_000


def har_features(returns):
    squared = returns**2
    features = pd.DataFrame({"intercept": 1.0}, index=returns.index)
    for days in (1, 5, 22):
        trailing = 252 * squared.rolling(days).mean()
        features[f"log_rv_{days}"] = np.log(trailing.clip(lower=VARIANCE_FLOOR))
    return features


def build_forecasts(daily, first_year=FIRST_YEAR):
    dates = daily.index
    first = dates.searchsorted(pd.Timestamp(year=first_year, month=1, day=1))
    if first == 0 or first == len(dates):
        raise ValueError(f"Expected training history before {first_year} and forecast origins in {first_year}")
    # extend the session schedule so the last origins also have a 30-day forecast
    last_end = dates[-1] + HORIZON
    calendar = xcals.get_calendar("XNYS", start=dates[0], end=last_end)
    sessions = calendar.sessions
    if not dates.equals(sessions[:len(dates)]):
        raise ValueError("Stage 1 origins must match the scheduled XNYS sessions")
    steps = sessions.searchsorted(dates + HORIZON, side="right") - np.arange(len(dates)) - 1

    returns = daily["daily_log_return"]
    percent_returns = 100 * returns
    features = har_features(returns)
    training = features.assign(log_target=np.log(daily[TARGET].clip(lower=VARIANCE_FLOOR))).dropna()
    training_end = daily.loc[training.index, "target_end_date"]

    garch, har = [], []
    for pos in range(first, len(dates)):
        origin = dates[pos]
        if (pos - first) % REFIT_SESSIONS == 0:
            omega, alpha, beta, next_variance = fit_garch(percent_returns.iloc[:pos + 1])
            mature = training.loc[training_end <= origin]
            if len(mature) < features.shape[1]:
                raise ValueError(f"Too few mature HAR training targets at {origin.date()}")
            coefficients = np.linalg.lstsq(mature[features.columns], mature["log_target"], rcond=None)[0]
        else:
            observed_return = percent_returns.iloc[pos]
            squared_shock = observed_return**2 if pd.notna(observed_return) else next_variance
            next_variance = omega + alpha * squared_shock + beta * next_variance

        garch.append(garch_forecast(next_variance, omega, alpha + beta, steps[pos]))
        har.append(np.exp(features.iloc[pos].to_numpy() @ coefficients))

    forecasts = daily.iloc[first:][["target_end_date", TARGET, FORECAST_COLUMNS[2]]].copy()
    forecasts[FORECAST_COLUMNS[0]] = garch
    forecasts[FORECAST_COLUMNS[1]] = har
    forecasts["scored"] = scoring_mask(forecasts)
    return forecasts[["target_end_date", TARGET, *FORECAST_COLUMNS, "scored"]]


def scoring_mask(forecasts):
    return (
        np.isfinite(forecasts[TARGET])
        & np.isfinite(forecasts[FORECAST_COLUMNS]).all(axis=1)
        & forecasts[FORECAST_COLUMNS].gt(0).all(axis=1)
    )


def period_samples(shared):
    for period, (first, last) in PERIODS.items():
        yield period, shared.loc[str(first):str(last) if last else None]


def compare(forecasts, cutoff):
    shared = forecasts.loc[forecasts["scored"]]
    rows = []
    for period, sample in [("Overall", shared), *period_samples(shared)]:
        for model, column in zip(MODELS, FORECAST_COLUMNS):
            result = score(sample, cutoff, forecast_column=column)
            result.insert(0, "model", model)
            result.insert(0, "period", period)
            rows.append(result)
    return pd.concat(rows, ignore_index=True)


def plot_comparison(shared, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(12, 8), layout="constrained")
    realized = 100 * np.sqrt(shared[TARGET])
    for ax, (title, column) in zip(axes, zip(MODELS.values(), FORECAST_COLUMNS)):
        ax.plot(shared.index, realized, color="#35383d", linewidth=0.85, label="Forward realized")
        ax.plot(
            shared.index, 100 * np.sqrt(shared[column]),
            color="#2166ac", linewidth=0.8, label="Forecast",
        )
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", color="#e3e6e9")
        ax.set_axisbelow(True)
        ax.legend(loc="upper right", frameon=False, ncols=2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Forecast origin")
    axes[-1].set_ylim(bottom=0)
    fig.supylabel("Annualized volatility (%)")
    fig.suptitle("S&P 500 · 30-calendar-day volatility")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    processed = DATA / "processed"
    daily = pd.read_csv(
        processed / "stage1_daily.csv", index_col="date", parse_dates=["date", "target_end_date"]
    )
    cutoff = pd.Timestamp(pd.read_csv(processed / "stage1_metrics.csv").at[0, "snapshot_cutoff"])
    forecasts = build_forecasts(daily)
    metrics = compare(forecasts, cutoff)
    forecasts.to_csv(processed / "stage2_forecasts.csv")
    metrics.to_csv(processed / "stage2_metrics.csv", index=False)
    shared = forecasts.loc[forecasts["scored"]]
    plot_comparison(shared, processed / "stage2_comparison.png")

    warmup = daily.loc[daily.index < forecasts.index[0], "daily_log_return"]
    print(
        f"Warm-up: {warmup.index[0].date()} to {warmup.index[-1].date()}, "
        f"{warmup.count()} daily returns. First refit: {forecasts.index[0].date()}."
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
