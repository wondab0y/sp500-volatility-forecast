"""calibrate saved stage 2 forecasts"""

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from .build_stage1 import DATA, HORIZON, score
from .compare_stage2 import FORECAST_COLUMNS, MODELS, TARGET, period_samples, scoring_mask


MIN_MATURED = 126
UPDATE_SESSIONS = 21
MODEL_KEYS = ("raw_vix", "raw_garch", "raw_har")
MODEL_LABELS = ["VIX", "GARCH", "HAR"]
MULTIPLIERS = [f"{model.removeprefix('raw_')}_multiplier" for model in MODELS]
SCALED_COLUMNS = [column.replace("raw_", "scaled_", 1) for column in FORECAST_COLUMNS]


def calibrate(forecasts):
    dates = forecasts.index
    if not dates.is_monotonic_increasing or not dates.is_unique:
        raise ValueError("Expected sorted, unique Stage 2 forecast origins.")
    if not (forecasts["target_end_date"] == dates + HORIZON).all():
        raise ValueError("Stage 2 target end dates must be 30 calendar days after each origin.")
    common_mask = scoring_mask(forecasts)
    if not common_mask.equals(forecasts["scored"]):
        raise ValueError("The saved Stage 2 scoring mask does not match the forecast data.")

    common = forecasts.loc[common_mask]
    ends = common["target_end_date"]
    # include outcomes ending on this date, keep all scheduled origins for the update cadence
    counts = ends.searchsorted(dates, side="right")
    first = counts.searchsorted(MIN_MATURED)
    if first == len(dates):
        raise ValueError(f"Fewer than {MIN_MATURED} common forecast outcomes have matured.")
    updates = dates[first::UPDATE_SESSIONS]
    matured = counts[first::UPDATE_SESSIONS]
    ratios = common[FORECAST_COLUMNS].rdiv(common[TARGET], axis=0)
    history = ratios.expanding().mean().iloc[matured - 1].set_axis(updates)
    history.columns = MULTIPLIERS
    if not (np.isfinite(history) & history.gt(0)).all(axis=None):
        raise ValueError("Calibration multipliers must all be positive and finite.")
    history.insert(0, "last_matured_target_end_date", ends.iloc[matured - 1].to_numpy())
    history.insert(0, "last_matured_origin", common.index[matured - 1])
    history.insert(0, "matured_outcomes", matured)
    history.index.name = "calibration_date"

    result = forecasts.iloc[first:].copy()
    active = history.reindex(result.index, method="ffill")
    update_date = history.index.to_series().reindex(result.index, method="ffill")
    result.insert(0, "calibration_date", update_date)
    raw = result[FORECAST_COLUMNS]
    scaled = raw.to_numpy() * active[MULTIPLIERS].to_numpy()
    valid = np.isfinite(scaled) & (scaled > 0)
    if not (valid | raw.isna().to_numpy()).all():
        raise ValueError("An available scaled forecast is nonpositive or nonfinite.")
    if not valid[result["scored"].to_numpy()].all():
        raise ValueError("The raw and scaled variants do not have identical scoring dates.")
    result[SCALED_COLUMNS] = scaled
    return result, history


def compare(forecasts, cutoff):
    shared = forecasts.loc[forecasts["scored"]]
    rows = []
    for period, sample in [("Overall", shared), *period_samples(shared)]:
        for model, raw_column, scaled_column in zip(MODELS, FORECAST_COLUMNS, SCALED_COLUMNS):
            raw = score(sample, cutoff, forecast_column=raw_column).iloc[0].to_dict()
            scaled = score(sample, cutoff, forecast_column=scaled_column).iloc[0]
            if any(raw[key] != scaled[key] for key in ("scored_origins", "first_scored_date", "last_scored_date")):
                raise ValueError(f"Scoring dates disagree for {period} {model}.")
            row = {"period": period, "model": model, **raw}
            for metric in ("qlike", "mae_volatility_pp"):
                row[f"raw_{metric}"] = row.pop(metric)
                row[f"scaled_{metric}"] = scaled[metric]
                row[f"delta_{metric}"] = scaled[metric] - row[f"raw_{metric}"]
            rows.append(row)
    return pd.DataFrame(rows)


def plot_scores(scores, processed):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    overall = scores.loc[scores["period"].eq("Overall")].set_index("model").loc[list(MODEL_KEYS)]
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(10, 4.2))
    fig.suptitle("Raw and scaled 30-calendar-day forecasts", y=0.96, fontweight="bold", fontsize=16)
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.23, top=0.76, wspace=0.24)
    y = np.arange(len(overall))
    for ax, (metric, title, units) in zip(axes, (
        ("qlike", "QLIKE", "Loss on annualized decimal variance"),
        ("mae_volatility_pp", "MAE", "Annualized volatility percentage points"),
    )):
        raw, scaled = overall[f"raw_{metric}"], overall[f"scaled_{metric}"]
        ax.hlines(y, raw, scaled, color="#a5acb7")
        ax.scatter(raw, y, s=95, facecolors="white", edgecolors="#555f70", linewidths=1.6, label="Raw")
        ax.scatter(scaled, y, s=38, color="#2166ac", label="Scaled", zorder=3)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel(units)
        ax.grid(axis="x", color="#e3e6e9")
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, MODEL_LABELS)
    axes[0].set_ylim(len(y) - 0.5, -0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=2, frameon=False)
    fig.savefig(processed / "stage3_paired_scores.png", dpi=160)
    plt.close(fig)

    by_period = scores.loc[scores["period"].ne("Overall")]
    cells = by_period.pivot(index="model", columns="period", values="delta_qlike").loc[list(MODEL_KEYS)]
    cells = cells.reindex(columns=by_period["period"].unique())
    limit = max(abs(cells.to_numpy()).max(), 1e-12)
    fig, ax = plt.subplots(figsize=(9, 4.1), layout="constrained")
    fig.suptitle("Change in QLIKE after scaling", fontsize=16, fontweight="bold")
    ax.set_title("Scaled minus raw · negative values indicate improvement", loc="left", fontsize=10)
    image = ax.imshow(cells, cmap="RdBu_r", norm=TwoSlopeNorm(0, vmin=-limit, vmax=limit), aspect="auto")
    ax.set_xticks(range(len(cells.columns)), cells.columns)
    ax.set_yticks(range(len(cells.index)), MODEL_LABELS)
    for (row, col), value in np.ndenumerate(cells.to_numpy()):
        color = "white" if abs(value) > 0.65 * limit else "#182234"
        ax.text(col, row, f"{value:+.4f}", ha="center", va="center", color=color, fontweight="bold")
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    fig.colorbar(image, ax=ax, label="QLIKE change", shrink=0.84)
    fig.savefig(processed / "stage3_period_qlike.png", dpi=160)
    plt.close(fig)


def main():
    processed = DATA / "processed"
    forecasts = pd.read_csv(
        processed / "stage2_forecasts.csv", index_col="date", parse_dates=["date", "target_end_date"]
    )
    stage2_scores = pd.read_csv(processed / "stage2_metrics.csv")
    cutoff = pd.Timestamp(stage2_scores["snapshot_cutoff"].iloc[0])
    calendar = xcals.get_calendar("XNYS", start=forecasts.index[0], end=forecasts.index[-1])
    if not forecasts.index.equals(calendar.sessions):
        raise ValueError("Stage 2 forecast origins must include every scheduled XNYS session.")

    calibrated, history = calibrate(forecasts)
    scores = compare(calibrated, cutoff)
    calibrated.to_csv(processed / "stage3_forecasts.csv")
    scores.to_csv(processed / "stage3_scores.csv", index=False)
    history.to_csv(processed / "stage3_multipliers.csv")
    plot_scores(scores, processed)
    print(f"First update: {history.index[0].date()} ({history['matured_outcomes'].iloc[0]} mature outcomes).")
    print(scores.to_string(index=False))


if __name__ == "__main__":
    main()
