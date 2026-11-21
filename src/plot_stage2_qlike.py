"""plot QLIKE comparisons from the saved stage 2 csvs"""

import matplotlib
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from .build_stage1 import DATA, qlike_loss
from .compare_stage2 import FORECAST_COLUMNS, MODELS, TARGET, period_samples, scoring_mask


PERIOD_LABELS = {
    "2007–09": "Global financial crisis",
    "2010–19": "Pre-COVID",
    "2020–21": "COVID era",
    "2022–end": "Later period",
}
COLORS = ("#5943c2", "#087e8b")
SUBTITLE = "Below zero = beating VIX since the start of the panel. A falling line = gaining on VIX over that stretch."


def period_curves(forecasts, metrics):
    forecasts = forecasts.sort_index()
    if not forecasts.index.is_unique or not scoring_mask(forecasts).equals(forecasts["scored"]):
        raise ValueError("Stage 2 forecast dates or common scoring mask do not match the saved data.")
    shared = forecasts.loc[forecasts["scored"]]
    model_keys = list(MODELS)
    vix = model_keys[-1]
    panels = []
    for period, sample in period_samples(shared):
        if sample.empty:
            raise ValueError(f"{period} has no scored forecasts.")
        start, end = sample.index[[0, -1]]
        losses = pd.DataFrame({
            model: qlike_loss(sample[column], sample[TARGET])
            for model, column in zip(model_keys, FORECAST_COLUMNS)
        })
        reference = metrics.loc[period].reindex(model_keys)
        if not (
            reference["scored_origins"].eq(len(sample)).all()
            and reference["first_scored_date"].eq(start).all()
            and reference["last_scored_date"].eq(end).all()
        ):
            raise ValueError(f"{period} scored dates do not match the Stage 2 period metrics.")
        np.testing.assert_allclose(
            losses.mean(), reference["qlike"], rtol=0, atol=1e-12,
            err_msg=f"{period} QLIKE differs from the Stage 2 period metrics.",
        )
        differences = losses[model_keys[:-1]].sub(losses[vix], axis=0)
        mean = differences.mean()
        np.testing.assert_allclose(
            mean, reference.loc[mean.index, "qlike"] - reference.at[vix, "qlike"],
            rtol=0, atol=1e-12, err_msg=f"{period} paired mean differences disagree with Stage 2.",
        )
        zero = pd.DataFrame(0.0, index=[start - pd.Timedelta(days=1)], columns=differences.columns)
        cumulative = pd.concat([zero, differences.cumsum()])
        np.testing.assert_allclose(cumulative.iloc[-1], differences.sum(), rtol=0, atol=1e-12)
        expected_sign = -1 if period in ("2010–19", "2022–end") else 1
        if np.sign(cumulative.iloc[-1]["raw_garch"]) != expected_sign:
            raise ValueError(f"{period} GARCH end value has an unexpected sign.")
        panels.append((PERIOD_LABELS[period], start, end, mean, cumulative))
    if sum(len(curve) - 1 for _, _, _, _, curve in panels) != len(shared):
        raise ValueError("The panels do not cover every scored origin.")
    return panels


def plot_periods(panels, path):
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.8))
    fig.suptitle("When did raw GARCH and HAR beat raw VIX?", y=0.975, fontsize=19, fontweight="bold")
    fig.text(0.5, 0.922, SUBTITLE, ha="center", fontsize=11, color="#414855")
    fig.subplots_adjust(left=0.092, right=0.979, bottom=0.095, top=0.79, hspace=0.60, wspace=0.18)
    for ax, (label, start, end, mean, cumulative) in zip(axes.flat, panels):
        for model, color in zip(cumulative.columns, COLORS):
            ax.plot(cumulative.index, cumulative[model], color=color, linewidth=1.65, label=MODELS[model])
        ax.axhline(0, color="#515765", linewidth=1.1, label="Raw VIX reference")
        ax.set_title(label, loc="left", pad=38, fontsize=13, fontweight="bold")
        ax.text(
            0, 1.035,
            f"{start.date()} to {end.date()}  ·  {len(cumulative) - 1:,} forecasts\n"
            f"Mean ΔQLIKE   GARCH {mean['raw_garch']:+.4f}   ·   HAR {mean['raw_har']:+.4f}",
            transform=ax.transAxes, fontsize=10, color="#414855", linespacing=1.45,
        )
        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.margins(x=0.02)
        ax.grid(color="#e1e5e9", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.902), ncols=3, frameon=False)
    fig.supylabel("Cumulative QLIKE difference vs. raw VIX", x=0.022)
    fig.supxlabel("Forecast origin", y=0.026)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    processed = DATA / "processed"
    forecasts = pd.read_csv(
        processed / "stage2_forecasts.csv", index_col="date", parse_dates=["date"]
    )
    metrics = pd.read_csv(
        processed / "stage2_metrics.csv", index_col=["period", "model"],
        parse_dates=["first_scored_date", "last_scored_date"],
    )
    panels = period_curves(forecasts, metrics)
    path = processed / "stage2_cumulative_qlike_by_period.png"
    plot_periods(panels, path)
    for label, start, end, mean, cumulative in panels:
        print(f"{label} | {start.date()} to {end.date()} | {len(cumulative) - 1:,} forecasts")
        for model in cumulative.columns:
            print(f"  {MODELS[model]}  mean {mean[model]:+.6f}  total {cumulative[model].iloc[-1]:+.6f}")
    print(f"Verified against the saved Stage 2 metrics. Wrote {path}")


if __name__ == "__main__":
    main()
