"""build daily dataset and score raw VIX against 30-day realized variance"""

from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

import exchange_calendars as xcals
import numpy as np
import pandas as pd


DATA = Path(__file__).resolve().parents[1] / "data"
START = pd.Timestamp("2006-01-01")
END = pd.Timestamp("2026-12-31")
SPLICE = pd.Timestamp("2018-01-02")  # first session using fred
HORIZON = pd.Timedelta(days=30)
ANNUALIZE = 365 / HORIZON.days
FRED_URL = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv"
    "?id=SP500&cosd=2016-11-01&coed=2026-12-31"
)
VIX_URL = "https://cdn-api.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"


def read_close(source, date_column, close_column, date_format):
    frame = pd.read_csv(source, usecols=[date_column, close_column], na_values=".")
    dates = pd.to_datetime(frame[date_column], format=date_format)
    close = pd.Series(frame[close_column].to_numpy(dtype=float), index=dates)
    if close.empty or close.index.has_duplicates:
        raise ValueError(f"Empty or duplicate dates in {close_column} snapshot")
    return close.sort_index()


def source_snapshot(path, url, date_column, close_column, date_format):
    if path.exists():
        return read_close(path, date_column, close_column, date_format)
    with urlopen(url, timeout=60) as response:
        contents = response.read()
    close = read_close(BytesIO(contents), date_column, close_column, date_format)
    path.write_bytes(contents)
    return close


def arch_snapshot(path):
    if not path.exists():
        from arch.data import sp500

        sp500.load()["Close"].rename_axis("Date").to_csv(path)
    return read_close(path, "Date", "Close", "%Y-%m-%d")


def join_spx(arch_close, fred_close):
    overlap = pd.concat(
        [arch_close.rename("arch"), fred_close.rename("FRED")], axis=1, join="inner"
    ).dropna()
    if SPLICE not in overlap.index:
        raise ValueError(f"Both SPX sources must have a close on {SPLICE.date()}")
    rounded = overlap.round(2)
    differences = rounded.loc[rounded["arch"] != rounded["FRED"]]
    for date, arch_value, fred_value in differences.itertuples():
        # historical discrepancy is after the splice, so the build uses fred
        if (date, arch_value, fred_value) != (pd.Timestamp("2018-11-29"), 2737.80, 2737.76):
            raise ValueError(f"SPX sources disagree on {date.date()}: {arch_value} vs {fred_value}")
        print(f"SPX overlap exception on {date.date()}: arch {arch_value:.2f}, FRED {fred_value:.2f}")
    print(
        f"SPX overlap {overlap.index.min().date()} through {overlap.index.max().date()}: "
        f"{len(overlap) - len(differences)}/{len(overlap)} closes agree to cents. "
        f"FRED starts in the joined series on {SPLICE.date()}."
    )
    return pd.concat([arch_close.loc[arch_close.index < SPLICE], fred_close.loc[SPLICE:]])


def realized_target(returns, cutoff):
    """Use daily log returns on sorted scheduled sessions through the cutoff."""
    sessions = returns.index
    ends = sessions + HORIZON
    stops = sessions.searchsorted(ends, side="right")
    squared_returns = returns.to_numpy() ** 2
    variance = np.full(len(sessions), np.nan)
    for origin, stop in enumerate(stops):
        if ends[origin] <= cutoff:
            # a missing return propagates through the sum for the entire window
            variance[origin] = ANNUALIZE * squared_returns[origin + 1 : stop].sum()
    return pd.DataFrame(
        {"realized_target_variance": variance, "target_end_date": ends}, index=sessions
    )


def qlike_loss(forecast, realized):
    return np.log(forecast) + realized / forecast


def score(dataset, cutoff, forecast_column="raw_vix_forecast_variance"):
    forecast = dataset[forecast_column]
    realized = dataset["realized_target_variance"]
    scored = np.isfinite(realized) & np.isfinite(forecast) & (forecast > 0)
    forecast, realized = forecast[scored], realized[scored]
    if forecast.empty:
        raise ValueError("No forecasts with complete targets and positive finite variance")
    return pd.DataFrame([{
        "snapshot_cutoff": cutoff.date(),
        "scored_origins": len(forecast),
        "first_scored_date": forecast.index.min().date(),
        "last_scored_date": forecast.index.max().date(),
        "mae_volatility_pp": (100 * (np.sqrt(forecast) - np.sqrt(realized)).abs()).mean(),
        "qlike": qlike_loss(forecast, realized).mean(),
    }])


def main():
    raw = DATA / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    arch_close = arch_snapshot(raw / "arch_sp500.csv")
    fred_close = source_snapshot(
        raw / "fred_sp500.csv", FRED_URL, "observation_date", "SP500", "%Y-%m-%d"
    )
    vix_close = source_snapshot(
        raw / "cboe_vix.csv", VIX_URL, "DATE", "CLOSE", "%m/%d/%Y"
    ).loc[:END]
    spx_close = join_spx(arch_close, fred_close.loc[:END])

    # keep sessions through the latest close in either snapshot
    cutoff = max(spx_close.last_valid_index(), vix_close.last_valid_index())
    if cutoff.year != END.year:
        raise ValueError(f"Expected a 2026 snapshot cutoff, found {cutoff.date()}")
    calendar = xcals.get_calendar("XNYS", start="2005-12-01", end=END)
    sessions = calendar.sessions_in_range(START, cutoff)
    previous = calendar.previous_session(sessions[0])
    if pd.isna(spx_close.get(previous)):
        raise ValueError(f"Missing the last 2005 SPX close on {previous.date()}")
    spx_on_sessions = spx_close.reindex(sessions.insert(0, previous))
    dataset = pd.DataFrame(index=sessions.rename("date"))
    dataset["spx_close"] = spx_on_sessions
    dataset["vix_close"] = vix_close
    dataset["daily_log_return"] = np.log(spx_on_sessions).diff()
    dataset["raw_vix_forecast_variance"] = (dataset["vix_close"] / 100) ** 2
    dataset = dataset.join(realized_target(dataset["daily_log_return"], cutoff))
    metrics = score(dataset, cutoff)

    processed = DATA / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(processed / "stage1_daily.csv")
    metrics.to_csv(processed / "stage1_metrics.csv", index=False)
    print(
        f"{len(dataset)} XNYS sessions from {sessions[0].date()} to {sessions[-1].date()}. "
        f"Missing SPX closes: {dataset['spx_close'].isna().sum()}. "
        f"Missing VIX closes: {dataset['vix_close'].isna().sum()}."
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
