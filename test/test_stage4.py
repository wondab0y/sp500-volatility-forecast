import unittest

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from src.build_stage1 import ANNUALIZE
from src.compare_stage2 import REFIT_SESSIONS
from src.compare_stage4 import (
    FILTERED,
    PROBABILITIES,
    STATIONARY,
    build_forecasts,
    filter_history,
    fit_hmm,
    forward_variance,
    observed_sequences,
)


class RegimeForecastTest(unittest.TestCase):
    def test_gap_is_not_fitted_as_an_adjacent_transition(self):
        returns = np.array([0.002, -0.005, np.nan, 0.01, np.nan, 0.015, -0.003])
        observations, lengths = observed_sequences(returns)
        np.testing.assert_array_equal(observations[:, 0], returns[np.isfinite(returns)])
        self.assertEqual(lengths, [2, 1, 2])

        start = np.array([0.1, 0.3, 0.6])
        transition = np.full((3, 3), 0.025)
        np.fill_diagonal(transition, 0.95)
        variances = np.array([0.00002, 0.00012, 0.0006])
        probability = filter_history([np.nan] * 3, start, transition, variances)
        np.testing.assert_allclose(probability, start @ np.linalg.matrix_power(transition, 2))
        expected = sum(start @ np.linalg.matrix_power(transition, h) @ variances for h in range(1, 5))
        np.testing.assert_allclose(
            forward_variance(start[None, :], transition, variances, 4), [ANNUALIZE * expected]
        )

    def test_future_returns_do_not_change_refit_or_between_refit_forecasts(self):
        dates = xcals.get_calendar("XNYS", start="2006-01-01", end="2007-03-30").sessions
        rng = np.random.default_rng(42)
        levels = np.array([0.004, 0.010, 0.024])
        returns = pd.Series(rng.normal(size=len(dates)) * levels[(np.arange(len(dates)) // 35) % 3], index=dates)
        first = dates[dates.searchsorted(pd.Timestamp("2007-01-01"))]
        baseline = build_forecasts(returns, first)
        columns = [FILTERED, STATIONARY, *PROBABILITIES]
        for origin in baseline.index[[0, 10]]:
            with self.subTest(origin=origin):
                changed = returns.copy()
                changed.loc[changed.index > origin] *= -2
                revised = build_forecasts(changed, first)
                pd.testing.assert_series_equal(baseline.loc[origin, columns], revised.loc[origin, columns])
        refits = baseline.drop_duplicates("fit_date")["fit_date"]
        np.testing.assert_array_equal(np.diff(dates.get_indexer(refits)), REFIT_SESSIONS)

        history = returns.loc[:first].to_numpy()
        model = fit_hmm(history, None)
        np.testing.assert_array_equal(model.means_, np.zeros((3, 1)))
        self.assertGreaterEqual(np.diff(model.monitor_.history).min(), -1e-8)
        self.assertTrue((model.covars_[:, 0, 0] <= np.max(history**2)).all())
        np.testing.assert_allclose(
            baseline.loc[first, PROBABILITIES].to_numpy(dtype=float),
            model.predict_proba(history[:, None])[-1],
            rtol=1e-10,
        )
