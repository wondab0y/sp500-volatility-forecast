import unittest

import pandas as pd

from src.build_stage1 import realized_target


class TargetWindowTest(unittest.TestCase):
    def test_excludes_origin_and_includes_day_30(self):
        dates = pd.to_datetime(["2024-01-03", "2024-01-04", "2024-02-02", "2024-02-05"])
        returns = pd.Series([0.5, 0.01, -0.02, 0.5], index=dates)
        target = realized_target(returns, dates[-1])
        self.assertEqual(target.loc[dates[0], "target_end_date"], dates[2])
        self.assertAlmostEqual(
            target.loc[dates[0], "realized_target_variance"],
            (365 / 30) * (0.01**2 + 0.02**2),
        )

        returns.loc[dates[2]] = float("nan")
        target = realized_target(returns, dates[-1])
        self.assertTrue(pd.isna(target.loc[dates[0], "realized_target_variance"]))

    def test_requires_all_30_calendar_days_even_over_a_weekend(self):
        sessions = pd.to_datetime(["2024-01-04", "2024-02-02"])
        returns = pd.Series([0.5, 0.01], index=sessions)
        cutoff = pd.Timestamp("2024-02-02")
        target = realized_target(returns, cutoff)
        self.assertTrue(pd.isna(target.iloc[0]["realized_target_variance"]))
        target = realized_target(returns, cutoff + pd.Timedelta(days=1))
        self.assertAlmostEqual(target.iloc[0]["realized_target_variance"], (365 / 30) * 0.01**2)
