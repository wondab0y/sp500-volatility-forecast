import unittest

import numpy as np
import pandas as pd

from src.build_stage1 import HORIZON
from src.calibrate_stage3 import MIN_MATURED, MULTIPLIERS, UPDATE_SESSIONS, calibrate
from src.compare_stage2 import FORECAST_COLUMNS, TARGET


class CalibrationTimingTest(unittest.TestCase):
    def test_uses_only_mature_outcomes_and_updates_on_scheduled_rows(self):
        dates = pd.bdate_range("2007-01-03", periods=220)
        forecasts = pd.DataFrame(
            {
                TARGET: np.linspace(0.01, 0.05, len(dates)),
                "target_end_date": dates + HORIZON,
                **dict(zip(FORECAST_COLUMNS, (0.02, 0.03, 0.04))),
            },
            index=dates,
        )
        forecasts.loc[dates[45], TARGET] = np.nan
        forecasts["scored"] = forecasts[TARGET].notna()
        calibrated, history = calibrate(forecasts)
        first, second = history.index[:2]
        matured = forecasts["scored"] & (forecasts["target_end_date"] <= first)
        self.assertGreaterEqual(matured.sum(), MIN_MATURED)
        self.assertLess(
            (forecasts["scored"] & (forecasts["target_end_date"] <= dates[dates.get_loc(first) - 1])).sum(),
            MIN_MATURED,
        )
        self.assertEqual(dates.get_loc(second) - dates.get_loc(first), UPDATE_SESSIONS)
        self.assertTrue(calibrated.loc[first:second, "calibration_date"].iloc[:-1].eq(first).all())
        expected = forecasts.loc[matured, FORECAST_COLUMNS].rdiv(forecasts.loc[matured, TARGET], axis=0)
        np.testing.assert_allclose(history.loc[first, MULTIPLIERS].to_numpy(dtype=float), expected.mean())

        changed = forecasts.copy()
        changed.loc[changed["target_end_date"] > second, TARGET] *= 3
        _, changed_history = calibrate(changed)
        pd.testing.assert_frame_equal(history.loc[:second], changed_history.loc[:second])
        self.assertFalse(history.loc[history.index > second].equals(changed_history.loc[history.index > second]))
