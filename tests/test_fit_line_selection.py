import sys
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

fake_error_evaluation = types.ModuleType("error_evaluation")
fake_error_evaluation.U_Calculate = lambda *args, **kwargs: None
fake_error_evaluation.rel_intensity = lambda *args, **kwargs: None
sys.modules.setdefault("error_evaluation", fake_error_evaluation)

import Elements_detectation as ed


class FitLineSelectionTests(unittest.TestCase):
    def test_matplotlib_prefers_chinese_font_candidates(self):
        import matplotlib

        sans_serif_fonts = list(matplotlib.rcParams["font.sans-serif"])

        self.assertEqual(
            sans_serif_fonts[:4],
            ["Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS"],
        )
        self.assertFalse(matplotlib.rcParams["axes.unicode_minus"])

    def test_coarse_payload_under_five_gets_top_valid_non_duplicate_lines_until_five(self):
        coarse_target_rows = pd.DataFrame(
            {
                "TargetWavelength": [200.0, 250.0],
                "SourceElement": ["COARSE_A", "COARSE_B"],
                "PeakIndex": [101, 102],
            },
            index=[0, 1],
        )
        element_df = pd.DataFrame(
            {
                "RelativeIntensity": [999.0, 900.0, 800.0, 700.0, 600.0, 500.0],
                "RawWavelength": [1000.0, 2000.0, 3000.0, 4000.0, np.nan, 5000.0],
                "ThirdColumn": ["ok", "duplicate", "", "ok", "bad_wl", "ok"],
            },
            index=[10, 11, 12, 13, 14, 15],
        )

        (
            wl_tofit,
            fit_source_lookup,
            fit_peak_index_lookup,
            fit_line_source_lookup,
        ) = ed._build_coarse_payload_wl_tofit(
            coarse_target_rows,
            element_df,
            max_total=5,
        )

        self.assertEqual(wl_tofit.astype(float).tolist(), [200.0, 250.0, 100.0, 400.0, 500.0])

        payload_key = wl_tofit.index[0]
        self.assertEqual(fit_source_lookup[payload_key], "COARSE_A")
        self.assertEqual(fit_peak_index_lookup[payload_key], 101)
        self.assertEqual(fit_line_source_lookup[payload_key], "coarse_matched")

        supplement_keys = list(wl_tofit.index[2:])
        self.assertEqual(
            [fit_source_lookup[key] for key in supplement_keys],
            [
                "TOP_RELATIVE_INTENSITY_SUPPLEMENT",
                "TOP_RELATIVE_INTENSITY_SUPPLEMENT",
                "TOP_RELATIVE_INTENSITY_SUPPLEMENT",
            ],
        )
        self.assertEqual([fit_peak_index_lookup[key] for key in supplement_keys], [10, 13, 15])
        self.assertEqual(
            [fit_line_source_lookup[key] for key in supplement_keys],
            [
                "top_relative_intensity_supplement",
                "top_relative_intensity_supplement",
                "top_relative_intensity_supplement",
            ],
        )

    def test_coarse_payload_supplement_keeps_total_at_five(self):
        coarse_target_rows = pd.DataFrame(
            {
                "TargetWavelength": [210.0, 220.0, 230.0, 240.0],
                "SourceElement": ["COARSE"] * 4,
                "PeakIndex": [1, 2, 3, 4],
            }
        )
        element_df = pd.DataFrame(
            {
                "RelativeIntensity": [1000.0, 900.0],
                "RawWavelength": [2600.0, 2700.0],
                "ThirdColumn": ["ok", "ok"],
            },
            index=[20, 21],
        )

        wl_tofit, _, _, _ = ed._build_coarse_payload_wl_tofit(
            coarse_target_rows,
            element_df,
            max_total=5,
        )

        self.assertEqual(wl_tofit.astype(float).tolist(), [210.0, 220.0, 230.0, 240.0, 260.0])

    def test_relative_candidates_allow_enabled_lines_and_any_conflict_marked_lines(self):
        element_df = pd.DataFrame(
            {
                "RelativeIntensity": [1000.0, 900.0, 800.0, 700.0],
                "RawWavelength": [2100.0, 2200.0, 2300.0, 2400.0],
                "ThirdColumn": ["ok", "ok", "ok", "ok"],
                "C3": ["", "", "", ""],
                "C4": ["", "", "", ""],
                "C5": ["", "", "", ""],
                "C6": ["", "", "", ""],
                "C7": ["", "", "", ""],
                "EnableFlag": ["N", "N", "", "Y"],
                "ConflictElem": ["", "Fe", "", ""],
            },
            index=[1, 2, 3, 4],
        )

        candidates = ed._build_relative_intensity_fit_candidates(element_df)

        self.assertEqual(candidates["TargetWavelength"].astype(float).tolist(), [220.0, 230.0, 240.0])

    def test_target_wavelength_window_check_reports_out_of_range_target(self):
        in_window, left, right = ed._target_wavelength_in_segment_window(600.452, [594.52, 594.8, 595.08])

        self.assertFalse(in_window)
        self.assertEqual((left, right), (594.52, 595.08))

    def test_target_wavelength_window_check_accepts_in_range_target(self):
        in_window, left, right = ed._target_wavelength_in_segment_window(594.8, [594.52, 594.8, 595.08])

        self.assertTrue(in_window)
        self.assertEqual((left, right), (594.52, 595.08))


if __name__ == "__main__":
    unittest.main()
