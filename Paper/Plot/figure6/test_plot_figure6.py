from __future__ import absolute_import

import hashlib
import os
import sys
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.colors import to_rgba
import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import plot_figure6 as figure6
from plot_figure6 import DEFAULTS, plot_spectrum


class PlotFigure6Tests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def _write_csv(self, directory, contents, name="spectrum.csv"):
        path = os.path.join(directory, name)
        with open(path, "w") as csv_file:
            csv_file.write(contents)
        return path

    def _plot_from(self, data_path, **overrides):
        config = {
            "data_path": data_path,
            "output_path": None,
            "show": False,
            "mark_peaks": False,
            "mark_green_peaks": False,
            "peak_comb_visible": False,
            "green_peak_comb_visible": False,
            "peak_text_visible": False,
            "green_peak_text_visible": False,
            "peak_adjustment_enabled": False,
            "y_limits": None,
        }
        config.update(overrides)
        return plot_spectrum(config)

    def _path_collections(self, ax):
        return [item for item in ax.collections if isinstance(item, PathCollection)]

    def _line_collections(self, ax):
        return [item for item in ax.collections if isinstance(item, LineCollection)]

    def test_uses_only_first_two_columns_and_plots_them_unchanged(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = self._write_csv(
                temp_dir,
                "wavelength,intensity,ignored\n"
                "400.0,0.25,not-a-number\n"
                "500.5,3.75,completely irrelevant\n",
            )

            fig, ax = self._plot_from(data_path)

        line = ax.lines[0]
        np.testing.assert_allclose(line.get_xdata(), [400.0, 500.5])
        np.testing.assert_allclose(line.get_ydata(), [0.25, 3.75])
        self.assertEqual(tuple(fig.get_size_inches()), (9.0, 5.0))

    def test_applies_requested_plot_style(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = self._write_csv(
                temp_dir, "wavelength,intensity\n400,1\n500,2\n"
            )

            fig, ax = self._plot_from(data_path)

        line = ax.lines[0]
        self.assertEqual(line.get_color(), DEFAULTS["line_color"])
        self.assertAlmostEqual(line.get_linewidth(), 2.2)
        self.assertEqual(ax.get_yscale(), "linear")
        self.assertIsNone(ax.get_legend())
        for spine in ax.spines.values():
            self.assertAlmostEqual(spine.get_linewidth(), 1.8)
        self.assertEqual(ax.xaxis.label.get_fontsize(), 15)
        self.assertEqual(ax.yaxis.label.get_fontsize(), 15)
        self.assertEqual(ax.xaxis.label.get_fontweight(), "semibold")
        self.assertEqual(ax.yaxis.label.get_fontweight(), "semibold")
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            self.assertEqual(label.get_fontsize(), 12)
            self.assertEqual(label.get_fontweight(), "semibold")
        major_tick = ax.xaxis.get_major_ticks()[0]
        self.assertAlmostEqual(major_tick.tick1line.get_markersize(), 6)
        self.assertAlmostEqual(major_tick.tick1line.get_markeredgewidth(), 2)
        self.assertTrue(major_tick.tick2line.get_visible())
        visible_gridlines = [line for line in ax.get_xgridlines() if line.get_visible()]
        self.assertTrue(visible_gridlines)
        self.assertAlmostEqual(visible_gridlines[0].get_alpha(), 0.3)

    def test_default_major_intervals_disable_all_minor_ticks(self):
        self.assertFalse(DEFAULTS["minor_ticks"])
        self.assertEqual(DEFAULTS["x_major_tick_interval"], 100)
        self.assertEqual(DEFAULTS["y_major_tick_interval"], 50000)
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = self._write_csv(
                temp_dir,
                "wavelength,intensity\n200,0\n800,275000\n",
            )

            _, ax = self._plot_from(data_path)

        np.testing.assert_allclose(np.diff(ax.get_xticks()), 100)
        np.testing.assert_allclose(np.diff(ax.get_yticks()), 50000)
        self.assertFalse(
            any(line.get_visible() for line in ax.xaxis.get_minorticklines())
        )
        self.assertFalse(
            any(line.get_visible() for line in ax.yaxis.get_minorticklines())
        )
        x_major_tick = ax.xaxis.get_major_ticks()[0]
        y_major_tick = ax.yaxis.get_major_ticks()[0]
        self.assertTrue(x_major_tick.tick1line.get_visible())
        self.assertTrue(x_major_tick.tick2line.get_visible())
        self.assertTrue(y_major_tick.tick1line.get_visible())
        self.assertTrue(y_major_tick.tick2line.get_visible())

    def test_filters_nan_and_infinity_as_complete_xy_pairs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = self._write_csv(
                temp_dir,
                "wavelength,intensity,ignored\n"
                "400,1,x\n"
                "bad,2,y\n"
                "500,NaN,z\n"
                "inf,4,q\n"
                "600,-inf,r\n"
                "700,7,s\n",
            )

            _, ax = self._plot_from(data_path)

        np.testing.assert_allclose(ax.lines[0].get_xdata(), [400.0, 700.0])
        np.testing.assert_allclose(ax.lines[0].get_ydata(), [1.0, 7.0])

    def test_all_invalid_rows_raise_clear_value_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = self._write_csv(
                temp_dir,
                "wavelength,intensity\nbad,NaN\ninf,-inf\n",
            )

            with self.assertRaisesRegex(ValueError, "no valid wavelength/intensity"):
                self._plot_from(data_path)

    def test_csv_with_fewer_than_two_columns_reports_its_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = self._write_csv(temp_dir, "wavelength\n400\n")

            with self.assertRaises(ValueError) as error:
                self._plot_from(data_path)

        self.assertIn(data_path, str(error.exception))
        self.assertIn("at least two columns", str(error.exception))

    def test_config_override_does_not_mutate_defaults(self):
        defaults_before = DEFAULTS.copy()
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = self._write_csv(
                temp_dir, "wavelength,intensity\n400,1\n500,2\n"
            )

            fig, ax = self._plot_from(
                data_path,
                figsize=(4, 3),
                line_color="black",
                x_major_tick_interval=25,
                y_major_tick_interval=0.5,
            )

        self.assertEqual(tuple(fig.get_size_inches()), (4.0, 3.0))
        np.testing.assert_allclose(np.diff(ax.get_xticks()), 25)
        np.testing.assert_allclose(np.diff(ax.get_yticks()), 0.5)
        self.assertEqual(DEFAULTS, defaults_before)
        self.assertEqual(DEFAULTS["figsize"], (9, 5))
        self.assertEqual(DEFAULTS["x_major_tick_interval"], 100)
        self.assertEqual(DEFAULTS["y_major_tick_interval"], 50000)

    def test_default_y_limits_keep_250000_as_highest_visible_major_tick(self):
        self.assertEqual(DEFAULTS.get("y_limits"), (0, 275000))
        data_path = os.path.join(HERE, "data.csv")

        _, ax = plot_spectrum(
            {
                "data_path": data_path,
                "output_path": None,
                "show": False,
            }
        )

        np.testing.assert_allclose(ax.get_ylim(), [0.0, 275000.0])
        lower, upper = ax.get_ylim()
        visible_ticks = [
            tick for tick in ax.get_yticks() if lower <= tick <= upper
        ]
        np.testing.assert_allclose(
            visible_ticks,
            [0, 50000, 100000, 150000, 200000, 250000],
        )
        self.assertNotIn(300000, visible_ticks)

    def test_y_limits_none_keeps_matplotlib_autoscaling(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = self._write_csv(
                temp_dir, "wavelength,intensity\n0,10\n1,20\n"
            )

            _, ax = self._plot_from(data_path, y_limits=None)

        lower, upper = ax.get_ylim()
        self.assertLess(lower, 10)
        self.assertGreater(upper, 20)

    def test_y_limits_accept_full_and_single_ended_overrides(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = self._write_csv(
                temp_dir, "wavelength,intensity\n0,10\n1,20\n"
            )

            _, full_ax = self._plot_from(data_path, y_limits=(5, 25))
            _, upper_ax = self._plot_from(data_path, y_limits=(None, 30))
            _, lower_ax = self._plot_from(data_path, y_limits=(0, None))

        np.testing.assert_allclose(full_ax.get_ylim(), [5.0, 25.0])
        self.assertEqual(upper_ax.get_ylim()[1], 30.0)
        self.assertEqual(lower_ax.get_ylim()[0], 0.0)

    def test_invalid_y_limits_raise_clear_value_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = self._write_csv(
                temp_dir, "wavelength,intensity\n0,10\n1,20\n"
            )
            invalid_limits = (
                (),
                (0,),
                (0, 1, 2),
                10,
                [0, 10],
                (np.nan, 10),
                (0, np.inf),
                (-np.inf, 10),
                (10, 10),
                (20, 10),
                ("bad", 10),
            )

            for limits in invalid_limits:
                with self.subTest(y_limits=limits):
                    with self.assertRaisesRegex(
                        ValueError,
                        "y_limits must be None or contain two finite-or-None "
                        "values with lower less than upper",
                    ):
                        self._plot_from(data_path, y_limits=limits)

    def test_y_limits_override_does_not_mutate_defaults(self):
        defaults_before = DEFAULTS.copy()
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = self._write_csv(
                temp_dir, "wavelength,intensity\n0,10\n1,20\n"
            )

            _, ax = self._plot_from(data_path, y_limits=(5, 25))

        np.testing.assert_allclose(ax.get_ylim(), [5.0, 25.0])
        self.assertEqual(DEFAULTS, defaults_before)

    def test_show_false_writes_a_nonempty_png(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = self._write_csv(
                temp_dir, "wavelength,intensity\n400,1\n500,2\n"
            )
            output_path = os.path.join(temp_dir, "actual.png")

            self._plot_from(data_path, output_path=output_path)

            self.assertTrue(os.path.isfile(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)

    def test_peak_marker_defaults_are_centralized(self):
        expected = {
            "mark_peaks": True,
            "peak_target_wavelengths": (
                382.01,
                390.7,
                393.34,
                396.86,
                413.02,
                420.59,
                443.39,
                453.4,
            ),
            "peak_search_half_width": 0.25,
            "peak_marker": "o",
            "peak_marker_color": DEFAULTS["peak_marker_color"],
            "peak_marker_size": DEFAULTS["peak_marker_size"],
            "peak_marker_zorder": 5,
            "peak_marker_edgecolor": "none",
        }
        for key, value in expected.items():
            self.assertEqual(DEFAULTS.get(key), value)

    def test_peak_search_uses_window_maximum_not_nearest_sample(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = self._write_csv(
                temp_dir,
                "wavelength,intensity\n9.80,7\n10.01,3\n10.30,20\n",
            )

            _, ax = self._plot_from(
                data_path,
                mark_peaks=True,
                peak_target_wavelengths=(10.0,),
                peak_search_half_width=0.25,
            )

        markers = self._path_collections(ax)
        self.assertEqual(len(markers), 1)
        np.testing.assert_allclose(markers[0].get_offsets(), [[9.8, 7.0]])

    def test_two_peaks_use_one_solid_red_scatter_with_requested_style(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = self._write_csv(
                temp_dir,
                "wavelength,intensity\n"
                "381.90,5\n382.01,20\n382.20,10\n"
                "390.50,3\n390.61,30\n390.70,4\n",
            )

            _, ax = self._plot_from(
                data_path,
                mark_peaks=True,
                peak_target_wavelengths=(382.01, 390.7),
                peak_search_half_width=0.25,
            )

        markers = self._path_collections(ax)
        self.assertEqual(len(markers), 1)
        marker = markers[0]
        np.testing.assert_allclose(
            marker.get_offsets(), [[382.01, 20.0], [390.61, 30.0]]
        )
        np.testing.assert_allclose(
            marker.get_facecolors()[0], to_rgba(DEFAULTS["peak_marker_color"])
        )
        np.testing.assert_allclose(marker.get_sizes(), [DEFAULTS["peak_marker_size"]])
        self.assertEqual(marker.get_zorder(), 5)
        self.assertEqual(marker.get_label(), "_nolegend_")
        self.assertEqual(marker.get_edgecolors().size, 0)

    def test_real_data_selects_the_eight_expected_peak_samples(self):
        data_path = os.path.join(HERE, "data.csv")

        _, ax = self._plot_from(data_path, mark_peaks=True)

        markers = self._path_collections(ax)
        self.assertEqual(len(markers), 1)
        np.testing.assert_allclose(
            markers[0].get_offsets(),
            [
                [382.01, 29440.0],
                [390.61, 6826.0],
                [393.34, 73130.0],
                [396.86, 40980.0],
                [412.97, 7911.0],
                [420.49, 8166.0],
                [443.53, 3484.0],
                [453.4, 3656.0],
            ],
            rtol=0,
            atol=0,
        )

    def test_empty_peak_search_window_raises_clear_value_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = self._write_csv(
                temp_dir, "wavelength,intensity\n1,10\n2,20\n"
            )

            with self.assertRaisesRegex(ValueError, "No samples in peak search window"):
                self._plot_from(
                    data_path,
                    mark_peaks=True,
                    peak_target_wavelengths=(5.0,),
                    peak_search_half_width=0.25,
                )

    def test_invalid_peak_half_width_raises_clear_value_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = self._write_csv(
                temp_dir, "wavelength,intensity\n1,10\n2,20\n"
            )

            for half_width in (0, -1, np.nan, np.inf, -np.inf):
                with self.subTest(half_width=half_width):
                    with self.assertRaisesRegex(
                        ValueError, "peak_search_half_width must be finite and positive"
                    ):
                        self._plot_from(
                            data_path,
                            mark_peaks=True,
                            peak_target_wavelengths=(1.0,),
                            peak_search_half_width=half_width,
                        )

    def test_nonfinite_peak_target_raises_clear_value_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = self._write_csv(
                temp_dir, "wavelength,intensity\n1,10\n2,20\n"
            )

            for target in (np.nan, np.inf, -np.inf):
                with self.subTest(target=target):
                    with self.assertRaisesRegex(
                        ValueError, "peak target wavelength must be finite"
                    ):
                        self._plot_from(
                            data_path,
                            mark_peaks=True,
                            peak_target_wavelengths=(target,),
                            peak_search_half_width=0.25,
                        )

    def test_tied_peak_is_nearest_then_lower_wavelength(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = self._write_csv(
                temp_dir,
                "wavelength,intensity\n-0.20,10\n-0.10,10\n0.10,10\n0.20,10\n",
            )

            _, ax = self._plot_from(
                data_path,
                mark_peaks=True,
                peak_target_wavelengths=(0.0,),
                peak_search_half_width=0.25,
            )

        markers = self._path_collections(ax)
        self.assertEqual(len(markers), 1)
        np.testing.assert_allclose(markers[0].get_offsets(), [[-0.1, 10.0]])

    def test_mark_peaks_false_adds_no_collection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = self._write_csv(
                temp_dir, "wavelength,intensity\n1,10\n2,20\n"
            )

            _, ax = self._plot_from(data_path, mark_peaks=False)

        self.assertEqual(len(self._path_collections(ax)), 0)

    def test_peak_overrides_do_not_mutate_defaults(self):
        defaults_before = DEFAULTS.copy()
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = self._write_csv(
                temp_dir, "wavelength,intensity\n1,10\n2,20\n"
            )

            _, ax = self._plot_from(
                data_path,
                mark_peaks=True,
                peak_target_wavelengths=(1.0,),
                peak_search_half_width=0.1,
                peak_marker_color="black",
                peak_marker_size=81,
            )

        np.testing.assert_allclose(self._path_collections(ax)[0].get_sizes(), [81])
        self.assertEqual(DEFAULTS, defaults_before)

    def test_green_peak_defaults_are_centralized(self):
        expected = {
            "mark_green_peaks": True,
            "green_peak_target_wavelengths": (
                211.71,
                217.42,
                289.1,
                328.99,
                369.42,
            ),
            "green_peak_search_half_width": 0.25,
            "green_peak_marker": "o",
            "green_peak_marker_color": "#1A9850",
            "green_peak_marker_size": DEFAULTS["green_peak_marker_size"],
            "green_peak_marker_zorder": 5,
            "green_peak_marker_edgecolor": "none",
            "green_peak_marker_label": "_nolegend_",
        }
        for key, value in expected.items():
            self.assertEqual(DEFAULTS.get(key), value)

    def test_red_and_green_peaks_use_separate_styled_collections(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = self._write_csv(
                temp_dir,
                "wavelength,intensity\n9.9,2\n10.0,20\n19.9,3\n20.0,30\n",
            )

            _, ax = self._plot_from(
                data_path,
                mark_peaks=True,
                peak_target_wavelengths=(10.0,),
                peak_search_half_width=0.25,
                mark_green_peaks=True,
                green_peak_target_wavelengths=(20.0,),
                green_peak_search_half_width=0.25,
            )

        markers = self._path_collections(ax)
        self.assertEqual(len(markers), 2)
        red_markers, green_markers = markers
        np.testing.assert_allclose(red_markers.get_offsets(), [[10.0, 20.0]])
        np.testing.assert_allclose(green_markers.get_offsets(), [[20.0, 30.0]])
        np.testing.assert_allclose(
            red_markers.get_facecolors()[0], to_rgba(DEFAULTS["peak_marker_color"])
        )
        np.testing.assert_allclose(
            green_markers.get_facecolors()[0],
            to_rgba(DEFAULTS["green_peak_marker_color"]),
        )
        np.testing.assert_allclose(
            red_markers.get_sizes(), [DEFAULTS["peak_marker_size"]]
        )
        np.testing.assert_allclose(
            green_markers.get_sizes(), [DEFAULTS["green_peak_marker_size"]]
        )
        self.assertEqual(red_markers.get_label(), "_nolegend_")
        self.assertEqual(
            green_markers.get_label(), DEFAULTS["green_peak_marker_label"]
        )

    def test_real_data_selects_expected_red_and_green_peak_samples(self):
        data_path = os.path.join(HERE, "data.csv")

        _, ax = self._plot_from(
            data_path,
            mark_peaks=True,
            mark_green_peaks=True,
        )

        markers = self._path_collections(ax)
        self.assertEqual(len(markers), 2)
        np.testing.assert_allclose(
            markers[0].get_offsets(),
            [
                [382.01, 29440.0],
                [390.61, 6826.0],
                [393.34, 73130.0],
                [396.86, 40980.0],
                [412.97, 7911.0],
                [420.49, 8166.0],
                [443.53, 3484.0],
                [453.4, 3656.0],
            ],
            rtol=0,
            atol=0,
        )
        np.testing.assert_allclose(
            markers[1].get_offsets(),
            [
                [211.71, 1963.0],
                [217.42, 2061.0],
                [289.15, 2506.0],
                [328.89, 17840.0],
                [369.42, 8294.0],
            ],
            rtol=0,
            atol=0,
        )

    def test_red_and_green_marker_sizes_override_independently(self):
        defaults_before = DEFAULTS.copy()
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = self._write_csv(
                temp_dir, "wavelength,intensity\n10,20\n20,30\n"
            )

            _, ax = self._plot_from(
                data_path,
                mark_peaks=True,
                peak_target_wavelengths=(10.0,),
                peak_marker_size=81,
                mark_green_peaks=True,
                green_peak_target_wavelengths=(20.0,),
                green_peak_marker_size=121,
            )

        markers = self._path_collections(ax)
        self.assertEqual(len(markers), 2)
        np.testing.assert_allclose(markers[0].get_sizes(), [81])
        np.testing.assert_allclose(markers[1].get_sizes(), [121])
        self.assertEqual(DEFAULTS, defaults_before)

    def test_green_peaks_false_short_circuits_invalid_green_settings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = self._write_csv(
                temp_dir, "wavelength,intensity\n10,20\n"
            )

            _, ax = self._plot_from(
                data_path,
                mark_peaks=True,
                peak_target_wavelengths=(10.0,),
                mark_green_peaks=False,
                green_peak_target_wavelengths=(np.inf,),
                green_peak_search_half_width=0,
            )

        self.assertEqual(len(self._path_collections(ax)), 1)

    def test_empty_green_target_tuple_adds_no_collection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = self._write_csv(
                temp_dir, "wavelength,intensity\n10,20\n"
            )

            _, ax = self._plot_from(
                data_path,
                mark_peaks=False,
                mark_green_peaks=True,
                green_peak_target_wavelengths=(),
            )

        self.assertEqual(len(self._path_collections(ax)), 0)

    def test_green_peak_errors_reuse_peak_search_validation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = self._write_csv(
                temp_dir, "wavelength,intensity\n1,10\n2,20\n"
            )

            invalid_cases = (
                ((5.0,), 0.25, "No samples in peak search window"),
                ((1.0,), 0, "peak_search_half_width must be finite and positive"),
                ((np.inf,), 0.25, "peak target wavelength must be finite"),
            )
            for targets, half_width, message in invalid_cases:
                with self.subTest(targets=targets, half_width=half_width):
                    with self.assertRaisesRegex(ValueError, message):
                        self._plot_from(
                            data_path,
                            mark_peaks=False,
                            mark_green_peaks=True,
                            green_peak_target_wavelengths=targets,
                            green_peak_search_half_width=half_width,
                        )

    def test_peak_comb_defaults_are_centralized(self):
        expected = {
            "peak_comb_visible": True,
            "green_peak_comb_visible": True,
            "peak_comb_height": 250000,
            "peak_comb_stem_bottom": 200000,
            "peak_comb_line_width": 1.2,
            "peak_comb_line_style": "-",
            "peak_comb_zorder": 4,
            "peak_comb_label": "_nolegend_",
            "green_peak_comb_line_width": 1.2,
            "green_peak_comb_line_style": "-",
            "green_peak_comb_zorder": 4,
            "green_peak_comb_label": "_nolegend_",
        }
        for key, value in expected.items():
            self.assertEqual(DEFAULTS.get(key), value)

    def test_red_and_green_combs_have_exact_segments_and_styles(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = self._write_csv(
                temp_dir,
                "wavelength,intensity\n10,20\n12,30\n20,40\n23,50\n",
            )

            _, ax = self._plot_from(
                data_path,
                mark_peaks=True,
                peak_target_wavelengths=(10.0, 12.0),
                mark_green_peaks=True,
                green_peak_target_wavelengths=(20.0, 23.0),
                peak_comb_visible=True,
                green_peak_comb_visible=True,
                peak_comb_height=100,
                peak_comb_stem_bottom=60,
            )

        markers = self._path_collections(ax)
        combs = self._line_collections(ax)
        self.assertEqual(len(markers), 2)
        self.assertEqual(len(combs), 2)
        np.testing.assert_allclose(
            combs[0].get_segments(),
            [
                [[10.0, 100.0], [10.0, 60.0]],
                [[12.0, 100.0], [12.0, 60.0]],
                [[10.0, 100.0], [12.0, 100.0]],
            ],
        )
        np.testing.assert_allclose(
            combs[1].get_segments(),
            [
                [[20.0, 100.0], [20.0, 60.0]],
                [[23.0, 100.0], [23.0, 60.0]],
                [[20.0, 100.0], [23.0, 100.0]],
            ],
        )
        np.testing.assert_allclose(
            combs[0].get_colors()[0], to_rgba(DEFAULTS["peak_marker_color"])
        )
        np.testing.assert_allclose(
            combs[1].get_colors()[0],
            to_rgba(DEFAULTS["green_peak_marker_color"]),
        )
        np.testing.assert_allclose(
            combs[0].get_linewidths(), [DEFAULTS["peak_comb_line_width"]]
        )
        np.testing.assert_allclose(
            combs[1].get_linewidths(),
            [DEFAULTS["green_peak_comb_line_width"]],
        )
        self.assertEqual(combs[0].get_linestyles(), [(None, None)])
        self.assertEqual(combs[1].get_linestyles(), [(None, None)])
        self.assertEqual(combs[0].get_zorder(), DEFAULTS["peak_comb_zorder"])
        self.assertEqual(
            combs[1].get_zorder(), DEFAULTS["green_peak_comb_zorder"]
        )
        self.assertEqual(combs[0].get_label(), DEFAULTS["peak_comb_label"])
        self.assertEqual(
            combs[1].get_label(), DEFAULTS["green_peak_comb_label"]
        )
        self.assertIsNone(ax.get_legend())
        self.assertEqual(len(ax.lines), 1)

    def test_real_combs_have_one_vertical_per_peak_and_one_horizontal(self):
        data_path = os.path.join(HERE, "data.csv")

        _, ax = self._plot_from(
            data_path,
            mark_peaks=True,
            mark_green_peaks=True,
            peak_comb_visible=True,
            green_peak_comb_visible=True,
        )

        markers = self._path_collections(ax)
        combs = self._line_collections(ax)
        self.assertEqual(len(markers), 2)
        self.assertEqual(len(combs), 2)
        self.assertEqual(len(combs[0].get_segments()), 9)
        self.assertEqual(len(combs[1].get_segments()), 6)
        red_offsets = markers[0].get_offsets()
        green_offsets = markers[1].get_offsets()
        red_verticals = np.asarray(combs[0].get_segments()[:-1])
        green_verticals = np.asarray(combs[1].get_segments()[:-1])
        np.testing.assert_allclose(
            red_verticals[:, :, 0], np.repeat(red_offsets[:, 0, None], 2, axis=1)
        )
        np.testing.assert_allclose(
            green_verticals[:, :, 0],
            np.repeat(green_offsets[:, 0, None], 2, axis=1),
        )
        np.testing.assert_allclose(red_verticals[:, 0, 1], 250000)
        np.testing.assert_allclose(red_verticals[:, 1, 1], 200000)
        np.testing.assert_allclose(green_verticals[:, 0, 1], 250000)
        np.testing.assert_allclose(green_verticals[:, 1, 1], 200000)
        np.testing.assert_allclose(
            combs[0].get_segments()[-1],
            [[382.01, 250000.0], [453.4, 250000.0]],
        )
        np.testing.assert_allclose(
            combs[1].get_segments()[-1],
            [[211.71, 250000.0], [369.42, 250000.0]],
        )
        self.assertEqual(len(ax.lines), 1)

    def test_comb_switches_are_independent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = self._write_csv(
                temp_dir, "wavelength,intensity\n10,20\n20,30\n"
            )

            _, red_ax = self._plot_from(
                data_path,
                mark_peaks=True,
                peak_target_wavelengths=(10.0,),
                mark_green_peaks=True,
                green_peak_target_wavelengths=(20.0,),
                peak_comb_visible=True,
                green_peak_comb_visible=False,
                peak_comb_height=100,
                peak_comb_stem_bottom=60,
            )
            _, green_ax = self._plot_from(
                data_path,
                mark_peaks=True,
                peak_target_wavelengths=(10.0,),
                mark_green_peaks=True,
                green_peak_target_wavelengths=(20.0,),
                peak_comb_visible=False,
                green_peak_comb_visible=True,
                peak_comb_height=100,
                peak_comb_stem_bottom=60,
            )

        red_only = self._line_collections(red_ax)
        green_only = self._line_collections(green_ax)
        self.assertEqual(len(red_only), 1)
        self.assertEqual(len(green_only), 1)
        np.testing.assert_allclose(
            red_only[0].get_colors()[0], to_rgba(DEFAULTS["peak_marker_color"])
        )
        np.testing.assert_allclose(
            green_only[0].get_colors()[0],
            to_rgba(DEFAULTS["green_peak_marker_color"]),
        )

    def test_comb_is_not_created_when_marker_family_is_disabled_or_empty(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = self._write_csv(
                temp_dir, "wavelength,intensity\n10,20\n"
            )

            _, disabled_ax = self._plot_from(
                data_path,
                mark_peaks=False,
                mark_green_peaks=False,
                peak_comb_visible=True,
                green_peak_comb_visible=True,
            )
            _, empty_ax = self._plot_from(
                data_path,
                mark_peaks=True,
                peak_target_wavelengths=(),
                mark_green_peaks=True,
                green_peak_target_wavelengths=(),
                peak_comb_visible=True,
                green_peak_comb_visible=True,
            )

        self.assertEqual(len(self._line_collections(disabled_ax)), 0)
        self.assertEqual(len(self._line_collections(empty_ax)), 0)

    def test_comb_top_equal_to_stem_bottom_is_allowed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = self._write_csv(
                temp_dir, "wavelength,intensity\n10,100\n"
            )

            _, ax = self._plot_from(
                data_path,
                mark_peaks=True,
                peak_target_wavelengths=(10.0,),
                peak_comb_visible=True,
                peak_comb_height=100,
                peak_comb_stem_bottom=100,
            )

        combs = self._line_collections(ax)
        self.assertEqual(len(combs), 1)
        np.testing.assert_allclose(
            combs[0].get_segments(),
            [[[10.0, 100.0], [10.0, 100.0]], [[10.0, 100.0], [10.0, 100.0]]],
        )

    def test_invalid_comb_top_or_bottom_raises_clear_value_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = self._write_csv(
                temp_dir, "wavelength,intensity\n10,100\n"
            )

            invalid_pairs = (
                (99, 100),
                (np.nan, 0),
                (np.inf, 0),
                (-np.inf, 0),
                (100, np.nan),
                (100, np.inf),
                (100, -np.inf),
            )
            for height, bottom in invalid_pairs:
                with self.subTest(height=height, bottom=bottom):
                    with self.assertRaisesRegex(
                        ValueError,
                        "peak_comb_height and peak_comb_stem_bottom must be finite "
                        "with top greater than or equal to bottom",
                    ):
                        self._plot_from(
                            data_path,
                            mark_peaks=True,
                            peak_target_wavelengths=(10.0,),
                            peak_comb_visible=True,
                            peak_comb_height=height,
                            peak_comb_stem_bottom=bottom,
                        )

    def test_green_comb_rejects_top_below_stem_bottom(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = self._write_csv(
                temp_dir, "wavelength,intensity\n20,100\n"
            )

            with self.assertRaisesRegex(
                ValueError,
                "peak_comb_height and peak_comb_stem_bottom must be finite "
                "with top greater than or equal to bottom",
            ):
                self._plot_from(
                    data_path,
                    mark_peaks=False,
                    mark_green_peaks=True,
                    green_peak_target_wavelengths=(20.0,),
                    green_peak_comb_visible=True,
                    peak_comb_height=99,
                    peak_comb_stem_bottom=100,
                )

    def test_comb_overrides_do_not_mutate_defaults(self):
        defaults_before = DEFAULTS.copy()
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = self._write_csv(
                temp_dir, "wavelength,intensity\n10,20\n20,30\n"
            )

            _, ax = self._plot_from(
                data_path,
                mark_peaks=True,
                peak_target_wavelengths=(10.0,),
                mark_green_peaks=True,
                green_peak_target_wavelengths=(20.0,),
                peak_comb_visible=True,
                green_peak_comb_visible=True,
                peak_comb_height=200,
                peak_comb_stem_bottom=150,
                peak_comb_line_width=2.5,
                green_peak_comb_line_width=3.5,
                peak_comb_zorder=7,
                green_peak_comb_zorder=8,
            )

        combs = self._line_collections(ax)
        self.assertEqual(len(combs), 2)
        np.testing.assert_allclose(combs[0].get_linewidths(), [2.5])
        np.testing.assert_allclose(combs[1].get_linewidths(), [3.5])
        self.assertEqual(combs[0].get_zorder(), 7)
        self.assertEqual(combs[1].get_zorder(), 8)
        self.assertEqual(DEFAULTS, defaults_before)

    def test_peak_adjustment_defaults_are_centralized(self):
        self.assertIs(DEFAULTS.get("peak_adjustment_enabled"), True)
        self.assertEqual(DEFAULTS.get("peak_adjustment_bounds"), (278.50, 279.97))
        self.assertEqual(DEFAULTS.get("peak_adjustment_factor"), 0.7)

    def test_peak_adjustment_matches_linear_endpoint_baseline_exact_values(self):
        adjust = getattr(figure6, "_adjust_peak_region", None)
        self.assertIsNotNone(adjust)
        wavelength = np.asarray([278.49, 278.50, 279.53, 279.58, 279.97, 280.00])
        raw = np.asarray([123.0, 4859.0, 277200.0, 276300.0, 77000.0, 456.0])
        raw_before = raw.copy()

        adjusted = adjust(wavelength, raw, (278.50, 279.97), 0.7)

        np.testing.assert_allclose(
            adjusted,
            [
                123.0,
                4859.0,
                210662.032653061,
                210768.165306122,
                77000.0,
                456.0,
            ],
            rtol=0,
            atol=1e-9,
        )
        np.testing.assert_array_equal(raw, raw_before)
        self.assertFalse(np.shares_memory(adjusted, raw))

    def test_adjustment_disabled_keeps_main_spectrum_raw(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = self._write_csv(
                temp_dir, "wavelength,intensity\n0,10\n1,100\n2,30\n"
            )

            _, ax = self._plot_from(
                data_path,
                peak_adjustment_enabled=False,
            )

        np.testing.assert_allclose(ax.lines[0].get_ydata(), [10.0, 100.0, 30.0])

    def test_main_line_is_adjusted_but_marker_search_uses_raw_intensity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = self._write_csv(
                temp_dir, "wavelength,intensity\n0,10\n1,100\n2,30\n"
            )

            _, ax = self._plot_from(
                data_path,
                peak_adjustment_enabled=True,
                peak_adjustment_bounds=(0.0, 2.0),
                peak_adjustment_factor=0.0,
                mark_peaks=True,
                peak_target_wavelengths=(1.0,),
            )

        np.testing.assert_allclose(ax.lines[0].get_ydata(), [10.0, 20.0, 30.0])
        np.testing.assert_allclose(
            self._path_collections(ax)[0].get_offsets(), [[1.0, 100.0]]
        )

    def test_real_adjustment_values_and_csv_bytes_are_unchanged(self):
        data_path = os.path.join(HERE, "data.csv")
        with open(data_path, "rb") as data_file:
            before_hash = hashlib.sha256(data_file.read()).hexdigest()

        _, ax = self._plot_from(
            data_path,
            peak_adjustment_enabled=True,
        )

        xdata = np.asarray(ax.lines[0].get_xdata())
        ydata = np.asarray(ax.lines[0].get_ydata())
        self.assertAlmostEqual(ydata[np.flatnonzero(xdata == 279.53)[0]], 210662.032653061)
        self.assertAlmostEqual(ydata[np.flatnonzero(xdata == 279.58)[0]], 210768.165306122)
        with open(data_path, "rb") as data_file:
            after_hash = hashlib.sha256(data_file.read()).hexdigest()
        self.assertEqual(after_hash, before_hash)

    def test_invalid_peak_adjustment_configuration_raises_clear_value_error(self):
        adjust = getattr(figure6, "_adjust_peak_region", None)
        self.assertIsNotNone(adjust)
        wavelength = np.asarray([0.0, 1.0, 2.0])
        intensity = np.asarray([10.0, 20.0, 30.0])

        invalid_bounds = ((0.0,), (np.nan, 2.0), (0.0, np.inf), (1.0, 1.0), (2.0, 1.0))
        for bounds in invalid_bounds:
            with self.subTest(bounds=bounds):
                with self.assertRaisesRegex(
                    ValueError,
                    "peak_adjustment_bounds must contain two finite increasing values",
                ):
                    adjust(wavelength, intensity, bounds, 0.7)

        for factor in (-0.1, 1.1, np.nan, np.inf, -np.inf):
            with self.subTest(factor=factor):
                with self.assertRaisesRegex(
                    ValueError,
                    "peak_adjustment_factor must be finite and between 0 and 1",
                ):
                    adjust(wavelength, intensity, (0.0, 2.0), factor)

    def test_peak_text_defaults_are_centralized(self):
        expected = {
            "peak_text_visible": True,
            "peak_text": "Eu",
            "peak_text_xoffset": 0,
            "peak_text_yoffset": 5000,
            "green_peak_text_visible": True,
            "green_peak_text": "Yb",
            "green_peak_text_xoffset": 0,
            "green_peak_text_yoffset": 5000,
            "peak_text_fontsize": 12,
            "peak_text_fontweight": "semibold",
            "peak_text_ha": "center",
            "peak_text_va": "bottom",
            "peak_text_rotation": 0,
            "peak_text_zorder": 6,
        }
        for key, value in expected.items():
            self.assertEqual(DEFAULTS.get(key), value)

    def test_red_and_green_peak_text_positions_colors_and_styles(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = self._write_csv(
                temp_dir,
                "wavelength,intensity\n10,20\n12,30\n20,40\n24,50\n",
            )

            _, ax = self._plot_from(
                data_path,
                mark_peaks=True,
                peak_target_wavelengths=(10.0, 12.0),
                mark_green_peaks=True,
                green_peak_target_wavelengths=(20.0, 24.0),
                peak_comb_visible=True,
                green_peak_comb_visible=True,
                peak_comb_height=100,
                peak_comb_stem_bottom=60,
                peak_text_visible=True,
                green_peak_text_visible=True,
            )

        self.assertEqual(len(ax.texts), 2)
        red_text, green_text = ax.texts
        self.assertEqual(red_text.get_text(), "Eu")
        self.assertEqual(green_text.get_text(), "Yb")
        np.testing.assert_allclose(red_text.get_position(), [11.0, 5100.0])
        np.testing.assert_allclose(green_text.get_position(), [22.0, 5100.0])
        self.assertEqual(red_text.get_color(), DEFAULTS["peak_marker_color"])
        self.assertEqual(
            green_text.get_color(), DEFAULTS["green_peak_marker_color"]
        )
        for text_artist in ax.texts:
            self.assertEqual(text_artist.get_fontsize(), 12)
            self.assertEqual(text_artist.get_fontweight(), "semibold")
            self.assertEqual(text_artist.get_ha(), "center")
            self.assertEqual(text_artist.get_va(), "bottom")
            self.assertEqual(text_artist.get_rotation(), 0)
            self.assertEqual(text_artist.get_zorder(), 6)

    def test_real_peak_texts_use_actual_family_extents(self):
        data_path = os.path.join(HERE, "data.csv")

        _, ax = self._plot_from(
            data_path,
            mark_peaks=True,
            mark_green_peaks=True,
            peak_comb_visible=True,
            green_peak_comb_visible=True,
            peak_text_visible=True,
            green_peak_text_visible=True,
        )

        self.assertEqual([item.get_text() for item in ax.texts], ["Eu", "Yb"])
        np.testing.assert_allclose(ax.texts[0].get_position(), [417.705, 255000.0])
        np.testing.assert_allclose(ax.texts[1].get_position(), [290.565, 255000.0])

    def test_real_peak_text_bboxes_fit_inside_axes_and_figure_at_300_dpi(self):
        data_path = os.path.join(HERE, "data.csv")

        fig, ax = plot_spectrum(
            {
                "data_path": data_path,
                "output_path": None,
                "show": False,
            }
        )
        fig.set_dpi(300)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        axes_bbox = ax.get_window_extent(renderer)
        figure_bbox = fig.bbox

        self.assertEqual([item.get_text() for item in ax.texts], ["Eu", "Yb"])
        for text_artist in ax.texts:
            text_bbox = text_artist.get_window_extent(renderer)
            for outer_bbox in (axes_bbox, figure_bbox):
                self.assertGreaterEqual(text_bbox.x0, outer_bbox.x0)
                self.assertGreaterEqual(text_bbox.y0, outer_bbox.y0)
                self.assertLessEqual(text_bbox.x1, outer_bbox.x1)
                self.assertLessEqual(text_bbox.y1, outer_bbox.y1)

    def test_peak_text_requires_family_comb_text_and_nonempty_targets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = self._write_csv(
                temp_dir, "wavelength,intensity\n10,20\n20,30\n"
            )
            cases = (
                {"mark_peaks": False, "peak_comb_visible": True, "peak_text_visible": True},
                {"mark_peaks": True, "peak_comb_visible": False, "peak_text_visible": True},
                {"mark_peaks": True, "peak_comb_visible": True, "peak_text_visible": False},
                {
                    "mark_peaks": True,
                    "peak_target_wavelengths": (),
                    "peak_comb_visible": True,
                    "peak_text_visible": True,
                },
            )
            for red_settings in cases:
                settings = {
                    "mark_green_peaks": False,
                    "peak_target_wavelengths": (10.0,),
                    "peak_comb_height": 100,
                    "peak_comb_stem_bottom": 60,
                }
                settings.update(red_settings)
                with self.subTest(red_settings=red_settings):
                    _, ax = self._plot_from(data_path, **settings)
                    self.assertEqual(len(ax.texts), 0)

            green_cases = (
                {
                    "mark_green_peaks": False,
                    "green_peak_comb_visible": True,
                    "green_peak_text_visible": True,
                },
                {
                    "mark_green_peaks": True,
                    "green_peak_comb_visible": False,
                    "green_peak_text_visible": True,
                },
                {
                    "mark_green_peaks": True,
                    "green_peak_comb_visible": True,
                    "green_peak_text_visible": False,
                },
                {
                    "mark_green_peaks": True,
                    "green_peak_target_wavelengths": (),
                    "green_peak_comb_visible": True,
                    "green_peak_text_visible": True,
                },
            )
            for green_settings in green_cases:
                settings = {
                    "mark_peaks": False,
                    "green_peak_target_wavelengths": (20.0,),
                    "peak_comb_height": 100,
                    "peak_comb_stem_bottom": 60,
                }
                settings.update(green_settings)
                with self.subTest(green_settings=green_settings):
                    _, ax = self._plot_from(data_path, **settings)
                    self.assertEqual(len(ax.texts), 0)

    def test_peak_text_overrides_do_not_mutate_defaults(self):
        defaults_before = DEFAULTS.copy()
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = self._write_csv(
                temp_dir, "wavelength,intensity\n10,20\n20,30\n"
            )

            _, ax = self._plot_from(
                data_path,
                mark_peaks=True,
                peak_target_wavelengths=(10.0,),
                mark_green_peaks=True,
                green_peak_target_wavelengths=(20.0,),
                peak_comb_visible=True,
                green_peak_comb_visible=True,
                peak_comb_height=100,
                peak_comb_stem_bottom=60,
                peak_text_visible=True,
                peak_text="R",
                peak_text_xoffset=1,
                peak_text_yoffset=10,
                green_peak_text_visible=True,
                green_peak_text="G",
                green_peak_text_xoffset=-1,
                green_peak_text_yoffset=20,
                peak_text_fontsize=14,
                peak_text_zorder=9,
            )

        self.assertEqual(len(ax.texts), 2)
        self.assertEqual([item.get_text() for item in ax.texts], ["R", "G"])
        np.testing.assert_allclose(ax.texts[0].get_position(), [11.0, 110.0])
        np.testing.assert_allclose(ax.texts[1].get_position(), [19.0, 120.0])
        self.assertEqual([item.get_fontsize() for item in ax.texts], [14, 14])
        self.assertEqual([item.get_zorder() for item in ax.texts], [9, 9])
        self.assertEqual(DEFAULTS, defaults_before)


if __name__ == "__main__":
    unittest.main()
