# -*- coding: utf-8 -*-
"""Regression tests for the 378.5-385.6 nm Sum + Eu II plot."""

from __future__ import absolute_import

import copy
import os
import sys
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
from PIL import Image


HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import plot_example1 as subject


class PlotExample1Tests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def _write_csv(self, directory, rows=None):
        if rows is None:
            rows = [
                (378.5, 1.0, 0.1, 9.0, 0.2, 0.3),
                (380.0, 2.0, 0.5, 8.0, 0.4, 0.6),
                (382.0, 6.0, 3.0, 7.0, 2.0, 1.0),
                (384.0, 2.0, 0.5, 6.0, 0.4, 0.6),
                (385.6, 1.0, 0.1, 5.0, 0.2, 0.3),
            ]
        path = os.path.join(directory, "spectrum.csv")
        with open(path, "w") as stream:
            stream.write(
                "Wavelength (nm),Sum(calc),Eu II (8.6e-3),"
                "Tm II (6.2e-3),Ti II (1.0e-1),Fe I (9.7e-2)\n")
            for row in rows:
                stream.write(",".join(str(value) for value in row) + "\n")
        return path

    def _write_text(self, directory, text, name="spectrum.csv"):
        path = os.path.join(directory, name)
        with open(path, "w") as stream:
            stream.write(text)
        return path

    def _plot(self, data_path, **overrides):
        config = {
            "data_path": data_path,
            "output_path": None,
            "show": False,
            "draw_clipped_left_label": False,
        }
        config.update(overrides)
        return subject.plot_example1(config)

    def test_all_common_tuning_sections_are_centralized_in_defaults(self):
        expected = {
            "data_path", "output_path", "figsize", "dpi", "background",
            "reference_canvas_px", "axes", "wavelength_column", "minimum_points",
            "xlim", "ylim", "data_window", "xticks", "xticks_minor",
            "yticks", "yticks_minor", "grid_major", "grid_minor", "font",
            "curves", "smoothing", "spines", "legend", "labels", "show",
        }
        self.assertEqual(expected - set(subject.DEFAULTS), set())

    def test_default_schema_and_window_target_eu_ii_at_378_5_to_385_6_nm(self):
        self.assertEqual(subject.DEFAULTS["wavelength_column"], "Wavelength (nm)")
        self.assertEqual(subject.DEFAULTS["curves"]["blue"]["column"], "Sum(calc)")
        self.assertIn("red", subject.DEFAULTS["curves"])
        self.assertEqual(subject.DEFAULTS["curves"]["red"]["column"],
                         "Eu II (8.6e-3)")
        self.assertEqual(subject.DEFAULTS["curves"]["red"]["color"],
                         "#D73027")
        self.assertIn("ti", subject.DEFAULTS["curves"])
        self.assertEqual(subject.DEFAULTS["curves"]["ti"]["column"],
                         "Ti II (1.0e-1)")
        self.assertEqual(subject.DEFAULTS["curves"]["ti"]["color"],
                         "#897CD3")
        self.assertIn("fe", subject.DEFAULTS["curves"])
        self.assertEqual(subject.DEFAULTS["curves"]["fe"]["column"],
                         "Fe I (9.7e-2)")
        self.assertEqual(subject.DEFAULTS["curves"]["fe"]["color"],
                         "#FEE090")
        self.assertTrue(subject.DEFAULTS["legend"]["frameon"])
        self.assertIn("linewidth", subject.DEFAULTS["legend"])
        self.assertEqual(subject.DEFAULTS["legend"]["framealpha"], 0)
        self.assertEqual(subject.DEFAULTS["data_window"], (378.5, 385.6))

    def test_defaults_draw_four_curves_inside_plain_black_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path = self._write_csv(directory)
            fig, ax = self._plot(data_path)

        self.assertEqual(len(ax.lines), 4)
        np.testing.assert_allclose(ax.lines[0].get_ydata(), [1, 2, 6, 2, 1])
        np.testing.assert_allclose(ax.lines[1].get_ydata(), [0.1, 0.5, 3, 0.5, 0.1])
        np.testing.assert_allclose(ax.lines[2].get_ydata(), [0.2, 0.4, 2, 0.4, 0.2])
        np.testing.assert_allclose(ax.lines[3].get_ydata(), [0.3, 0.6, 1, 0.6, 0.3])
        self.assertEqual(to_rgba(ax.lines[0].get_color()), to_rgba("#4575B4"))
        self.assertEqual(to_rgba(ax.lines[1].get_color()), to_rgba("#D73027"))
        self.assertEqual(to_rgba(ax.lines[2].get_color()), to_rgba("#897CD3"))
        self.assertEqual(to_rgba(ax.lines[3].get_color()), to_rgba("#FEE090"))
        legend = ax.get_legend()
        self.assertIsNotNone(legend)
        self.assertEqual(legend.get_frame().get_linewidth(),
                         subject.DEFAULTS["legend"]["linewidth"])
        self.assertEqual(legend.get_frame().get_alpha(), 0)
        self.assertEqual(ax.get_xlabel(), "")
        self.assertEqual(ax.get_ylabel(), "")
        self.assertTrue(all(spine.get_visible() for spine in ax.spines.values()))
        self.assertTrue(all(to_rgba(spine.get_edgecolor()) == to_rgba("black")
                            for spine in ax.spines.values()))
        self.assertTrue(all(spine.get_linewidth() ==
                            subject.DEFAULTS["spines"]["linewidth"]
                            for spine in ax.spines.values()))
        self.assertFalse(any(line.get_visible() for line in ax.get_xgridlines()))
        self.assertFalse(any(line.get_visible() for line in ax.get_ygridlines()))
        self.assertFalse(any(label.get_visible()
                             for label in ax.get_xticklabels(which="both")))
        self.assertFalse(any(label.get_visible()
                             for label in ax.get_yticklabels(which="both")))
        self.assertEqual(tuple(fig.get_size_inches()), (12.53, 8.0))

    def test_nested_overrides_change_style_without_discarding_curve_columns(self):
        before = copy.deepcopy(subject.DEFAULTS)
        with tempfile.TemporaryDirectory() as directory:
            data_path = self._write_csv(directory)
            _, ax = self._plot(
                data_path,
                curves={
                    "blue": {"color": "black", "linewidth": 5.0},
                    "red": {"color": "magenta", "linewidth": 1.0},
                },
                labels={"x": "nm", "y": "intensity", "title": "Eu II"},
                legend={"visible": True, "location": "upper right"},
                spines={"visible": True, "color": "green", "linewidth": 1.25},
            )

        self.assertEqual(ax.lines[0].get_color(), "black")
        self.assertEqual(ax.lines[0].get_linewidth(), 5.0)
        self.assertEqual(ax.lines[1].get_color(), "magenta")
        self.assertEqual(ax.lines[1].get_linewidth(), 1.0)
        self.assertEqual(ax.get_xlabel(), "nm")
        self.assertEqual(ax.get_ylabel(), "intensity")
        self.assertEqual(ax.get_title(), "Eu II")
        self.assertIsNotNone(ax.get_legend())
        self.assertTrue(all(spine.get_visible() for spine in ax.spines.values()))
        self.assertEqual(subject.DEFAULTS, before)

    def test_optional_moving_average_smoothing_is_configurable(self):
        rows = [
            (379.0, 0.0, 0.0, 9.0, 0.0, 0.0),
            (380.0, 0.0, 0.0, 8.0, 0.0, 0.0),
            (381.0, 9.0, 3.0, 7.0, 6.0, 3.0),
            (382.0, 0.0, 0.0, 6.0, 0.0, 0.0),
            (383.0, 0.0, 0.0, 5.0, 0.0, 0.0),
        ]
        with tempfile.TemporaryDirectory() as directory:
            data_path = self._write_csv(directory, rows)
            _, ax = self._plot(
                data_path,
                smoothing={"enabled": True, "method": "moving_average", "window": 3},
            )

        np.testing.assert_allclose(ax.lines[0].get_ydata(), [0, 3, 3, 3, 0])
        np.testing.assert_allclose(ax.lines[1].get_ydata(), [0, 1, 1, 1, 0])

    def test_missing_required_column_is_clear_and_does_not_write_png(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path = self._write_text(
                directory,
                "Wavelength (nm),Sum(calc),Eu I\n328,1,9\n329,2,8\n",
            )
            output_path = os.path.join(directory, "should-not-exist.png")
            with self.assertRaisesRegex(ValueError, "missing required columns.*Eu II"):
                subject.plot_example1({"data_path": data_path,
                                       "output_path": output_path})
            self.assertFalse(os.path.exists(output_path))

    def test_configured_column_names_are_used_without_guessing(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path = self._write_text(
                directory,
                "lambda,total,eu_target,Eu II decoy,ti_target,fe_target\n"
                "379,1,0.1,99,0.4,0.7\n"
                "380,2,0.2,98,0.5,0.8\n"
                "381,3,0.3,97,0.6,0.9\n",
            )
            _, ax = subject.plot_example1({
                "data_path": data_path,
                "output_path": None,
                "show": False,
                "draw_clipped_left_label": False,
                "wavelength_column": "lambda",
                "curves": {
                    "blue": {"column": "total"},
                    "red": {"column": "eu_target"},
                    "ti": {"column": "ti_target"},
                    "fe": {"column": "fe_target"},
                },
            })

        np.testing.assert_allclose(ax.lines[0].get_xdata(), [379, 380, 381])
        np.testing.assert_allclose(ax.lines[0].get_ydata(), [1, 2, 3])
        np.testing.assert_allclose(ax.lines[1].get_ydata(), [0.1, 0.2, 0.3])

    def test_empty_window_and_too_few_points_are_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path = self._write_text(
                directory,
                "Wavelength (nm),Sum(calc),Eu II (8.6e-3),Ti II (1.0e-1),Fe I (9.7e-2)\n"
                "100,1,0.1,0.2,0.3\n101,2,0.2,0.4,0.5\n",
            )
            output_path = os.path.join(directory, "empty.png")
            with self.assertRaisesRegex(ValueError, "data window.*no rows"):
                subject.plot_example1({"data_path": data_path,
                                       "output_path": output_path})
            self.assertFalse(os.path.exists(output_path))

            one_point = self._write_text(
                directory,
                "Wavelength (nm),Sum(calc),Eu II (8.6e-3),Ti II (1.0e-1),Fe I (9.7e-2)\n"
                "382,1,0.1,0.2,0.3\n",
                name="one.csv",
            )
            with self.assertRaisesRegex(ValueError, "at least 2"):
                subject.plot_example1({"data_path": one_point,
                                       "output_path": output_path})
            self.assertFalse(os.path.exists(output_path))

    def test_wavelength_must_be_finite_strictly_increasing_and_unique(self):
        cases = (
            ("379,1,0.1,0.2,0.3\nbad,2,0.2,0.3,0.4\n381,3,0.3,0.4,0.5\n", "finite numeric"),
            ("379,1,0.1,0.2,0.3\n379,2,0.2,0.3,0.4\n381,3,0.3,0.4,0.5\n", "duplicate"),
            ("380,1,0.1,0.2,0.3\n379,2,0.2,0.3,0.4\n381,3,0.3,0.4,0.5\n", "strictly increasing"),
        )
        header = ("Wavelength (nm),Sum(calc),Eu II (8.6e-3),"
                  "Ti II (1.0e-1),Fe I (9.7e-2)\n")
        with tempfile.TemporaryDirectory() as directory:
            for index, (rows, message) in enumerate(cases):
                data_path = self._write_text(directory, header + rows,
                                             name="case{0}.csv".format(index))
                with self.assertRaisesRegex(ValueError, message):
                    subject.plot_example1({"data_path": data_path,
                                           "output_path": None})

    def test_curve_values_must_be_finite_numeric_inside_window(self):
        cases = (
            ("379,1,0.1,0.2,0.3\n380,bad,0.2,0.3,0.4\n381,3,0.3,0.4,0.5\n", "Sum\\(calc\\).*finite numeric"),
            ("379,1,0.1,0.2,0.3\n380,2,inf,0.3,0.4\n381,3,0.3,0.4,0.5\n", "Eu II.*finite numeric"),
        )
        header = ("Wavelength (nm),Sum(calc),Eu II (8.6e-3),"
                  "Ti II (1.0e-1),Fe I (9.7e-2)\n")
        with tempfile.TemporaryDirectory() as directory:
            for index, (rows, message) in enumerate(cases):
                data_path = self._write_text(directory, header + rows,
                                             name="curve{0}.csv".format(index))
                with self.assertRaisesRegex(ValueError, message):
                    subject.plot_example1({"data_path": data_path,
                                           "output_path": None})

    def test_axes_fraction_is_stable_when_figsize_or_dpi_changes(self):
        baseline = subject._axes_rect(subject.DEFAULTS)
        changed = copy.deepcopy(subject.DEFAULTS)
        changed.update({"figsize": (6.0, 4.0), "dpi": 240})
        self.assertEqual(subject._axes_rect(changed), baseline)
        np.testing.assert_allclose(baseline, [0.035, 0.05, 0.94, 0.92])

    def test_rcparam_sensitive_styles_are_explicit_and_applied(self):
        for grid_key in ("grid_major", "grid_minor"):
            self.assertEqual(
                {"color", "linewidth", "linestyle", "alpha", "antialiased",
                 "visible"} - set(subject.DEFAULTS[grid_key]),
                set(),
            )
        for curve_key in ("blue", "red", "ti", "fe"):
            self.assertEqual(
                {"antialiased", "solid_capstyle", "solid_joinstyle"}
                - set(subject.DEFAULTS["curves"][curve_key]),
                set(),
            )

        with tempfile.TemporaryDirectory() as directory:
            data_path = self._write_csv(directory)
            _, ax = self._plot(
                data_path,
                axes={"facecolor": "#F0F0F0", "axisbelow": False},
                curves={
                    "blue": {"antialiased": False,
                             "solid_capstyle": "round",
                             "solid_joinstyle": "bevel"},
                },
                grid_major={"visible": True, "linestyle": ":", "alpha": 0.4,
                            "antialiased": False},
            )

        self.assertEqual(to_rgba(ax.get_facecolor()), to_rgba("#F0F0F0"))
        self.assertFalse(ax.get_axisbelow())
        self.assertFalse(ax.lines[0].get_antialiased())
        self.assertEqual(ax.lines[0].get_solid_capstyle(), "round")
        self.assertEqual(ax.lines[0].get_solid_joinstyle(), "bevel")
        major_grid = ax.get_xgridlines()[0]
        self.assertEqual(major_grid.get_linestyle(), ":")
        self.assertEqual(major_grid.get_alpha(), 0.4)
        self.assertFalse(major_grid.get_antialiased())

    def test_png_integration_creates_directory_preserves_input_and_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path = self._write_csv(directory)
            with open(data_path, "rb") as stream:
                before = stream.read()
            output_path = os.path.join(directory, "nested", "result.png")

            subject.plot_example1({
                "data_path": data_path,
                "output_path": output_path,
                "figsize": (4.0, 3.0),
                "dpi": 50,
                "show": False,
                "draw_clipped_left_label": False,
            })

            self.assertTrue(os.path.isfile(output_path))
            with Image.open(output_path) as image:
                self.assertEqual(image.format, "PNG")
                self.assertEqual(image.size, (200, 150))
            with open(data_path, "rb") as stream:
                self.assertEqual(stream.read(), before)

    def test_non_png_output_is_rejected_before_file_or_directory_is_created(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path = self._write_csv(directory)
            output_path = os.path.join(directory, "nested", "misleading.jpg")

            with self.assertRaisesRegex(ValueError, "output_path.*\\.png"):
                subject.plot_example1({"data_path": data_path,
                                       "output_path": output_path})

            self.assertFalse(os.path.exists(output_path))
            self.assertFalse(os.path.exists(os.path.dirname(output_path)))


if __name__ == "__main__":
    unittest.main()
