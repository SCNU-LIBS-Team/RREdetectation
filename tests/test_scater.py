import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import matplotlib
from matplotlib.colors import to_rgba

if "matplotlib.pyplot" not in sys.modules:
    matplotlib.use("Agg")

warnings.filterwarnings(
    "ignore",
    message="Using or importing the ABCs from 'collections'",
    category=DeprecationWarning,
    module="matplotlib.cbook",
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import Scater


class ScaterPlotTests(unittest.TestCase):
    def test_load_abs_residuals_reads_numeric_first_column(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "residuals.csv"
            csv_path.write_text(
                "abs_residual,other\n0.12,10\n-0.30,20\nbad,30\n0.05,40\n",
                encoding="utf-8",
            )

            values = Scater.load_abs_residuals_from_csv(csv_path)

        self.assertEqual(values.tolist(), [0.12, 0.3, 0.05])

    def test_load_abs_residuals_keeps_first_numeric_row_without_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "residuals.csv"
            csv_path.write_text("0.12\n0.30\n0.05\n", encoding="utf-8")

            values = Scater.load_abs_residuals_from_csv(csv_path)

        self.assertEqual(values.tolist(), [0.12, 0.3, 0.05])

    def test_plot_residual_abs_scatter_uses_expected_axes_and_style(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "residuals.csv"
            csv_path.write_text("abs_residual\n0.10\n0.20\n0.15\n", encoding="utf-8")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                fig, ax, x_values, y_values = Scater.plot_residual_abs_scatter(
                    csv_path,
                    show_plot=False,
                )

        self.assertEqual(x_values.tolist(), [1, 2, 3])
        self.assertEqual(y_values.tolist(), [0.1, 0.2, 0.15])
        self.assertEqual(ax.get_xlabel(), "Index")
        self.assertEqual(ax.get_ylabel(), "Absolute Residual (K)")
        self.assertEqual(ax.get_title(), "Temperature iteration residual plot")
        legend = ax.get_legend()
        self.assertIsNotNone(legend)
        self.assertTrue(legend.get_frame_on())
        self.assertTrue(legend.get_frame().get_visible())
        self.assertEqual([text.get_text() for text in legend.get_texts()], ["Mean = 0.1500 K", "Std = 0.0408 K"])
        self.assertEqual([text.get_color() for text in legend.get_texts()], ["black", "black"])
        self.assertEqual(len(ax.lines), 1)
        self.assertEqual(ax.lines[0].get_color(), "tab:orange")
        self.assertEqual(ax.lines[0].get_linestyle(), "--")
        self.assertEqual(ax.lines[0].get_linewidth(), 2)
        for y_value in ax.lines[0].get_ydata():
            self.assertAlmostEqual(y_value, 0.15)
        self.assertEqual(len(ax.collections), 2)
        self.assertEqual(ax.collections[0].get_facecolors()[0].tolist(), list(to_rgba("tab:green", 0.85)))
        self.assertEqual(ax.collections[1].get_facecolors()[0].tolist(), list(to_rgba("tab:red", 0.85)))
        self.assertEqual(ax.collections[0].get_offsets()[:, 1].tolist(), [0.1, 0.15])
        self.assertEqual(ax.collections[1].get_offsets()[:, 1].tolist(), [0.2])
        visible_gridlines = [
            line for line in ax.get_xgridlines() + ax.get_ygridlines() if line.get_visible()
        ]
        self.assertGreater(len(visible_gridlines), 0)
        for line in visible_gridlines:
            self.assertAlmostEqual(line.get_alpha(), 0.3)
        self.assertEqual(ax.xaxis._minor_tick_kw["tickdir"], "in")
        self.assertEqual(ax.yaxis._minor_tick_kw["tickdir"], "in")
        self.assertEqual(ax.xaxis._minor_tick_kw["width"], 2.0)
        self.assertEqual(ax.yaxis._minor_tick_kw["width"], 2.0)
        self.assertEqual(ax.xaxis._minor_tick_kw["size"], 6)
        self.assertEqual(ax.yaxis._minor_tick_kw["size"], 6)
        self.assertEqual(ax.spines["left"].get_linewidth(), 1.8)


if __name__ == "__main__":
    unittest.main()
