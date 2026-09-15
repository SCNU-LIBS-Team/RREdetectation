import sys
import types
import unittest
from pathlib import Path

import matplotlib
from matplotlib.colors import to_rgba
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

fake_error_evaluation = types.ModuleType("error_evaluation")
fake_error_evaluation.U_Calculate = lambda *args, **kwargs: None
fake_error_evaluation.rel_intensity = lambda *args, **kwargs: None
sys.modules.setdefault("error_evaluation", fake_error_evaluation)

import Elements_detectation as ed


class ComputeElementConfidencePlotStyleTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_original_spectrum_plot_uses_white_background_and_black_foreground(self):
        element_name = "XeI"
        elements = {
            element_name: {
                "data": np.array(
                    [[500.0, 1.0, 1.0, 1.0, 1.0]],
                    dtype=float,
                )
            }
        }

        ed.compute_element_confidence_shape(
            elements,
            peak_wl=np.array([500.0]),
            peak_int=np.array([10.0]),
            global_wl=np.array([499.0, 500.0, 501.0]),
            global_intensity=np.array([0.0, 10.0, 0.0]),
            plot=True,
            target=element_name,
        )

        expected_title = f"Original Spectrum with {element_name} Peaks Marked"
        spectrum_axes = [
            ax
            for figure_number in plt.get_fignums()
            for ax in plt.figure(figure_number).axes
            if ax.get_title() == expected_title
        ]
        self.assertEqual(len(spectrum_axes), 1)

        ax = spectrum_axes[0]
        self.assertEqual(to_rgba(ax.figure.get_facecolor()), to_rgba("white"))
        self.assertEqual(to_rgba(ax.get_facecolor()), to_rgba("white"))

        original_spectrum = next(
            line for line in ax.lines if line.get_label() == "Original Spectrum"
        )
        self.assertEqual(to_rgba(original_spectrum.get_color()), to_rgba("black"))

        for spine in ax.spines.values():
            self.assertEqual(to_rgba(spine.get_edgecolor()), to_rgba("black"))

        self.assertEqual(to_rgba(ax.title.get_color()), to_rgba("black"))
        self.assertEqual(to_rgba(ax.xaxis.label.get_color()), to_rgba("black"))
        self.assertEqual(to_rgba(ax.yaxis.label.get_color()), to_rgba("black"))

        ax.figure.canvas.draw()
        for tick in [*ax.xaxis.get_major_ticks(), *ax.yaxis.get_major_ticks()]:
            self.assertEqual(to_rgba(tick.tick1line.get_color()), to_rgba("black"))
            self.assertEqual(to_rgba(tick.label1.get_color()), to_rgba("black"))


if __name__ == "__main__":
    unittest.main()
