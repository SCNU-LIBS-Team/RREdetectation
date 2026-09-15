import unittest

import matplotlib
from matplotlib.collections import PolyCollection
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt

from MultiPeakfit.Gaussfit import GaussMultiPeakFitter


class GaussMultiPeakFitterPlotTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_wavelength_intensity_plot_has_curves_without_area_fills(self):
        wavelength = np.array([499.8, 500.0, 500.2], dtype=float)
        spectrum = np.array([1.0, 5.0, 1.0], dtype=float)
        component = np.array([0.8, 4.0, 0.8], dtype=float)

        fitter = GaussMultiPeakFitter(
            wl=wavelength,
            rel_int=spectrum,
            extrema_idx=[1],
            fwhm_selected=[0.1],
            wl_np=wavelength,
            selected_idx=[1],
        )
        fitter.component_fits = [component]
        fitter.total_fit = component.copy()
        fitter.fitted_params = [(4.0, 500.0, 0.04)]
        fitter.fitted_mu = np.array([500.0])
        fitter.fitted_amp = np.array([4.0])

        fitter.plot(
            peak_wl=np.array([500.0]),
            peak_int=np.array([5.0]),
        )

        ax = plt.gca()
        self.assertEqual(ax.get_title(), "Wavelength-Intensity Spectrum")

        curve_labels = {line.get_label() for line in ax.lines}
        self.assertTrue(
            {"wl-int", "Gaussian Components", "Gaussian Sum Fit"}
            <= curve_labels
        )

        area_fills = [
            collection
            for collection in ax.collections
            if isinstance(collection, PolyCollection)
        ]
        self.assertEqual(area_fills, [])

    def test_wavelength_intensity_plot_skips_zero_amplitude_components(self):
        wavelength = np.array([499.8, 500.0, 500.2], dtype=float)
        spectrum = np.array([1.0, 5.0, 1.0], dtype=float)
        positive_component = np.array([0.8, 4.0, 0.8], dtype=float)
        zero_component = np.zeros_like(wavelength)

        fitter = GaussMultiPeakFitter(
            wl=wavelength,
            rel_int=spectrum,
            extrema_idx=[1, 2],
            fwhm_selected=[0.1, 0.1],
            wl_np=wavelength,
            selected_idx=[1, 2],
        )
        fitter.component_fits = [zero_component, positive_component]
        fitter.total_fit = positive_component.copy()
        fitter.fitted_params = [
            (0.0, 500.2, 0.04),
            (4.0, 500.0, 0.04),
        ]
        fitter.fitted_mu = np.array([500.2, 500.0])
        fitter.fitted_amp = np.array([0.0, 4.0])

        fitter.plot(
            peak_wl=np.array([500.0]),
            peak_int=np.array([5.0]),
        )

        green_component_lines = [
            line
            for line in plt.gca().lines
            if line.get_color() == "tab:green"
        ]
        self.assertEqual(len(green_component_lines), 1)
        self.assertEqual(
            green_component_lines[0].get_label(),
            "Gaussian Components",
        )
        np.testing.assert_allclose(
            green_component_lines[0].get_ydata(),
            positive_component,
        )


if __name__ == "__main__":
    unittest.main()
