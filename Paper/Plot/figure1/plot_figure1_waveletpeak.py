"""Detect wavelet-ridge peaks in the spectrum used by Figure 1."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


FIGURE_DIR = Path(__file__).resolve().parent
SOURCE_SCRIPT_PATH = FIGURE_DIR / "plot_figure1.py"
REPO_ROOT = FIGURE_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The legacy detector configures Chinese fonts as an import side effect. Keep
# that behavior local to the detector so this Figure 1 script retains exactly
# the same font resolution as plot_figure1.py.
_FONT_RC_KEYS = ("font.family", "font.sans-serif", "axes.unicode_minus")
_font_rc_snapshot = {
    key: list(plt.rcParams[key]) if isinstance(plt.rcParams[key], list) else plt.rcParams[key]
    for key in _FONT_RC_KEYS
}
try:
    from Wavelet_peakfinding import wavelet_peak_detection
finally:
    for _key, _value in _font_rc_snapshot.items():
        plt.rcParams[_key] = _value
    del _key, _value, _font_rc_snapshot


DEFAULT_DATA_PATH = FIGURE_DIR / "data3.csv"
DEFAULT_OUTPUT_PATH = FIGURE_DIR / "figure1_waveletpeak.png"


def _load_current_figure1_module():
    """Load the existing Figure 1 plot as the single source of style settings."""
    if not SOURCE_SCRIPT_PATH.is_file():
        raise FileNotFoundError("Figure 1 script not found: {}".format(SOURCE_SCRIPT_PATH))
    spec = importlib.util.spec_from_file_location(
        "_figure1_source_for_wavelet_peaks",
        str(SOURCE_SCRIPT_PATH),
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not load Figure 1 script: {}".format(SOURCE_SCRIPT_PATH))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SOURCE = _load_current_figure1_module()

DEFAULT_WAVELENGTH_MIN = _SOURCE.DEFAULT_WAVELENGTH_MIN
DEFAULT_WAVELENGTH_MAX = _SOURCE.DEFAULT_WAVELENGTH_MAX
DEFAULT_WAVELET = "mexh"
DEFAULT_SCALES = np.arange(1, 11)
DEFAULT_NEIGHBOR = 4
DEFAULT_MIN_LENGTH = 3
DEFAULT_COEFFICIENT_THRESHOLD = 700.0
DEFAULT_CORRECTION_WINDOW = 5

FIGSIZE = _SOURCE.FIGSIZE
SAVE_DPI = _SOURCE.SAVE_DPI
SPECTRUM_COLOR = _SOURCE.SPECTRUM_COLOR
SPECTRUM_LINEWIDTH = _SOURCE.SPECTRUM_LINEWIDTH
SPECTRUM_ALPHA = _SOURCE.DEFAULT_SPECTRUM_ALPHA
PEAK_COLOR = "#D73027"
PEAK_MARKER_SIZE = _SOURCE.DEFAULT_PEAK_MARKER_SIZE
LEGEND_LOCATION = "upper right"
LEGEND_FONTSIZE = _SOURCE.TICK_LABELSIZE


def figure1_reference_y_limits(
    wavelengths,
    intensities,
    wavelength_min: float = DEFAULT_WAVELENGTH_MIN,
    wavelength_max: float = DEFAULT_WAVELENGTH_MAX,
) -> Tuple[float, float]:
    """Return the y span used by the current ``figure1_spectrum`` plot."""
    wavelengths = np.asarray(wavelengths, dtype=float)
    intensities = np.asarray(intensities, dtype=float)
    region_mask = (wavelengths >= wavelength_min) & (wavelengths <= wavelength_max)
    regional_intensities = intensities[region_mask]
    if regional_intensities.size == 0:
        raise ValueError("The selected wavelength interval contains no data points.")

    regional_max = max(float(np.max(regional_intensities)), 1.0)
    try:
        reference_peak_indices = _SOURCE.select_manual_peak_indices(
            wavelengths,
            intensities,
            _SOURCE.DEFAULT_MANUAL_PEAK_WAVELENGTHS,
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
            boundary_margin_nm=_SOURCE.DEFAULT_BOUNDARY_MARGIN_NM,
            prominence_ratio=_SOURCE.DEFAULT_PEAK_PROMINENCE_RATIO,
            min_distance_nm=_SOURCE.DEFAULT_PEAK_MIN_DISTANCE_NM,
        )
        reference_peak_max = float(np.max(intensities[reference_peak_indices]))
    except ValueError:
        # Custom wavelength ranges may omit one or more of Figure 1's six
        # manual peaks. Preserve the same spacing rule using the regional max.
        reference_peak_max = regional_max

    top_base_y = regional_max * (1 + _SOURCE.DEFAULT_TOP_LINE_MARGIN_RATIO)
    top_line_top_y = top_base_y + reference_peak_max
    return 0.0, top_line_top_y * (1 + _SOURCE.DEFAULT_TOP_LINE_PADDING_RATIO)


def load_spectrum_region(
    data_path: Union[str, Path] = DEFAULT_DATA_PATH,
    wavelength_min: float = DEFAULT_WAVELENGTH_MIN,
    wavelength_max: float = DEFAULT_WAVELENGTH_MAX,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load the two-column Figure 1 CSV and return the selected wavelength region."""
    data_path = Path(data_path)
    if not data_path.is_file():
        raise FileNotFoundError("Spectrum data file not found: {}".format(data_path))
    if not np.isfinite(wavelength_min) or not np.isfinite(wavelength_max):
        raise ValueError("Wavelength limits must be finite numbers.")
    if wavelength_min >= wavelength_max:
        raise ValueError("wavelength_min must be smaller than wavelength_max.")

    data = np.genfromtxt(
        str(data_path),
        delimiter=",",
        names=True,
        dtype=float,
        encoding="utf-8-sig",
    )
    field_names = data.dtype.names
    if field_names is None or set(field_names) != {"wavelength", "intensity"}:
        raise ValueError(
            "CSV must contain exactly the columns 'wavelength' and 'intensity'."
        )

    wavelengths = np.atleast_1d(np.asarray(data["wavelength"], dtype=float))
    intensities = np.atleast_1d(np.asarray(data["intensity"], dtype=float))
    if wavelengths.size != intensities.size or wavelengths.size == 0:
        raise ValueError("Spectrum columns must be non-empty and have equal length.")
    if not np.isfinite(wavelengths).all() or not np.isfinite(intensities).all():
        raise ValueError("Spectrum data must contain only finite numeric values.")

    mask = (wavelengths >= wavelength_min) & (wavelengths <= wavelength_max)
    region_wavelengths = wavelengths[mask]
    region_intensities = intensities[mask]
    if region_wavelengths.size < 3:
        raise ValueError("The selected wavelength interval must contain at least 3 points.")
    if not np.all(np.diff(region_wavelengths) > 0):
        raise ValueError("Wavelength values must be strictly increasing.")
    return region_wavelengths, region_intensities


def detect_wavelet_peaks(
    wavelengths,
    intensities,
    wavelet: str = DEFAULT_WAVELET,
    scales=None,
    neighbor: int = DEFAULT_NEIGHBOR,
    min_length: int = DEFAULT_MIN_LENGTH,
    coefficient_threshold: float = DEFAULT_COEFFICIENT_THRESHOLD,
    correction_window: int = DEFAULT_CORRECTION_WINDOW,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the project's wavelet-ridge detector and return unique sorted peaks."""
    wavelengths = np.asarray(wavelengths, dtype=float)
    intensities = np.asarray(intensities, dtype=float)
    if wavelengths.ndim != 1 or intensities.ndim != 1:
        raise ValueError("wavelengths and intensities must be one-dimensional.")
    if wavelengths.size != intensities.size or wavelengths.size < 3:
        raise ValueError("Spectrum arrays must have equal length and at least 3 points.")
    if not np.isfinite(wavelengths).all() or not np.isfinite(intensities).all():
        raise ValueError("Spectrum arrays must contain only finite values.")

    selected_scales = DEFAULT_SCALES if scales is None else np.asarray(scales)
    raw_indices, _, _ = wavelet_peak_detection(
        intensities,
        wavelengths,
        wavelet=wavelet,
        scales=selected_scales,
        neighbor=neighbor,
        min_length=min_length,
        coeffi_threshold=coefficient_threshold,
        window=correction_window,
    )
    peak_indices = np.unique(np.asarray(raw_indices, dtype=int))
    return (
        peak_indices,
        wavelengths[peak_indices],
        intensities[peak_indices],
    )


def plot_wavelet_peaks(
    wavelengths,
    intensities,
    peak_indices,
    output_path: Union[str, Path, None] = DEFAULT_OUTPUT_PATH,
    wavelength_min: float = DEFAULT_WAVELENGTH_MIN,
    wavelength_max: float = DEFAULT_WAVELENGTH_MAX,
    show: bool = False,
):
    """Plot the Figure 1 spectrum and mark all detected wavelet peaks."""
    wavelengths = np.asarray(wavelengths, dtype=float)
    intensities = np.asarray(intensities, dtype=float)
    peak_indices = np.asarray(peak_indices, dtype=int).reshape(-1)
    if wavelengths.ndim != 1 or intensities.ndim != 1:
        raise ValueError("wavelengths and intensities must be one-dimensional.")
    if wavelengths.size != intensities.size or wavelengths.size == 0:
        raise ValueError("Spectrum arrays must be non-empty and have equal length.")
    if np.any(peak_indices < 0) or np.any(peak_indices >= wavelengths.size):
        raise ValueError("peak_indices contains an out-of-range index.")

    fig, ax = plt.subplots(figsize=FIGSIZE, facecolor="white")
    ax.plot(
        wavelengths,
        intensities,
        color=SPECTRUM_COLOR,
        linewidth=SPECTRUM_LINEWIDTH,
        alpha=SPECTRUM_ALPHA,
        label="Original Spectrum",
        zorder=2,
    )
    ax.scatter(
        wavelengths[peak_indices],
        intensities[peak_indices],
        color=PEAK_COLOR,
        s=PEAK_MARKER_SIZE,
        label="Detected Peaks",
        zorder=4,
    )

    ax.set_xlim(wavelength_min, wavelength_max)
    ax.set_ylim(
        *figure1_reference_y_limits(
            wavelengths,
            intensities,
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
        )
    )
    _SOURCE.apply_spectrum_style(ax, show_axes=True)
    ax.legend(
        loc=LEGEND_LOCATION,
        frameon=False,
        prop={
            "size": LEGEND_FONTSIZE,
            "weight": _SOURCE.FONT_WEIGHT,
        },
    )

    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            str(output_path),
            dpi=SAVE_DPI,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
            format="png",
        )
    if show:
        plt.show()
    return fig, ax


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Two-column CSV containing wavelength,intensity.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--wavelength-min",
        type=float,
        default=DEFAULT_WAVELENGTH_MIN,
        help="Lower wavelength limit in nm.",
    )
    parser.add_argument(
        "--wavelength-max",
        type=float,
        default=DEFAULT_WAVELENGTH_MAX,
        help="Upper wavelength limit in nm.",
    )
    parser.add_argument("--wavelet", default=DEFAULT_WAVELET)
    parser.add_argument("--scale-min", type=int, default=int(DEFAULT_SCALES.min()))
    parser.add_argument("--scale-max", type=int, default=int(DEFAULT_SCALES.max()))
    parser.add_argument("--neighbor", type=int, default=DEFAULT_NEIGHBOR)
    parser.add_argument("--min-length", type=int, default=DEFAULT_MIN_LENGTH)
    parser.add_argument(
        "--coefficient-threshold",
        type=float,
        default=DEFAULT_COEFFICIENT_THRESHOLD,
    )
    parser.add_argument(
        "--correction-window",
        type=int,
        default=DEFAULT_CORRECTION_WINDOW,
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if args.scale_min < 1 or args.scale_min > args.scale_max:
        parser.error("scale-min must be positive and no larger than scale-max")

    wavelengths, intensities = load_spectrum_region(
        args.data,
        wavelength_min=args.wavelength_min,
        wavelength_max=args.wavelength_max,
    )
    peak_indices, peak_wavelengths, _ = detect_wavelet_peaks(
        wavelengths,
        intensities,
        wavelet=args.wavelet,
        scales=np.arange(args.scale_min, args.scale_max + 1),
        neighbor=args.neighbor,
        min_length=args.min_length,
        coefficient_threshold=args.coefficient_threshold,
        correction_window=args.correction_window,
    )
    fig, _ = plot_wavelet_peaks(
        wavelengths,
        intensities,
        peak_indices,
        output_path=args.output,
        wavelength_min=args.wavelength_min,
        wavelength_max=args.wavelength_max,
        show=args.show,
    )
    plt.close(fig)

    print("Saved wavelet peak figure to: {}".format(args.output))
    print("Detected peaks: {}".format(peak_indices.size))
    print(
        "Peak wavelengths (nm): "
        + ", ".join("{:.2f}".format(value) for value in peak_wavelengths)
    )


if __name__ == "__main__":
    main()
