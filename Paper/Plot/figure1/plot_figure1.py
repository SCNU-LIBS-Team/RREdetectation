"""Plot and annotate a selected LIBS spectrum region.

Run from this directory with::

    python plot_figure1.py

The script reads ``data3.csv``, uses the manually configured peak
wavelengths below, snaps them to the nearest detected local maxima, and writes
``figure1_spectrum.png``. It also draws matching short vertical lines above the
spectrum.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


FIGURE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = FIGURE_DIR / "data3.csv"
DEFAULT_OUTPUT_PATH = FIGURE_DIR / "figure1_spectrum.png"

# Values mirror the current Figure 1 plotting style and the Figure 3 reference
# image (approximately 1106 x 577 px, aspect ratio 1.917).
FIGSIZE = (9.6, 5.0)
FONT_WEIGHT = "semibold"
TICK_LABELSIZE = 12
LABEL_FONTSIZE = 15
TITLE_FONTSIZE = 15
SPINE_WIDTH = 1.8
TICK_LENGTH = 6
TICK_WIDTH = 2.0
SPECTRUM_LINEWIDTH = 1.2
SPECTRUM_COLOR = "gray"
DEFAULT_SPECTRUM_ALPHA = 0.5
SAVE_DPI = 1200
DEFAULT_WAVELENGTH_MIN = 220.0
DEFAULT_WAVELENGTH_MAX = 330.0
DEFAULT_BOUNDARY_MARGIN_NM = 10.0

# Peak-detection and annotation defaults.
DEFAULT_PEAK_COUNT = 6
DEFAULT_PEAK_PROMINENCE_RATIO = 0.01
DEFAULT_PEAK_MIN_DISTANCE_NM = 0.5
DEFAULT_MANUAL_PEAK_WAVELENGTHS = [
    234.42,
    249.01,
    259.90,
    274.83,
    282.84,
    302.07,
]
DEFAULT_RANDOM_SEED = 42
DEFAULT_PEAK_MARKER_COLOR = "#A4C686"
DEFAULT_PEAK_MARKER_SIZE = 20.0
DEFAULT_VERTICAL_LINE_COLOR = "#B2D990"
DEFAULT_VERTICAL_LINE_WIDTH = 3
DEFAULT_VERTICAL_LINE_ALPHA = 1
DEFAULT_TOP_LINE_COLOR = "#897CD3"
DEFAULT_TOP_LINE_RANDOM_SEED = 42
DEFAULT_TOP_LINE_X_JITTER_NM = 1.0
DEFAULT_TOP_LINE_MIN_GAP_NM = 2.0
DEFAULT_TOP_LINE_MARGIN_RATIO = 0.35
DEFAULT_TOP_LINE_PADDING_RATIO = 0.05
DEFAULT_HORIZONTAL_LINE_EXTENSION_NM = 5.0
DEFAULT_HORIZONTAL_LINE_ZORDER = 5
    

def load_spectrum_data(
    csv_path: Union[str, Path] = DEFAULT_DATA_PATH,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load wavelength and intensity columns from a two-column CSV file."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Spectrum data file not found: {csv_path}")

    try:
        data = np.genfromtxt(
            csv_path,
            delimiter=",",
            names=True,
            dtype=float,
            encoding="utf-8-sig",
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read spectrum CSV: {csv_path}") from exc

    field_names = data.dtype.names
    if field_names is None or set(field_names) != {"wavelength", "intensity"}:
        raise ValueError(
            "CSV must contain exactly the columns 'wavelength' and 'intensity'. "
            f"Found: {field_names}"
        )

    wavelengths = np.atleast_1d(np.asarray(data["wavelength"], dtype=float))
    intensities = np.atleast_1d(np.asarray(data["intensity"], dtype=float))
    _validate_spectrum_arrays(wavelengths, intensities)
    return wavelengths, intensities


def _validate_spectrum_arrays(
    wavelengths: np.ndarray,
    intensities: np.ndarray,
) -> None:
    """Validate the arrays before sending them to Matplotlib or SciPy."""
    if wavelengths.ndim != 1 or intensities.ndim != 1:
        raise ValueError("wavelengths and intensities must be one-dimensional arrays.")
    if wavelengths.size == 0:
        raise ValueError("Spectrum data must contain at least one row.")
    if wavelengths.size != intensities.size:
        raise ValueError(
            "wavelengths and intensities must have the same length. "
            f"Got {wavelengths.size} and {intensities.size}."
        )
    if not np.isfinite(wavelengths).all() or not np.isfinite(intensities).all():
        raise ValueError("Spectrum data must contain only finite numeric values.")


def _validate_wavelength_limits(
    wavelength_min: float,
    wavelength_max: float,
) -> None:
    """Validate the wavelength interval shown in the figure."""
    if not np.isfinite(wavelength_min) or not np.isfinite(wavelength_max):
        raise ValueError("Wavelength limits must be finite numbers.")
    if wavelength_min >= wavelength_max:
        raise ValueError(
            "wavelength_min must be smaller than wavelength_max. "
            f"Got {wavelength_min} and {wavelength_max}."
        )


def _validate_horizontal_line_extension(horizontal_extension_nm: float) -> None:
    """Validate the extra horizontal span beyond the outer vertical lines."""
    if not np.isfinite(horizontal_extension_nm) or horizontal_extension_nm < 0:
        raise ValueError(
            "horizontal_extension_nm must be a finite non-negative number."
        )


def find_local_peak_indices(
    wavelengths,
    intensities,
    wavelength_min: float = DEFAULT_WAVELENGTH_MIN,
    wavelength_max: float = DEFAULT_WAVELENGTH_MAX,
    prominence_ratio: Optional[float] = DEFAULT_PEAK_PROMINENCE_RATIO,
    min_distance_nm: float = DEFAULT_PEAK_MIN_DISTANCE_NM,
) -> np.ndarray:
    """Find local intensity maxima inside the selected wavelength interval.

    ``prominence_ratio`` is relative to the intensity range in the selected
    interval. ``min_distance_nm`` prevents multiple nearby points from being
    treated as separate peaks.
    """
    wavelengths = np.asarray(wavelengths, dtype=float)
    intensities = np.asarray(intensities, dtype=float)
    _validate_spectrum_arrays(wavelengths, intensities)
    _validate_wavelength_limits(wavelength_min, wavelength_max)

    if prominence_ratio is not None:
        if not np.isfinite(prominence_ratio) or prominence_ratio < 0:
            raise ValueError("prominence_ratio must be a finite non-negative number.")
    if not np.isfinite(min_distance_nm) or min_distance_nm < 0:
        raise ValueError("min_distance_nm must be a finite non-negative number.")

    region_indices = np.flatnonzero(
        (wavelengths >= wavelength_min) & (wavelengths <= wavelength_max)
    )
    if region_indices.size < 3:
        raise ValueError("The selected wavelength interval must contain at least 3 points.")

    region_wavelengths = wavelengths[region_indices]
    region_intensities = intensities[region_indices]
    if not np.all(np.diff(region_wavelengths) > 0):
        raise ValueError("Wavelength values must be strictly increasing in the selected interval.")

    point_spacing = float(np.median(np.diff(region_wavelengths)))
    distance_points = max(1, int(np.ceil(min_distance_nm / point_spacing)))
    prominence = None
    if prominence_ratio is not None:
        prominence = prominence_ratio * float(np.ptp(region_intensities))

    peak_positions, _ = find_peaks(
        region_intensities,
        prominence=prominence,
        distance=distance_points,
    )
    return region_indices[peak_positions]


def select_evenly_distributed_peak_indices(
    wavelengths,
    candidate_indices: Sequence[int],
    wavelength_min: float,
    wavelength_max: float,
    boundary_margin_nm: float = DEFAULT_BOUNDARY_MARGIN_NM,
    count: int = DEFAULT_PEAK_COUNT,
    random_seed: Optional[int] = DEFAULT_RANDOM_SEED,
) -> np.ndarray:
    """Randomly select peaks away from the interval boundaries.

    The first and last ``boundary_margin_nm`` of the displayed interval are
    excluded from selection. The remaining interval is split into equal-width
    bins, with one candidate randomly selected from each bin. If a bin has no
    candidate, a farthest-from-selected fallback fills the remaining slots.
    """
    wavelengths = np.asarray(wavelengths, dtype=float)
    _validate_wavelength_limits(wavelength_min, wavelength_max)
    if not np.isfinite(boundary_margin_nm) or boundary_margin_nm < 0:
        raise ValueError("boundary_margin_nm must be a finite non-negative number.")
    selection_min = wavelength_min + boundary_margin_nm
    selection_max = wavelength_max - boundary_margin_nm
    _validate_wavelength_limits(selection_min, selection_max)
    if wavelengths.ndim != 1 or not np.isfinite(wavelengths).all():
        raise ValueError("wavelengths must be a one-dimensional finite array.")
    if not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("count must be a positive integer.")

    candidates = np.asarray(candidate_indices, dtype=int).reshape(-1)
    candidates = np.unique(candidates)
    if candidates.size < count:
        raise ValueError(
            f"Only {candidates.size} candidate peaks are available; "
            f"cannot select {count}."
        )
    if np.any(candidates < 0) or np.any(candidates >= wavelengths.size):
        raise ValueError("candidate_indices contains an out-of-range index.")

    candidates = candidates[
        (wavelengths[candidates] >= selection_min)
        & (wavelengths[candidates] <= selection_max)
    ]
    if candidates.size < count:
        raise ValueError(
            f"Only {candidates.size} candidate peaks are inside the selected interval; "
            f"cannot select {count}."
        )

    rng = np.random.default_rng(random_seed)
    bin_edges = np.linspace(selection_min, selection_max, count + 1)
    selected = []
    remaining = candidates.tolist()

    for bin_number in range(count):
        left = bin_edges[bin_number]
        right = bin_edges[bin_number + 1]
        in_bin = [
            index
            for index in remaining
            if wavelengths[index] >= left
            and (wavelengths[index] <= right if bin_number == count - 1 else wavelengths[index] < right)
        ]
        if in_bin:
            chosen = int(rng.choice(in_bin))
            selected.append(chosen)
            remaining.remove(chosen)

    while len(selected) < count:
        if not remaining:
            raise ValueError("Not enough unused candidate peaks to complete the selection.")
        if not selected:
            chosen = int(rng.choice(remaining))
        else:
            available = np.asarray(remaining, dtype=int)
            selected_wavelengths = wavelengths[np.asarray(selected, dtype=int)]
            distances = np.min(
                np.abs(wavelengths[available, None] - selected_wavelengths[None, :]),
                axis=1,
            )
            largest_distance = distances.max()
            farthest = available[np.isclose(distances, largest_distance)]
            chosen = int(rng.choice(farthest))
        selected.append(chosen)
        remaining.remove(chosen)

    selected.sort(key=lambda index: wavelengths[index])
    return np.asarray(selected, dtype=int)


def select_manual_peak_indices(
    wavelengths,
    intensities,
    manual_peak_wavelengths: Sequence[float] = DEFAULT_MANUAL_PEAK_WAVELENGTHS,
    wavelength_min: float = DEFAULT_WAVELENGTH_MIN,
    wavelength_max: float = DEFAULT_WAVELENGTH_MAX,
    boundary_margin_nm: float = DEFAULT_BOUNDARY_MARGIN_NM,
    prominence_ratio: Optional[float] = DEFAULT_PEAK_PROMINENCE_RATIO,
    min_distance_nm: float = DEFAULT_PEAK_MIN_DISTANCE_NM,
) -> np.ndarray:
    """Map manually entered wavelengths to nearby local peak indices."""
    wavelengths = np.asarray(wavelengths, dtype=float)
    intensities = np.asarray(intensities, dtype=float)
    _validate_spectrum_arrays(wavelengths, intensities)
    _validate_wavelength_limits(wavelength_min, wavelength_max)

    if not np.isfinite(boundary_margin_nm) or boundary_margin_nm < 0:
        raise ValueError("boundary_margin_nm must be a finite non-negative number.")
    selection_min = wavelength_min + boundary_margin_nm
    selection_max = wavelength_max - boundary_margin_nm
    _validate_wavelength_limits(selection_min, selection_max)

    requested = np.asarray(manual_peak_wavelengths, dtype=float).reshape(-1)
    if requested.size != DEFAULT_PEAK_COUNT:
        raise ValueError(
            f"Please enter exactly {DEFAULT_PEAK_COUNT} manual peak wavelengths; "
            f"got {requested.size}."
        )
    if not np.isfinite(requested).all():
        raise ValueError("Manual peak wavelengths must be finite numbers.")
    if np.any(requested < selection_min) or np.any(requested > selection_max):
        raise ValueError(
            "Manual peak wavelengths must stay inside the allowed selection range "
            f"{selection_min:g}-{selection_max:g} nm."
        )

    candidate_indices = find_local_peak_indices(
        wavelengths,
        intensities,
        wavelength_min=selection_min,
        wavelength_max=selection_max,
        prominence_ratio=prominence_ratio,
        min_distance_nm=min_distance_nm,
    )
    if candidate_indices.size < requested.size:
        raise ValueError(
            f"Only {candidate_indices.size} local peaks are available in the allowed "
            f"range; cannot map {requested.size} manual wavelengths."
        )

    selected = []
    remaining = candidate_indices.tolist()
    for target_wavelength in requested:
        available = np.asarray(remaining, dtype=int)
        nearest_position = np.argmin(
            np.abs(wavelengths[available] - target_wavelength)
        )
        chosen = int(available[nearest_position])
        selected.append(chosen)
        remaining.remove(chosen)

    selected.sort(key=lambda index: wavelengths[index])
    return np.asarray(selected, dtype=int)


def add_peak_annotations(
    ax,
    wavelengths,
    intensities,
    peak_indices: Optional[Sequence[int]],
    vertical_line_color: str = DEFAULT_VERTICAL_LINE_COLOR,
    peak_marker_color: str = DEFAULT_PEAK_MARKER_COLOR,
    peak_marker_size: float = DEFAULT_PEAK_MARKER_SIZE,
    vertical_line_width: float = DEFAULT_VERTICAL_LINE_WIDTH,
    vertical_line_alpha: float = DEFAULT_VERTICAL_LINE_ALPHA,
    horizontal_extension_nm: float = DEFAULT_HORIZONTAL_LINE_EXTENSION_NM,
):
    """Mark selected peaks and draw each corresponding line down to ``y=0``."""
    if peak_indices is None:
        return ax

    wavelengths = np.asarray(wavelengths, dtype=float)
    intensities = np.asarray(intensities, dtype=float)
    _validate_spectrum_arrays(wavelengths, intensities)
    peak_indices = np.asarray(peak_indices, dtype=int).reshape(-1)
    if peak_indices.size == 0:
        return ax
    if np.any(peak_indices < 0) or np.any(peak_indices >= wavelengths.size):
        raise ValueError("peak_indices contains an out-of-range index.")
    _validate_horizontal_line_extension(horizontal_extension_nm)

    peak_wavelengths = wavelengths[peak_indices]
    peak_intensities = intensities[peak_indices]
    ax.scatter(
        peak_wavelengths,
        peak_intensities,
        color=peak_marker_color,
        s=peak_marker_size,
        zorder=4,
    )
    for wavelength, intensity in zip(peak_wavelengths, peak_intensities):
        ax.vlines(
            wavelength,
            0,
            intensity,
            colors=vertical_line_color,
            linewidth=vertical_line_width,
            alpha=vertical_line_alpha,
            zorder=1,
        )
    if peak_wavelengths.size > 1:
        ax.hlines(
            0,
            float(np.min(peak_wavelengths)) - horizontal_extension_nm,
            float(np.max(peak_wavelengths)) + horizontal_extension_nm,
            colors=vertical_line_color,
            linewidth=vertical_line_width,
            alpha=vertical_line_alpha,
            zorder=DEFAULT_HORIZONTAL_LINE_ZORDER,
        )
    return ax


def add_top_symmetric_lines(
    ax,
    wavelengths,
    intensities,
    peak_indices: Optional[Sequence[int]],
    wavelength_min: float = DEFAULT_WAVELENGTH_MIN,
    wavelength_max: float = DEFAULT_WAVELENGTH_MAX,
    line_color: str = DEFAULT_TOP_LINE_COLOR,
    line_width: float = DEFAULT_VERTICAL_LINE_WIDTH,
    line_alpha: float = DEFAULT_VERTICAL_LINE_ALPHA,
    random_seed: Optional[int] = DEFAULT_TOP_LINE_RANDOM_SEED,
    x_jitter_nm: float = DEFAULT_TOP_LINE_X_JITTER_NM,
    min_gap_nm: float = DEFAULT_TOP_LINE_MIN_GAP_NM,
    top_margin_ratio: float = DEFAULT_TOP_LINE_MARGIN_RATIO,
    top_padding_ratio: float = DEFAULT_TOP_LINE_PADDING_RATIO,
    horizontal_extension_nm: float = DEFAULT_HORIZONTAL_LINE_EXTENSION_NM,
    peak_marker_size: float = DEFAULT_PEAK_MARKER_SIZE,
):
    """Draw top lines with matching lengths and small random x offsets.

    Each top line has exactly the same data-coordinate length as its
    corresponding peak-to-zero line below the spectrum. Only the horizontal
    position is randomly offset, while a larger y-axis limit provides enough
    room for the upper set of lines.
    """
    if peak_indices is None:
        return ax

    wavelengths = np.asarray(wavelengths, dtype=float)
    intensities = np.asarray(intensities, dtype=float)
    _validate_spectrum_arrays(wavelengths, intensities)
    _validate_wavelength_limits(wavelength_min, wavelength_max)
    peak_indices = np.asarray(peak_indices, dtype=int).reshape(-1)
    if peak_indices.size == 0:
        return ax
    if np.any(peak_indices < 0) or np.any(peak_indices >= wavelengths.size):
        raise ValueError("peak_indices contains an out-of-range index.")
    if not np.isfinite(x_jitter_nm) or x_jitter_nm < 0:
        raise ValueError("x_jitter_nm must be a finite non-negative number.")
    if not np.isfinite(min_gap_nm) or min_gap_nm < 0:
        raise ValueError("min_gap_nm must be a finite non-negative number.")
    if not np.isfinite(top_margin_ratio) or top_margin_ratio <= 0:
        raise ValueError("top_margin_ratio must be a positive finite number.")
    if not np.isfinite(top_padding_ratio) or top_padding_ratio < 0:
        raise ValueError("top_padding_ratio must be a finite non-negative number.")
    if not np.isfinite(line_alpha) or not 0 <= line_alpha <= 1:
        raise ValueError("line_alpha must be between 0 and 1.")
    _validate_horizontal_line_extension(horizontal_extension_nm)

    region_mask = (wavelengths >= wavelength_min) & (wavelengths <= wavelength_max)
    regional_intensities = intensities[region_mask]
    if regional_intensities.size == 0:
        raise ValueError("The selected wavelength interval contains no data points.")
    regional_max = float(np.max(regional_intensities))
    if regional_max <= 0:
        regional_max = 1.0

    peak_order = np.argsort(wavelengths[peak_indices])
    base_wavelengths = wavelengths[peak_indices][peak_order]
    peak_intensities = intensities[peak_indices][peak_order]

    rng = np.random.default_rng(random_seed)
    positions = None
    for _ in range(100):
        trial = base_wavelengths + rng.uniform(
            -x_jitter_nm,
            x_jitter_nm,
            size=base_wavelengths.size,
        )
        if (
            np.all(trial >= wavelength_min)
            and np.all(trial <= wavelength_max)
            and np.all(np.diff(trial) >= min_gap_nm)
        ):
            positions = trial
            break

    if positions is None:
        positions = np.clip(base_wavelengths, wavelength_min, wavelength_max)
        for index in range(1, positions.size):
            positions[index] = max(positions[index], positions[index - 1] + min_gap_nm)
        if positions[-1] > wavelength_max:
            positions -= positions[-1] - wavelength_max
        if positions[0] < wavelength_min:
            positions += wavelength_min - positions[0]
        if np.any(np.diff(positions) < min_gap_nm):
            raise ValueError("Could not keep the top lines separated by min_gap_nm.")

    top_base_y = regional_max * (1 + top_margin_ratio)
    top_line_top_y = top_base_y + float(np.max(peak_intensities))
    ax.set_ylim(bottom=0, top=top_line_top_y * (1 + top_padding_ratio))

    for position, intensity in zip(positions, peak_intensities):
        ax.vlines(
            position,
            top_line_top_y - intensity,
            top_line_top_y,
            colors=line_color,
            linewidth=line_width,
            alpha=line_alpha,
            zorder=1,
        )
    ax.scatter(
        positions,
        top_line_top_y - peak_intensities,
        color=line_color,
        s=peak_marker_size,
        zorder=6,
    )
    if positions.size > 1:
        ax.hlines(
            top_line_top_y,
            float(np.min(positions)) - horizontal_extension_nm,
            float(np.max(positions)) + horizontal_extension_nm,
            colors=line_color,
            linewidth=line_width,
            alpha=line_alpha,
            zorder=DEFAULT_HORIZONTAL_LINE_ZORDER,
        )
    return ax


def apply_spectrum_style(ax, show_axes: bool = True):
    """Apply the spectrum style, optionally hiding all axes for the article figure."""
    ax.set_facecolor("white")
    ax.grid(False)

    if not show_axes:
        ax.set_axis_off()
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labeltop=False,
            labelleft=False,
            labelright=False,
        )
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        for axis in (ax.xaxis, ax.yaxis):
            for tick in axis.get_major_ticks():
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
                tick.label1.set_visible(False)
                tick.label2.set_visible(False)
        return ax

    ax.set_axis_on()
    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        length=TICK_LENGTH,
        width=TICK_WIDTH,
        labelsize=TICK_LABELSIZE,
        colors="black",
    )

    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_WIDTH)
        spine.set_color("black")

    for tick_label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        tick_label.set_fontweight(FONT_WEIGHT)
        tick_label.set_color("black")

    ax.set_xlabel(
        "Wavelength (nm)",
        fontsize=LABEL_FONTSIZE,
        fontweight=FONT_WEIGHT,
        color="black",
    )
    ax.set_ylabel(
        "Intensity",
        fontsize=LABEL_FONTSIZE,
        fontweight=FONT_WEIGHT,
        color="black",
    )
    ax.set_title("")
    return ax


def plot_spectrum(
    wavelengths,
    intensities,
    output_path: Optional[Union[str, Path]] = DEFAULT_OUTPUT_PATH,
    wavelength_min: float = DEFAULT_WAVELENGTH_MIN,
    wavelength_max: float = DEFAULT_WAVELENGTH_MAX,
    peak_indices: Optional[Sequence[int]] = None,
    vertical_line_color: str = DEFAULT_VERTICAL_LINE_COLOR,
    top_line_color: str = DEFAULT_TOP_LINE_COLOR,
    spectrum_alpha: float = DEFAULT_SPECTRUM_ALPHA,
    horizontal_extension_nm: float = DEFAULT_HORIZONTAL_LINE_EXTENSION_NM,
    show_top_lines: bool = True,
    show_axes: bool = True,
    show: bool = False,
):
    """Plot one spectrum region and optional selected-peak annotations."""
    wavelengths = np.asarray(wavelengths, dtype=float)
    intensities = np.asarray(intensities, dtype=float)
    _validate_spectrum_arrays(wavelengths, intensities)
    _validate_wavelength_limits(wavelength_min, wavelength_max)
    if not np.isfinite(spectrum_alpha) or not 0 <= spectrum_alpha <= 1:
        raise ValueError("spectrum_alpha must be between 0 and 1.")

    fig, ax = plt.subplots(figsize=FIGSIZE, facecolor="white")
    ax.plot(
        wavelengths,
        intensities,
        color=SPECTRUM_COLOR,
        linewidth=SPECTRUM_LINEWIDTH,
        alpha=spectrum_alpha,
        label="Original Spectrum",
        zorder=2,
    )
    ax.set_xlim(wavelength_min, wavelength_max)
    add_peak_annotations(
        ax,
        wavelengths,
        intensities,
        peak_indices,
        vertical_line_color=vertical_line_color,
        horizontal_extension_nm=horizontal_extension_nm,
    )
    if show_top_lines:
        add_top_symmetric_lines(
            ax,
            wavelengths,
            intensities,
            peak_indices,
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
            line_color=top_line_color,
            horizontal_extension_nm=horizontal_extension_nm,
        )
    apply_spectrum_style(ax, show_axes=show_axes)

    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_path,
            dpi=SAVE_DPI,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
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
        help="CSV file containing wavelength,intensity columns.",
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
        help="Left wavelength limit in nm (default: 220).",
    )
    parser.add_argument(
        "--wavelength-max",
        type=float,
        default=DEFAULT_WAVELENGTH_MAX,
        help="Right wavelength limit in nm (default: 330).",
    )
    parser.add_argument(
        "--boundary-margin-nm",
        type=float,
        default=DEFAULT_BOUNDARY_MARGIN_NM,
        help="Exclude this many nm at both ends of the display range (default: 10).",
    )
    parser.add_argument(
        "--peak-prominence-ratio",
        type=float,
        default=DEFAULT_PEAK_PROMINENCE_RATIO,
        help="Minimum peak prominence divided by regional intensity range.",
    )
    parser.add_argument(
        "--peak-distance-nm",
        type=float,
        default=DEFAULT_PEAK_MIN_DISTANCE_NM,
        help="Minimum distance between candidate peaks in nm.",
    )
    parser.add_argument(
        "--line-color",
        default=DEFAULT_VERTICAL_LINE_COLOR,
        help="Color name or hex code for the vertical peak lines.",
    )
    parser.add_argument(
        "--top-line-color",
        default=DEFAULT_TOP_LINE_COLOR,
        help="Color name or hex code for the upper symmetric lines.",
    )
    parser.add_argument(
        "--horizontal-extension-nm",
        type=float,
        default=DEFAULT_HORIZONTAL_LINE_EXTENSION_NM,
        help="Extra horizontal span beyond each outer vertical line in nm.",
    )
    parser.add_argument(
        "--spectrum-alpha",
        type=float,
        default=DEFAULT_SPECTRUM_ALPHA,
        help="Opacity of the original spectrum line, from 0 to 1.",
    )
    parser.add_argument(
        "--hide-top-lines",
        action="store_true",
        help="Do not draw the randomly offset top symmetric lines.",
    )
    axes_group = parser.add_mutually_exclusive_group()
    axes_group.add_argument(
        "--show-axes",
        dest="show_axes",
        action="store_true",
        help="Show axes, labels, ticks, and spines.",
    )
    axes_group.add_argument(
        "--hide-axes",
        dest="show_axes",
        action="store_false",
        help="Hide axes, labels, ticks, and spines.",
    )
    parser.set_defaults(show_axes=True)
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure after saving it.",
    )
    args = parser.parse_args()

    wavelengths, intensities = load_spectrum_data(args.data)
    peak_indices = select_manual_peak_indices(
        wavelengths,
        intensities,
        DEFAULT_MANUAL_PEAK_WAVELENGTHS,
        wavelength_min=args.wavelength_min,
        wavelength_max=args.wavelength_max,
        boundary_margin_nm=args.boundary_margin_nm,
        prominence_ratio=args.peak_prominence_ratio,
        min_distance_nm=args.peak_distance_nm,
    )
    fig, _ = plot_spectrum(
        wavelengths,
        intensities,
        output_path=args.output,
        wavelength_min=args.wavelength_min,
        wavelength_max=args.wavelength_max,
        peak_indices=peak_indices,
        vertical_line_color=args.line_color,
        top_line_color=args.top_line_color,
        spectrum_alpha=args.spectrum_alpha,
        horizontal_extension_nm=args.horizontal_extension_nm,
        show_top_lines=not args.hide_top_lines,
        show_axes=args.show_axes,
        show=args.show,
    )
    plt.close(fig)
    print(f"Saved spectrum figure to: {args.output}")
    print(
        f"Data points: {wavelengths.size}; "
        f"display range: {args.wavelength_min:g}-{args.wavelength_max:g} nm; "
        f"selection range: {args.wavelength_min + args.boundary_margin_nm:g}-"
        f"{args.wavelength_max - args.boundary_margin_nm:g} nm; "
        f"manual peaks: {len(DEFAULT_MANUAL_PEAK_WAVELENGTHS)}; "
        f"selected peaks: {peak_indices.size}"
    )
    print(
        "Selected peak wavelengths (nm): "
        + ", ".join(f"{wavelengths[index]:.2f}" for index in peak_indices)
    )
    print(f"Vertical line color: {args.line_color}")
    print(f"Top line color: {args.top_line_color}")
    print(f"Horizontal line extension: {args.horizontal_extension_nm:g} nm")


if __name__ == "__main__":
    main()
