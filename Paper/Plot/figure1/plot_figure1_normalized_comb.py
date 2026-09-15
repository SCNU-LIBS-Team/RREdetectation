"""Draw the normalized comb inset for Figure 1.

The inset reuses the current Figure 1 data-selection settings without changing
``plot_figure1.py`` or the source CSV. It draws only the upper theoretical
comb and lower experimental comb, with each comb normalized independently.
"""

from __future__ import annotations

import argparse
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np


FIGURE_DIR = Path(__file__).resolve().parent
SOURCE_SCRIPT_PATH = FIGURE_DIR / "plot_figure1.py"
DEFAULT_DATA_PATH = FIGURE_DIR / "data3.csv"
DEFAULT_OUTPUT_PATH = FIGURE_DIR / "figure1_normalized_comb.png"

FIGSIZE = (4, 4)
SAVE_DPI = 1200
TOP_LABEL = "Theoratical normalized comb"
BOTTOM_LABEL = "Expentional normalized comb"

DEFAULT_COMB_JOINSTYLE = "bevel"
DEFAULT_COMB_CAPSTYLE = "butt"
DEFAULT_COMB_MIDDLE_LINE_ZORDER = 2
DEFAULT_COMB_ENDPOINT_PATH_ZORDER = 3

# These values are intentionally read from the current Figure 1 script below.
# They are not independently redefined, so changes made in plot_figure1.py are
# reflected in this inset automatically.


def _load_current_figure1_module():
    """Load the existing Figure 1 module without modifying it."""
    if not SOURCE_SCRIPT_PATH.is_file():
        raise FileNotFoundError(f"Figure 1 script not found: {SOURCE_SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location(
        "_figure1_source_for_normalized_comb",
        SOURCE_SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Figure 1 script: {SOURCE_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SOURCE = _load_current_figure1_module()

DEFAULT_TOP_LINE_COLOR = _SOURCE.DEFAULT_TOP_LINE_COLOR
DEFAULT_VERTICAL_LINE_COLOR = _SOURCE.DEFAULT_VERTICAL_LINE_COLOR
DEFAULT_VERTICAL_LINE_WIDTH = _SOURCE.DEFAULT_VERTICAL_LINE_WIDTH
DEFAULT_VERTICAL_LINE_ALPHA = _SOURCE.DEFAULT_VERTICAL_LINE_ALPHA
DEFAULT_TOP_LINE_RANDOM_SEED = _SOURCE.DEFAULT_TOP_LINE_RANDOM_SEED
DEFAULT_TOP_LINE_X_JITTER_NM = _SOURCE.DEFAULT_TOP_LINE_X_JITTER_NM
DEFAULT_TOP_LINE_MIN_GAP_NM = _SOURCE.DEFAULT_TOP_LINE_MIN_GAP_NM
DEFAULT_WAVELENGTH_MIN = _SOURCE.DEFAULT_WAVELENGTH_MIN
DEFAULT_WAVELENGTH_MAX = _SOURCE.DEFAULT_WAVELENGTH_MAX
DEFAULT_BOUNDARY_MARGIN_NM = _SOURCE.DEFAULT_BOUNDARY_MARGIN_NM
DEFAULT_PEAK_PROMINENCE_RATIO = _SOURCE.DEFAULT_PEAK_PROMINENCE_RATIO
DEFAULT_PEAK_MIN_DISTANCE_NM = _SOURCE.DEFAULT_PEAK_MIN_DISTANCE_NM
DEFAULT_MANUAL_PEAK_WAVELENGTHS = tuple(_SOURCE.DEFAULT_MANUAL_PEAK_WAVELENGTHS)
DEFAULT_PEAK_COUNT = _SOURCE.DEFAULT_PEAK_COUNT
DEFAULT_BORDER_COLOR = "black"
DEFAULT_BORDER_LINEWIDTH = _SOURCE.SPINE_WIDTH
DEFAULT_BORDER_ZORDER = 1000

LOWER_BASE_Y = 0.0
UPPER_TOP_Y = 2.0
Y_LIMITS = (-0.35, 2.35)


@dataclass(frozen=True)
class NormalizedCombGeometry:
    """Data-coordinate geometry used by the normalized comb inset."""

    lower_positions: np.ndarray
    upper_positions: np.ndarray
    lower_lengths: np.ndarray
    upper_lengths: np.ndarray
    lower_base_y: float
    upper_top_y: float
    lower_horizontal_limits: np.ndarray
    upper_horizontal_limits: np.ndarray


def load_spectrum_data(
    csv_path: Union[str, Path] = DEFAULT_DATA_PATH,
) -> Tuple[np.ndarray, np.ndarray]:
    """Read spectrum data through the current Figure 1 implementation."""
    return _SOURCE.load_spectrum_data(csv_path)


def load_current_selection(
    data_path: Union[str, Path] = DEFAULT_DATA_PATH,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load data and select the manually configured current Figure 1 peaks."""
    wavelengths, intensities = load_spectrum_data(data_path)
    peak_indices = _SOURCE.select_manual_peak_indices(
        wavelengths,
        intensities,
        DEFAULT_MANUAL_PEAK_WAVELENGTHS,
        wavelength_min=DEFAULT_WAVELENGTH_MIN,
        wavelength_max=DEFAULT_WAVELENGTH_MAX,
        boundary_margin_nm=DEFAULT_BOUNDARY_MARGIN_NM,
        prominence_ratio=DEFAULT_PEAK_PROMINENCE_RATIO,
        min_distance_nm=DEFAULT_PEAK_MIN_DISTANCE_NM,
    )
    return wavelengths, intensities, peak_indices


def _normalize_lengths(lengths: Sequence[float]) -> np.ndarray:
    """Normalize positive comb lengths so the largest length equals one."""
    values = np.asarray(lengths, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("At least one comb length is required.")
    if not np.isfinite(values).all():
        raise ValueError("Comb lengths must contain only finite values.")
    maximum = float(np.max(values))
    if maximum <= 0:
        raise ValueError("Comb lengths must contain at least one positive value.")
    return values / maximum


def _compute_upper_positions(
    base_wavelengths: np.ndarray,
    wavelength_min: float,
    wavelength_max: float,
    random_seed: Optional[int],
    x_jitter_nm: float,
    min_gap_nm: float,
) -> np.ndarray:
    """Match Figure 1's random upper-line offset and spacing behavior."""
    if not np.isfinite(x_jitter_nm) or x_jitter_nm < 0:
        raise ValueError("x_jitter_nm must be a finite non-negative number.")
    if not np.isfinite(min_gap_nm) or min_gap_nm < 0:
        raise ValueError("min_gap_nm must be a finite non-negative number.")
    if base_wavelengths.size == 0:
        raise ValueError("At least one peak is required.")
    if not np.all(np.diff(base_wavelengths) > 0):
        raise ValueError("Peak wavelengths must be strictly increasing.")

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
            raise ValueError("Could not keep the upper comb lines separated.")

    return positions


def build_normalized_comb_geometry(
    wavelengths: Sequence[float],
    intensities: Sequence[float],
    peak_indices: Sequence[int],
    wavelength_min: float = DEFAULT_WAVELENGTH_MIN,
    wavelength_max: float = DEFAULT_WAVELENGTH_MAX,
    random_seed: Optional[int] = DEFAULT_TOP_LINE_RANDOM_SEED,
    x_jitter_nm: float = DEFAULT_TOP_LINE_X_JITTER_NM,
    min_gap_nm: float = DEFAULT_TOP_LINE_MIN_GAP_NM,
) -> NormalizedCombGeometry:
    """Build normalized upper/lower comb coordinates from selected peaks."""
    wavelengths = np.asarray(wavelengths, dtype=float).reshape(-1)
    intensities = np.asarray(intensities, dtype=float).reshape(-1)
    peak_indices = np.asarray(peak_indices, dtype=int).reshape(-1)

    if wavelengths.size == 0 or wavelengths.size != intensities.size:
        raise ValueError("Wavelengths and intensities must be non-empty and have equal length.")
    if not np.isfinite(wavelengths).all() or not np.isfinite(intensities).all():
        raise ValueError("Wavelengths and intensities must contain only finite values.")
    if peak_indices.size == 0:
        raise ValueError("At least one selected peak is required.")
    if np.any(peak_indices < 0) or np.any(peak_indices >= wavelengths.size):
        raise ValueError("peak_indices contains an out-of-range index.")
    if not np.isfinite(wavelength_min) or not np.isfinite(wavelength_max):
        raise ValueError("Wavelength limits must be finite numbers.")
    if wavelength_min >= wavelength_max:
        raise ValueError("wavelength_min must be smaller than wavelength_max.")

    order = np.argsort(wavelengths[peak_indices])
    lower_positions = wavelengths[peak_indices][order]
    peak_lengths = intensities[peak_indices][order]
    lower_lengths = _normalize_lengths(peak_lengths)
    upper_lengths = _normalize_lengths(peak_lengths)
    upper_positions = _compute_upper_positions(
        lower_positions,
        wavelength_min,
        wavelength_max,
        random_seed,
        x_jitter_nm,
        min_gap_nm,
    )

    return NormalizedCombGeometry(
        lower_positions=lower_positions,
        upper_positions=upper_positions,
        lower_lengths=lower_lengths,
        upper_lengths=upper_lengths,
        lower_base_y=LOWER_BASE_Y,
        upper_top_y=UPPER_TOP_Y,
        lower_horizontal_limits=np.array(
            [float(np.min(lower_positions)), float(np.max(lower_positions))]
        ),
        upper_horizontal_limits=np.array(
            [float(np.min(upper_positions)), float(np.max(upper_positions))]
        ),
    )


def _draw_comb_lines(
    ax,
    positions: np.ndarray,
    lengths: np.ndarray,
    base_y: float,
    top_aligned: bool,
    color: str,
) -> None:
    """Draw a normalized comb with continuous beveled endpoint paths.

    The two endpoint verticals and their connector are one ``Line2D`` path.
    This gives the endpoint corners a real join style, while middle verticals
    remain independent lines.
    """
    if top_aligned:
        starts = UPPER_TOP_Y - lengths
        ends = np.full_like(lengths, UPPER_TOP_Y)
    else:
        starts = np.full_like(lengths, base_y)
        ends = base_y + lengths

    # Keep the middle comb teeth as independent vertical lines. The two
    # endpoint teeth are included in the continuous path below.
    for position, start, end in zip(positions[1:-1], starts[1:-1], ends[1:-1]):
        ax.vlines(
            position,
            float(start),
            float(end),
            colors=color,
            linewidth=DEFAULT_VERTICAL_LINE_WIDTH,
            alpha=DEFAULT_VERTICAL_LINE_ALPHA,
            zorder=DEFAULT_COMB_MIDDLE_LINE_ZORDER,
        )

    if positions.size > 1:
        if top_aligned:
            path_y = [
                float(starts[0]),
                float(ends[0]),
                float(ends[-1]),
                float(starts[-1]),
            ]
        else:
            path_y = [
                float(ends[0]),
                float(starts[0]),
                float(starts[-1]),
                float(ends[-1]),
            ]
        path_x = [
            float(positions[0]),
            float(positions[0]),
            float(positions[-1]),
            float(positions[-1]),
        ]
        endpoint_path = Line2D(
            path_x,
            path_y,
            color=color,
            linewidth=DEFAULT_VERTICAL_LINE_WIDTH,
            alpha=DEFAULT_VERTICAL_LINE_ALPHA,
            solid_joinstyle=DEFAULT_COMB_JOINSTYLE,
            solid_capstyle=DEFAULT_COMB_CAPSTYLE,
            zorder=DEFAULT_COMB_ENDPOINT_PATH_ZORDER,
        )
        ax.add_line(endpoint_path)
    elif positions.size == 1:
        ax.vlines(
            float(positions[0]),
            float(starts[0]),
            float(ends[0]),
            colors=color,
            linewidth=DEFAULT_VERTICAL_LINE_WIDTH,
            alpha=DEFAULT_VERTICAL_LINE_ALPHA,
            zorder=DEFAULT_COMB_MIDDLE_LINE_ZORDER,
        )


def plot_normalized_comb(
    data_path: Union[str, Path] = DEFAULT_DATA_PATH,
    output_path: Optional[Union[str, Path]] = DEFAULT_OUTPUT_PATH,
    show: bool = False,
):
    """Plot and optionally save the normalized Figure 1 comb inset."""
    wavelengths, intensities, peak_indices = load_current_selection(data_path)
    geometry = build_normalized_comb_geometry(
        wavelengths,
        intensities,
        peak_indices,
    )

    fig, ax = plt.subplots(figsize=FIGSIZE, facecolor="white")
    ax.set_facecolor("white")
    _draw_comb_lines(
        ax,
        geometry.lower_positions,
        geometry.lower_lengths,
        geometry.lower_base_y,
        top_aligned=False,
        color=DEFAULT_VERTICAL_LINE_COLOR,
    )
    _draw_comb_lines(
        ax,
        geometry.upper_positions,
        geometry.upper_lengths,
        geometry.upper_top_y,
        top_aligned=True,
        color=DEFAULT_TOP_LINE_COLOR,
    )

    ax.set_xlim(
        min(geometry.lower_positions.min(), geometry.upper_positions.min()) - 3.0,
        max(geometry.lower_positions.max(), geometry.upper_positions.max()) + 3.0,
    )
    ax.set_ylim(*Y_LIMITS)
    ax.set_axis_off()
    ax.text(
        0.5,
        0.96,
        TOP_LABEL,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=13,
        fontweight="semibold",
        color=DEFAULT_TOP_LINE_COLOR,
    )
    ax.text(
        0.5,
        0.04,
        BOTTOM_LABEL,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="semibold",
        color=DEFAULT_VERTICAL_LINE_COLOR,
    )

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    figure_border = Rectangle(
        (0.0, 0.0),
        1.0,
        1.0,
        transform=fig.transFigure,
        fill=False,
        edgecolor=DEFAULT_BORDER_COLOR,
        linewidth=DEFAULT_BORDER_LINEWIDTH,
        clip_on=False,
        zorder=DEFAULT_BORDER_ZORDER,
        joinstyle="miter",
    )
    if hasattr(fig, "add_artist"):
        fig.add_artist(figure_border)
    else:
        # Matplotlib 2.2 has no Figure.add_artist; figure.patches
        # still renders a figure-level patch correctly.
        fig.patches.append(figure_border)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_path,
            dpi=SAVE_DPI,
            facecolor=fig.get_facecolor(),
        )
    if show:
        plt.show()
    return fig, ax, geometry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    fig, _, geometry = plot_normalized_comb(
        data_path=args.data,
        output_path=args.output,
        show=args.show,
    )
    plt.close(fig)
    print(f"Saved normalized comb figure to: {args.output}")
    print(f"Selected peaks: {geometry.lower_positions.size}")
    print(
        "Selected peak wavelengths (nm): "
        + ", ".join(f"{value:.2f}" for value in geometry.lower_positions)
    )
    print(f"Top line color: {DEFAULT_TOP_LINE_COLOR}")
    print(f"Vertical line color: {DEFAULT_VERTICAL_LINE_COLOR}")


if __name__ == "__main__":
    main()

