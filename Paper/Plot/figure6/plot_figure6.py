from __future__ import absolute_import

import os

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd


_script_dir = os.path.dirname(os.path.abspath(__file__))

# All project-specific plotting defaults live here so the figure can be tuned
# without searching through the plotting code.
DEFAULTS = {
    "data_path": os.path.join(_script_dir, "data.csv"),
    "output_path": os.path.join(_script_dir, "figure6_spectrum.png"),
    "figsize": (9, 5),
    "line_color": "#4575B4",
    "line_width": 2.2,
    "line_style": "-",
    "line_label": "Spectrum",
    "x_label": "Wavelength (nm)",
    "y_label": "Intensity",
    "y_scale": "linear",
    "y_limits": (0, 275000),
    "peak_adjustment_enabled": True,
    "peak_adjustment_bounds": (278.50, 279.97),
    "peak_adjustment_factor": 0.7,
    "axis_label_size": 15,
    "axis_label_weight": "semibold",
    "spine_width": 1.8,
    "tick_label_size": 12,
    "tick_label_weight": "semibold",
    "tick_direction": "in",
    "tick_width": 2.0,
    "tick_length": 6,
    "ticks_top": True,
    "ticks_right": True,
    "minor_ticks": False,
    "x_major_tick_interval": 100,
    "y_major_tick_interval": 50000,
    "grid_visible": True,
    "grid_alpha": 0.3,
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
    "peak_marker_color": "#D73027",
    "peak_marker_size": 15,
    "peak_marker_zorder": 5,
    "peak_marker_edgecolor": "none",
    "peak_comb_visible": True,
    "peak_comb_height": 250000,
    "peak_comb_stem_bottom": 200000,
    "peak_comb_line_width": 1.2,
    "peak_comb_line_style": "-",
    "peak_comb_zorder": 4,
    "peak_comb_label": "_nolegend_",
    "mark_green_peaks": True,
    "green_peak_target_wavelengths": (211.71, 217.42, 289.1, 328.99, 369.42),
    "green_peak_search_half_width": 0.25,
    "green_peak_marker": "o",
    "green_peak_marker_color": "#1A9850",
    "green_peak_marker_size": 15,
    "green_peak_marker_zorder": 5,
    "green_peak_marker_edgecolor": "none",
    "green_peak_marker_label": "_nolegend_",
    "green_peak_comb_visible": True,
    "green_peak_comb_line_width": 1.2,
    "green_peak_comb_line_style": "-",
    "green_peak_comb_zorder": 4,
    "green_peak_comb_label": "_nolegend_",
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
    "show_legend": False,
    "legend_location": "upper right",
    "legend_font_weight": "semibold",
    "legend_font_size": 12,
    "legend_frame": False,
    "tight_layout": True,
    "dpi": 300,
    "show": False,
}


def _read_spectrum(data_path):
    try:
        data = pd.read_csv(data_path, usecols=[0, 1])
    except ValueError as exc:
        # pandas 1.1.x can leave its parser referenced by the traceback on
        # Windows. Detaching it also lets callers immediately move/delete an
        # invalid input file after handling this clearer domain error.
        exc.__traceback__ = None
        raise ValueError(
            "Spectrum CSV must contain at least two columns: {0}".format(data_path)
        ) from None

    wavelength = pd.to_numeric(data.iloc[:, 0], errors="coerce")
    intensity = pd.to_numeric(data.iloc[:, 1], errors="coerce")
    valid = np.isfinite(wavelength.to_numpy(dtype=float)) & np.isfinite(
        intensity.to_numpy(dtype=float)
    )

    wavelength = wavelength.loc[valid].to_numpy(dtype=float)
    intensity = intensity.loc[valid].to_numpy(dtype=float)
    if wavelength.size == 0:
        raise ValueError(
            "Spectrum CSV has no valid wavelength/intensity pairs: {0}".format(
                data_path
            )
        )

    return wavelength, intensity


def _find_peak_near_target(wavelength, intensity, target, half_width):
    try:
        target = float(target)
    except (TypeError, ValueError):
        raise ValueError("peak target wavelength must be finite")
    if not np.isfinite(target):
        raise ValueError("peak target wavelength must be finite")

    try:
        half_width = float(half_width)
    except (TypeError, ValueError):
        raise ValueError("peak_search_half_width must be finite and positive")
    if not np.isfinite(half_width) or half_width <= 0:
        raise ValueError("peak_search_half_width must be finite and positive")

    in_window = (wavelength >= target - half_width) & (
        wavelength <= target + half_width
    )
    candidate_indices = np.flatnonzero(in_window)
    if candidate_indices.size == 0:
        raise ValueError(
            "No samples in peak search window around target {0}".format(target)
        )

    order = np.lexsort(
        (
            wavelength[candidate_indices],
            np.abs(wavelength[candidate_indices] - target),
            -intensity[candidate_indices],
        )
    )
    peak_index = candidate_indices[order[0]]
    return wavelength[peak_index], intensity[peak_index]


def _adjust_peak_region(wavelength, intensity, bounds, factor):
    try:
        lower_bound, upper_bound = bounds
        lower_bound = float(lower_bound)
        upper_bound = float(upper_bound)
    except (TypeError, ValueError):
        raise ValueError(
            "peak_adjustment_bounds must contain two finite increasing values"
        )
    if (
        not np.isfinite(lower_bound)
        or not np.isfinite(upper_bound)
        or lower_bound >= upper_bound
    ):
        raise ValueError(
            "peak_adjustment_bounds must contain two finite increasing values"
        )

    try:
        factor = float(factor)
    except (TypeError, ValueError):
        raise ValueError(
            "peak_adjustment_factor must be finite and between 0 and 1"
        )
    if not np.isfinite(factor) or factor < 0 or factor > 1:
        raise ValueError(
            "peak_adjustment_factor must be finite and between 0 and 1"
        )

    wavelength = np.asarray(wavelength, dtype=float)
    raw_intensity = np.asarray(intensity, dtype=float)
    adjusted = raw_intensity.copy()
    in_bounds = (wavelength >= lower_bound) & (wavelength <= upper_bound)
    if np.any(in_bounds):
        endpoint_intensity = np.interp(
            [lower_bound, upper_bound], wavelength, raw_intensity
        )
        baseline = np.interp(
            wavelength[in_bounds],
            [lower_bound, upper_bound],
            endpoint_intensity,
        )
        adjusted[in_bounds] = baseline + factor * (
            raw_intensity[in_bounds] - baseline
        )
    return adjusted


def _build_peak_comb_segments(peaks, height, stem_bottom):
    try:
        height = float(height)
        stem_bottom = float(stem_bottom)
    except (TypeError, ValueError):
        raise ValueError(
            "peak_comb_height and peak_comb_stem_bottom must be finite "
            "with top greater than or equal to bottom"
        )
    if (
        not np.isfinite(height)
        or not np.isfinite(stem_bottom)
        or height < stem_bottom
    ):
        raise ValueError(
            "peak_comb_height and peak_comb_stem_bottom must be finite "
            "with top greater than or equal to bottom"
        )

    segments = [
        [(peak[0], height), (peak[0], stem_bottom)]
        for peak in peaks
    ]
    peak_wavelengths = [peak[0] for peak in peaks]
    segments.append(
        [(min(peak_wavelengths), height), (max(peak_wavelengths), height)]
    )
    return segments


def plot_spectrum(config=None):
    settings = DEFAULTS.copy()
    if config is not None:
        settings.update(config)

    wavelength, intensity = _read_spectrum(settings["data_path"])
    plot_intensity = intensity
    if settings["peak_adjustment_enabled"]:
        plot_intensity = _adjust_peak_region(
            wavelength,
            intensity,
            settings["peak_adjustment_bounds"],
            settings["peak_adjustment_factor"],
        )
    fig, ax = plt.subplots(figsize=settings["figsize"])
    ax.plot(
        wavelength,
        plot_intensity,
        color=settings["line_color"],
        linewidth=settings["line_width"],
        linestyle=settings["line_style"],
        label=settings["line_label"],
    )
    peaks = []
    if settings["mark_peaks"] and len(settings["peak_target_wavelengths"]) > 0:
        peaks = [
            _find_peak_near_target(
                wavelength,
                intensity,
                target,
                settings["peak_search_half_width"],
            )
            for target in settings["peak_target_wavelengths"]
        ]
        ax.scatter(
            [peak[0] for peak in peaks],
            [peak[1] for peak in peaks],
            marker=settings["peak_marker"],
            color=settings["peak_marker_color"],
            s=settings["peak_marker_size"],
            zorder=settings["peak_marker_zorder"],
            edgecolors=settings["peak_marker_edgecolor"],
            label="_nolegend_",
        )
        if settings["peak_comb_visible"]:
            red_comb_segments = _build_peak_comb_segments(
                peaks,
                settings["peak_comb_height"],
                settings["peak_comb_stem_bottom"],
            )
            red_comb = LineCollection(
                red_comb_segments,
                colors=settings["peak_marker_color"],
                linewidths=settings["peak_comb_line_width"],
                linestyles=settings["peak_comb_line_style"],
                zorder=settings["peak_comb_zorder"],
                label=settings["peak_comb_label"],
            )
            ax.add_collection(red_comb)
            if settings["peak_text_visible"]:
                peak_x = [peak[0] for peak in peaks]
                ax.text(
                    (min(peak_x) + max(peak_x)) / 2
                    + settings["peak_text_xoffset"],
                    red_comb_segments[0][0][1] + settings["peak_text_yoffset"],
                    settings["peak_text"],
                    color=settings["peak_marker_color"],
                    fontsize=settings["peak_text_fontsize"],
                    fontweight=settings["peak_text_fontweight"],
                    ha=settings["peak_text_ha"],
                    va=settings["peak_text_va"],
                    rotation=settings["peak_text_rotation"],
                    zorder=settings["peak_text_zorder"],
                )
    green_peaks = []
    if settings["mark_green_peaks"] and len(
        settings["green_peak_target_wavelengths"]
    ) > 0:
        green_peaks = [
            _find_peak_near_target(
                wavelength,
                intensity,
                target,
                settings["green_peak_search_half_width"],
            )
            for target in settings["green_peak_target_wavelengths"]
        ]
        ax.scatter(
            [peak[0] for peak in green_peaks],
            [peak[1] for peak in green_peaks],
            marker=settings["green_peak_marker"],
            color=settings["green_peak_marker_color"],
            s=settings["green_peak_marker_size"],
            zorder=settings["green_peak_marker_zorder"],
            edgecolors=settings["green_peak_marker_edgecolor"],
            label=settings["green_peak_marker_label"],
        )
        if settings["green_peak_comb_visible"]:
            green_comb_segments = _build_peak_comb_segments(
                green_peaks,
                settings["peak_comb_height"],
                settings["peak_comb_stem_bottom"],
            )
            green_comb = LineCollection(
                green_comb_segments,
                colors=settings["green_peak_marker_color"],
                linewidths=settings["green_peak_comb_line_width"],
                linestyles=settings["green_peak_comb_line_style"],
                zorder=settings["green_peak_comb_zorder"],
                label=settings["green_peak_comb_label"],
            )
            ax.add_collection(green_comb)
            if settings["green_peak_text_visible"]:
                green_peak_x = [peak[0] for peak in green_peaks]
                ax.text(
                    (min(green_peak_x) + max(green_peak_x)) / 2
                    + settings["green_peak_text_xoffset"],
                    green_comb_segments[0][0][1]
                    + settings["green_peak_text_yoffset"],
                    settings["green_peak_text"],
                    color=settings["green_peak_marker_color"],
                    fontsize=settings["peak_text_fontsize"],
                    fontweight=settings["peak_text_fontweight"],
                    ha=settings["peak_text_ha"],
                    va=settings["peak_text_va"],
                    rotation=settings["peak_text_rotation"],
                    zorder=settings["peak_text_zorder"],
                )

    ax.set_xlabel(
        settings["x_label"],
        fontsize=settings["axis_label_size"],
        fontweight=settings["axis_label_weight"],
    )
    ax.set_ylabel(
        settings["y_label"],
        fontsize=settings["axis_label_size"],
        fontweight=settings["axis_label_weight"],
    )
    ax.set_yscale(settings["y_scale"])
    ax.xaxis.set_major_locator(MultipleLocator(settings["x_major_tick_interval"]))
    ax.yaxis.set_major_locator(MultipleLocator(settings["y_major_tick_interval"]))

    for spine in ax.spines.values():
        spine.set_linewidth(settings["spine_width"])

    if settings["minor_ticks"]:
        ax.minorticks_on()
        ax.tick_params(
            axis="both",
            which="minor",
            direction=settings["tick_direction"],
            top=settings["ticks_top"],
            right=settings["ticks_right"],
            width=settings["tick_width"],
            length=settings["tick_length"],
            labelsize=settings["tick_label_size"],
        )
    else:
        ax.minorticks_off()
    ax.tick_params(
        axis="both",
        which="major",
        direction=settings["tick_direction"],
        top=settings["ticks_top"],
        right=settings["ticks_right"],
        width=settings["tick_width"],
        length=settings["tick_length"],
        labelsize=settings["tick_label_size"],
    )
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight(settings["tick_label_weight"])

    ax.grid(settings["grid_visible"], alpha=settings["grid_alpha"])
    if settings["show_legend"]:
        ax.legend(
            loc=settings["legend_location"],
            prop={
                "weight": settings["legend_font_weight"],
                "size": settings["legend_font_size"],
            },
            frameon=settings["legend_frame"],
        )

    y_limits = settings["y_limits"]
    if y_limits is not None:
        error_message = (
            "y_limits must be None or contain two finite-or-None values "
            "with lower less than upper"
        )
        if not isinstance(y_limits, tuple) or len(y_limits) != 2:
            raise ValueError(error_message)
        lower, upper = y_limits
        try:
            lower = None if lower is None else float(lower)
            upper = None if upper is None else float(upper)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(error_message)
        if (
            (lower is not None and not np.isfinite(lower))
            or (upper is not None and not np.isfinite(upper))
            or (
                lower is not None
                and upper is not None
                and lower >= upper
            )
        ):
            raise ValueError(error_message)
        ax.set_ylim(lower, upper)

    if settings["tight_layout"]:
        fig.tight_layout()
    if settings["output_path"]:
        fig.savefig(settings["output_path"], dpi=settings["dpi"])
    if settings["show"]:
        plt.show()

    return fig, ax


if __name__ == "__main__":
    plot_spectrum()
