# -*- coding: utf-8 -*-
"""Four-curve Figure 6 spectrum for the 378.5-385.6 nm Eu II window.

The curve geometry and colors follow ``example1.png`` while the axes styling
follows ``plot_figure6.py``:
- 1253 x 800 px canvas (figsize 12.53 x 8.0 in at dpi 100);
- four curves read directly from data.csv:
      blue #4575B4 = Sum(calc), red #D73027 = Eu II (8.6e-3),
      purple #897CD3 = Ti II (1.0e-1), yellow #FEE090 = Fe I (9.7e-2);
- no grid, tick marks, tick labels, axis labels, or title;
- a transparent legend for all four curves (no visible legend border);
- an inset black frame whose width is controlled by
  ``DEFAULTS["spines"]["linewidth"]``.

Usage:
    python plot_example1.py [--blue-lw F] [--red-lw F]
                            [--grid-lw F] [--grid-minor-lw F]
                            [--font-size F] [--font-family NAME]
                            [--font-weight W] [--xtick-pad F]
                            [--figsize WxH] [--dpi N]
                            [--out PATH.png] [--show]

``--show`` displays the figure in addition to saving the configured PNG.
"""

from __future__ import absolute_import, print_function

import argparse
import copy
import os
import sys

import matplotlib

# Headless-safe backend for saving; must be selected before pyplot import.
if "--show" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Bbox

_script_dir = os.path.dirname(os.path.abspath(__file__))

# All tunable plot parameters live here (mirrors the style of plot_figure6.py).
DEFAULTS = {
    # Input / output.  Paths are resolved relative to this file, not the CWD.
    "data_path": os.path.join(_script_dir, os.pardir, "data.csv"),
    "output_path": os.path.join(
        _script_dir, "output", "spectrum_378_5_385_6_euii_tiii_fei.png"),
    # Canvas.  figsize (12.53, 8.0) at dpi 100 gives exactly 1253 x 800 px.
    "figsize": (12.53, 8.0),
    "dpi": 100,
    "background": "white",
    # Pixel measurements below are expressed in this reference canvas.  The
    # resulting normalized axes rectangle stays stable when figsize or DPI is
    # changed, while the default still renders exactly 1253 x 800 pixels.
    "reference_canvas_px": (1253.0, 800.0),
    "axis_bottom_px": 727.5,
    # Inset the axes so all four sides of the black frame are visible.
    "axes": {
        "rect": (0.035, 0.05, 0.94, 0.92),  # left, bottom, width, height
        "facecolor": "white",
        "axisbelow": True,
    },
    # Exact schema selection: no fuzzy Eu-column guessing is performed.
    "wavelength_column": "Wavelength (nm)",
    "minimum_points": 2,
    # Display limits retain the current left/right and top/bottom whitespace.
    "xlim": (378.316, 385.754),
    "ylim": (-1000, 30500),
    # Rows of data.csv used for plotting (wavelength window, nm).  The plot is
    # a per-point polyline; xlim crops both sides.
    "data_window": (378.5, 385.6),
    # Internal tick locations are retained only for optional grid use.
    # Tick marks and tick labels are hidden in the default frame-only style.
    "xticks": [379, 380, 381, 382, 383, 384, 385],
    "xticks_minor": [379.5, 380.5, 381.5, 382.5, 383.5, 384.5],
    "yticks": [0, 5000, 10000, 15000, 20000, 25000, 30000],
    "yticks_minor": [2500, 7500, 12500, 17500, 22500, 27500],
    # Optional grid styling.  Both grids are disabled by default.
    "grid_major": {
        "color": "#CCCCCC", "linewidth": 1.5, "linestyle": "-",
        "alpha": 1.0, "antialiased": True, "visible": False,
    },
    "grid_minor": {
        "color": "#EBEBEB", "linewidth": 1.5, "linestyle": "-",
        "alpha": 1.0, "antialiased": True, "visible": False,
    },
    # Font settings are retained for optional labels; labels are hidden by
    # default in the frame-only style.
    "font": {
        "family": "Arial",
        "size": 22.0,         # points
        "weight": "normal",
        "color": "#444444",
        "xtick_pad": 17.0,    # points between axis bottom and label bbox
    },
    # Legacy compatibility switch for manually drawing a clipped left label.
    "draw_clipped_left_label": False,
    # Optional extension of the curve clip box below the y=0 baseline.
    "clip_bottom_extension_px": 1.5,
    # Curves are drawn in this order; later entries appear on top.
    "draw_order": ["blue", "red", "ti", "fe"],
    # Curve definitions (structure is extensible: add entries + draw_order).
    # linewidth 2.9 pt @ 100 dpi = ~4.0 px stroke with a ~3 px pure-color
    # core, matching the reference (calibrated against pure-pixel counts).
    "curves": {
        "blue": {"column": "Sum(calc)", "color": "#4575B4",
                 "linewidth": 4.5, "linestyle": "-", "label": "Sum",
                 "antialiased": True, "solid_capstyle": "projecting",
                 "solid_joinstyle": "round",
                 "visible": True},
        "red": {"column": "Eu II (8.6e-3)", "color": "#D73027",
                "linewidth": 4.5, "linestyle": "-", "label": "Eu II",
                "antialiased": True, "solid_capstyle": "projecting",
                "solid_joinstyle": "round",
                "visible": True},
        "ti": {"column": "Ti II (1.0e-1)", "color": "#897CD3",
               "linewidth": 4.5, "linestyle": "-", "label": "Ti II",
               "antialiased": True, "solid_capstyle": "projecting",
               "solid_joinstyle": "round",
               "visible": True},
        "fe": {"column": "Fe I (9.7e-2)", "color": "#FEE090",
               "linewidth": 4.5, "linestyle": "-", "label": "Fe I",
               "antialiased": True, "solid_capstyle": "projecting",
               "solid_joinstyle": "round",
               "visible": True},
    },
    # Raw per-point polylines match the reference.  Enable this only when a
    # visually smoother presentation is preferred over exact source values.
    "smoothing": {
        "enabled": False,
        "method": "moving_average",
        "window": 3,
    },
    # Plain black axes frame.  Change ``linewidth`` to tune its thickness.
    "spines": {"visible": True, "color": "#000000", "linewidth": 5},
    "legend": {
        "visible": True,
        "location": "upper right",
        "frameon": True,
        "edgecolor": "#000000",
        "facecolor": "#FFFFFF",
        "framealpha": 0,
        "fancybox": False,
        # Used when framealpha is above 0; framealpha=0 hides the box.
        "linewidth": 4.0,
        "fontsize": 25.0,
    },
    "labels": {
        "x": "",
        "y": "",
        "title": "",
        "fontsize": 14.0,
        "fontweight": "normal",
        "color": "#444444",
    },
    "show": False,
}


def _axes_rect(settings):
    """Return the configured axes rectangle, with legacy geometry fallback."""
    rect = settings["axes"].get("rect")
    if rect is not None:
        return list(rect)
    reference_height_px = float(settings["reference_canvas_px"][1])
    bottom = 1.0 - float(settings["axis_bottom_px"]) / reference_height_px
    return [0.0, bottom, 1.0, 1.0 - bottom]


def _read_window(settings):
    """Read and validate the configured CSV window before any figure exists."""
    data_path = settings["data_path"]
    data = pd.read_csv(data_path)
    wavelength_column = settings["wavelength_column"]
    curves = settings["curves"]
    draw_order = settings.get("draw_order") or sorted(curves)

    curve_columns = []
    for key in draw_order:
        if key not in curves:
            raise ValueError("draw_order references unknown curve: {0!r}".format(key))
        curve = curves[key]
        if curve.get("visible", True):
            curve_columns.append(curve["column"])
    required = [wavelength_column] + curve_columns
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError("missing required columns in {0}: {1}".format(
            os.path.abspath(data_path), ", ".join(missing)))

    wavelength = pd.to_numeric(data[wavelength_column], errors="coerce")
    wavelength_values = wavelength.to_numpy(dtype=float)
    if not np.isfinite(wavelength_values).all():
        raise ValueError("{0} must contain only finite numeric values".format(
            wavelength_column))
    differences = np.diff(wavelength_values)
    if np.any(differences == 0):
        raise ValueError("{0} contains duplicate values".format(wavelength_column))
    if np.any(differences < 0):
        raise ValueError("{0} must be strictly increasing".format(
            wavelength_column))

    window = settings["data_window"]
    inside = (wavelength_values >= window[0]) & (wavelength_values <= window[1])
    count = int(inside.sum())
    minimum_points = settings["minimum_points"]
    if count == 0:
        raise ValueError("data window {0} contains no rows".format(window))
    if count < minimum_points:
        raise ValueError("data window must contain at least {0} points; got {1}"
                         .format(minimum_points, count))

    selected = data.loc[inside, required].copy()
    selected[wavelength_column] = wavelength_values[inside]
    for column in curve_columns:
        numeric = pd.to_numeric(selected[column], errors="coerce")
        values = numeric.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("{0} must contain only finite numeric values in "
                             "the data window".format(column))
        selected[column] = values
    return selected


def _deep_merge(base, overrides):
    """Return a recursive copy of ``base`` updated by ``overrides``."""
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _smooth_values(values, settings):
    """Apply the configured smoothing method while preserving sample count."""
    result = np.asarray(values, dtype=float)
    if not settings.get("enabled", False):
        return result
    method = settings.get("method")
    if method != "moving_average":
        raise ValueError("unsupported smoothing method: {0!r}".format(method))
    window = settings.get("window", 3)
    if isinstance(window, bool) or not isinstance(window, int):
        raise ValueError("smoothing window must be an odd integer")
    if window < 1 or window % 2 == 0:
        raise ValueError("smoothing window must be a positive odd integer")
    if window > len(result):
        raise ValueError("smoothing window cannot exceed the data length")
    if window == 1:
        return result.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(result, kernel, mode="same")


def plot_example1(config=None):
    """Plot the example1 spectrum; ``config`` recursively merges over defaults.

    Nested dicts (font / grid_* / curves) are merged key-wise so callers can
    override single fields, e.g. ``config={"curves": {"green":
    {"linewidth": 3.0}}}``.

    Returns ``(fig, ax)``.
    """
    settings = _deep_merge(DEFAULTS, config or {})

    output_path = settings["output_path"]
    if output_path is not None:
        output_path = os.fspath(output_path)
        if os.path.splitext(output_path)[1].lower() != ".png":
            raise ValueError("output_path must end with .png: {0}".format(
                output_path))
        settings["output_path"] = output_path

    grid_major = settings["grid_major"]
    grid_minor = settings["grid_minor"]
    axes = settings["axes"]
    font = settings["font"]
    curves = settings["curves"]
    smoothing = settings["smoothing"]
    spines = settings["spines"]
    legend = settings["legend"]
    labels = settings["labels"]

    data = _read_window(settings)
    wavelength = data[settings["wavelength_column"]]

    fig = plt.figure(figsize=tuple(settings["figsize"]), dpi=settings["dpi"])
    fig.patch.set_facecolor(settings["background"])
    ax = fig.add_axes(_axes_rect(settings))
    ax.set_facecolor(axes["facecolor"])

    # --- grid-only ticks: explicit major + minor lists --------------------
    # NOTE: set_xticks/set_yticks must run BEFORE set_xlim/set_ylim because
    # matplotlib 2.2.3's Axis.set_ticks() expands the view interval to cover
    # the tick values, which would otherwise corrupt the measured limits.
    ax.set_xticks(list(settings["xticks"]))
    ax.set_xticks(list(settings["xticks_minor"]), minor=True)
    ax.set_yticks(list(settings["yticks"]))
    ax.set_yticks(list(settings["yticks_minor"]), minor=True)
    ax.set_xlim(*settings["xlim"])
    ax.set_ylim(*settings["ylim"])

    if grid_major["visible"]:
        ax.grid(True, which="major", color=grid_major["color"],
                linewidth=grid_major["linewidth"],
                linestyle=grid_major["linestyle"], alpha=grid_major["alpha"],
                antialiased=grid_major["antialiased"])
    else:
        ax.grid(False, which="major")
    if grid_minor["visible"]:
        ax.grid(True, which="minor", color=grid_minor["color"],
                linewidth=grid_minor["linewidth"],
                linestyle=grid_minor["linestyle"], alpha=grid_minor["alpha"],
                antialiased=grid_minor["antialiased"])
    else:
        ax.grid(False, which="minor")
    # Matplotlib 2.2 may keep previously created grid artists visible even
    # after ``grid(False)``; hide them explicitly for the frame-only style.
    if not grid_major["visible"] and not grid_minor["visible"]:
        for gridline in ax.get_xgridlines() + ax.get_ygridlines():
            gridline.set_visible(False)
    ax.set_axisbelow(axes["axisbelow"])

    # --- plain frame only: no grid, tick marks, or tick labels ------------
    for spine in ax.spines.values():
        spine.set_visible(spines["visible"])
        spine.set_color(spines["color"])
        spine.set_linewidth(spines["linewidth"])
    ax.tick_params(axis="both", which="both",
                   bottom=False, top=False, left=False, right=False,
                   labelbottom=False, labeltop=False,
                   labelleft=False, labelright=False,
                   length=0, width=0)

    # --- curves: per-point polylines in configured draw order -------------
    draw_order = settings.get("draw_order") or sorted(curves)
    lines = []
    for key in draw_order:
        curve = curves[key]
        if not curve.get("visible", True):
            continue
        values = pd.to_numeric(data[curve["column"]], errors="coerce")
        plotted_values = _smooth_values(values.to_numpy(dtype=float), smoothing)
        lines.append(ax.plot(wavelength.to_numpy(dtype=float),
                             plotted_values,
                             color=curve["color"],
                             linewidth=curve["linewidth"],
                             linestyle=curve["linestyle"],
                             antialiased=curve["antialiased"],
                             solid_capstyle=curve["solid_capstyle"],
                             solid_joinstyle=curve["solid_joinstyle"],
                             label=curve["label"])[0])

    ax.set_xlabel(labels["x"], fontsize=labels["fontsize"],
                  fontweight=labels["fontweight"], color=labels["color"])
    ax.set_ylabel(labels["y"], fontsize=labels["fontsize"],
                  fontweight=labels["fontweight"], color=labels["color"])
    ax.set_title(labels["title"], fontsize=labels["fontsize"],
                 fontweight=labels["fontweight"], color=labels["color"])
    if legend["visible"]:
        legend_artist = ax.legend(
            loc=legend["location"], frameon=legend["frameon"],
            fontsize=legend["fontsize"], fancybox=legend["fancybox"],
            framealpha=legend["framealpha"],
            facecolor=legend["facecolor"], edgecolor=legend["edgecolor"])
        legend_artist.get_frame().set_linewidth(legend["linewidth"])

    # The reference strokes extend ~1 px below the y=0 axis edge (they were
    # not clipped there), so extend the curves' clip box downwards.
    extension_px = settings.get("clip_bottom_extension_px", 0.0)
    if extension_px and lines:
        bbox = ax.bbox
        output_height_px = settings["figsize"][1] * settings["dpi"]
        scale_y = output_height_px / float(settings["reference_canvas_px"][1])
        clip = Bbox.from_extents(bbox.x0, bbox.y0 - extension_px * scale_y,
                                 bbox.x1, bbox.y1)
        for line in lines:
            line.set_clip_box(clip)

    # Redraw the left-most x tick label if its tick sits left of the axes
    # (matplotlib skips off-view ticks; the reference still shows the ink,
    # clipped at the canvas edge).
    if settings.get("draw_clipped_left_label"):
        x0, x1 = settings["xlim"]
        axes_width_px = settings["figsize"][0] * settings["dpi"]
        first_tick = settings["xticks"][0]
        tick_px = (first_tick - x0) * axes_width_px / (x1 - x0)
        if tick_px < 0:
            props = FontProperties(family=font["family"],
                                   weight=font["weight"],
                                   size=font["size"])
            # NOTE: "offset points" expects POINTS (same unit as xtick_pad),
            # so pass the pad directly -- converting to px here would double
            # the offset and push the label ~9 px too low.
            ax.annotate(str(int(first_tick))
                        if float(first_tick).is_integer()
                        else str(first_tick),
                        xy=(tick_px / axes_width_px, 0.0),
                        xycoords="axes fraction",
                        xytext=(0, -font["xtick_pad"]),
                        textcoords="offset points",
                        ha="center", va="top",
                        fontproperties=props, color=font["color"],
                        annotation_clip=False, clip_on=False)

    if settings["output_path"]:
        out_dir = os.path.dirname(os.path.abspath(settings["output_path"]))
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        # NOTE: never use tight_layout / bbox_inches='tight' here; the canvas
        # must stay exactly figsize * dpi = 1253 x 800 px.
        fig.savefig(settings["output_path"], dpi=settings["dpi"],
                    facecolor=settings["background"], format="png")
    if settings["show"]:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def _parse_size(text):
    """Parse a WxH string such as ``12.53x8.0`` into a (w, h) tuple."""
    try:
        width, height = text.lower().split("x", 1)
        return (float(width), float(height))
    except ValueError:
        raise argparse.ArgumentTypeError(
            "figsize must look like 12.53x8.0, got {0!r}".format(text))


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Plot Sum, Eu II, Ti II, and Fe I at 378.5-385.6 nm "
                    "from data.csv.")
    parser.add_argument("--blue-lw", type=float, default=None,
                        help="linewidth of the Sum(calc) curve")
    parser.add_argument("--red-lw", type=float, default=None,
                        help="linewidth of the Eu II curve")
    parser.add_argument("--grid-lw", type=float, default=None,
                        help="linewidth of the major grid")
    parser.add_argument("--grid-minor-lw", type=float, default=None,
                        help="linewidth of the minor grid")
    parser.add_argument("--font-size", type=float, default=None,
                        help="x tick label font size (pt)")
    parser.add_argument("--font-family", default=None,
                        help="x tick label font family (e.g. Arial)")
    parser.add_argument("--font-weight", default=None,
                        help="x tick label font weight (normal/semibold/bold)")
    parser.add_argument("--xtick-pad", type=float, default=None,
                        help="pad (pt) between axis bottom and x tick labels")
    parser.add_argument("--figsize", type=_parse_size, default=None,
                        help="canvas size in inches, e.g. 12.53x8.0")
    parser.add_argument("--dpi", type=int, default=None,
                        help="canvas resolution (default 100)")
    parser.add_argument("--out", default=None,
                        help="output path (must end with .png)")
    parser.add_argument("--show", action="store_true",
                        help="display the figure in addition to saving it")
    return parser


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)

    config = {}
    curves = {}
    if args.blue_lw is not None:
        curves["blue"] = {"linewidth": args.blue_lw}
    if args.red_lw is not None:
        curves["red"] = {"linewidth": args.red_lw}
    if curves:
        config["curves"] = curves
    if args.grid_lw is not None:
        config["grid_major"] = dict(DEFAULTS["grid_major"],
                                    linewidth=args.grid_lw)
    if args.grid_minor_lw is not None:
        config["grid_minor"] = dict(DEFAULTS["grid_minor"],
                                    linewidth=args.grid_minor_lw)
    font = {}
    if args.font_size is not None:
        font["size"] = args.font_size
    if args.font_family is not None:
        font["family"] = args.font_family
    if args.font_weight is not None:
        font["weight"] = args.font_weight
    if args.xtick_pad is not None:
        font["xtick_pad"] = args.xtick_pad
    if font:
        config["font"] = font
    if args.figsize is not None:
        config["figsize"] = args.figsize
    if args.dpi is not None:
        config["dpi"] = args.dpi
    if args.out is not None:
        config["output_path"] = args.out
    if args.show:
        config["show"] = True

    fig, ax = plot_example1(config or None)
    out = config.get("output_path", DEFAULTS["output_path"])
    if out is not None:
        print("saved: {0}".format(os.path.abspath(out)))
    return fig, ax


if __name__ == "__main__":
    main()
