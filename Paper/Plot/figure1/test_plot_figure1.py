from pathlib import Path
import importlib.util
import sys
import tempfile

import numpy as np
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.colors import to_rgba
from PIL import Image


FIGURE_DIR = Path(__file__).resolve().parent
MODULE_PATH = FIGURE_DIR / "plot_figure1.py"
DATA_PATH = FIGURE_DIR / "data3.csv"


def load_module():
    assert MODULE_PATH.exists(), "plot_figure1.py must exist"
    spec = importlib.util.spec_from_file_location("plot_figure1", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _line_collections(ax):
    return [
        collection
        for collection in ax.collections
        if isinstance(collection, LineCollection)
    ]


def _is_vertical(collection):
    segments = collection.get_segments()
    return len(segments) == 1 and np.isclose(np.ptp(segments[0][:, 0]), 0.0)


def _is_horizontal(collection):
    segments = collection.get_segments()
    return len(segments) == 1 and np.isclose(np.ptp(segments[0][:, 1]), 0.0)


def _line_x(collection):
    return float(collection.get_segments()[0][0, 0])


def _line_y(collection):
    return float(collection.get_segments()[0][0, 1])


def test_load_spectrum_data_reads_the_current_excerpt_csv():
    module = load_module()

    wavelengths, intensities = module.load_spectrum_data(DATA_PATH)

    assert len(wavelengths) == 8294
    assert len(intensities) == 8294
    assert wavelengths[0] == 198.79
    assert wavelengths[-1] == 603.55
    assert intensities.max() == 356800.0


def test_default_layout_matches_figure3_reference_and_extended_range():
    module = load_module()

    assert module.FIGSIZE == (9.6, 5.0)
    assert module.DEFAULT_WAVELENGTH_MIN == 220.0
    assert module.DEFAULT_WAVELENGTH_MAX == 330.0
    assert module.DEFAULT_BOUNDARY_MARGIN_NM == 10.0
    assert module.DEFAULT_TOP_LINE_COLOR == "#897CD3"
    assert module.DEFAULT_HORIZONTAL_LINE_EXTENSION_NM == 5.0


def test_cli_can_override_top_line_color(monkeypatch, capsys):
    module = load_module()
    captured = {}
    original_plot_spectrum = module.plot_spectrum

    def recording_plot_spectrum(*args, **kwargs):
        captured.update(kwargs)
        result = original_plot_spectrum(*args, **kwargs)
        top_markers = [
            collection
            for collection in result[1].collections
            if isinstance(collection, PathCollection)
        ]
        captured["top_marker_facecolor"] = tuple(top_markers[-1].get_facecolors()[0])
        return result

    monkeypatch.setattr(module, "plot_spectrum", recording_plot_spectrum)
    with tempfile.TemporaryDirectory() as temporary_directory:
        output_path = Path(temporary_directory) / "cli_spectrum.png"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "plot_figure1.py",
                "--output",
                str(output_path),
                "--top-line-color",
                "#123ABC",
            ],
        )

        module.main()

        assert output_path.exists()
    assert "Top line color: #123ABC" in capsys.readouterr().out
    assert captured["top_line_color"] == "#123ABC"
    assert captured["top_marker_facecolor"] == to_rgba("#123ABC")
    assert captured["show_axes"] is True


def test_plot_spectrum_defaults_to_axes_with_labels():
    module = load_module()
    wavelengths, intensities = module.load_spectrum_data(DATA_PATH)

    fig, ax = module.plot_spectrum(
        wavelengths,
        intensities,
        output_path=None,
        wavelength_min=220.0,
        wavelength_max=300.0,
        peak_indices=None,
        show=False,
    )

    assert ax.axison is True
    assert ax.get_xlabel() == "Wavelength (nm)"
    assert ax.get_ylabel() == "Intensity"
    assert ax.get_title() == ""
    assert all(spine.get_visible() for spine in ax.spines.values())
    assert any(tick.tick1line.get_visible() for tick in ax.xaxis.get_major_ticks())
    assert any(tick.tick1line.get_visible() for tick in ax.yaxis.get_major_ticks())
    assert any(label.get_visible() for label in ax.get_xticklabels())
    assert any(label.get_visible() for label in ax.get_yticklabels())
    module.plt.close(fig)


def test_plot_spectrum_can_explicitly_hide_axes():
    module = load_module()
    wavelengths, intensities = module.load_spectrum_data(DATA_PATH)

    fig, ax = module.plot_spectrum(
        wavelengths,
        intensities,
        output_path=None,
        wavelength_min=220.0,
        wavelength_max=300.0,
        peak_indices=None,
        show_axes=False,
        show=False,
    )

    assert ax.axison is False
    assert ax.get_xlabel() == ""
    assert ax.get_ylabel() == ""
    assert ax.get_title() == ""
    assert all(not spine.get_visible() for spine in ax.spines.values())
    assert all(not tick.tick1line.get_visible() for tick in ax.xaxis.get_major_ticks())
    assert all(not tick.tick1line.get_visible() for tick in ax.yaxis.get_major_ticks())
    module.plt.close(fig)


def test_manual_peak_wavelengths_are_snapped_and_annotated():
    module = load_module()
    wavelengths, intensities = module.load_spectrum_data(DATA_PATH)
    wavelength_min, wavelength_max = 220.0, 300.0
    boundary_margin_nm = 10.0
    manual_wavelengths = [233.0, 242.0, 251.0, 261.0, 274.0, 283.0]

    selected_indices = module.select_manual_peak_indices(
        wavelengths,
        intensities,
        manual_wavelengths,
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
        boundary_margin_nm=boundary_margin_nm,
    )

    selected_wavelengths = wavelengths[selected_indices]
    assert len(module.DEFAULT_MANUAL_PEAK_WAVELENGTHS) == 6
    assert len(selected_indices) == 6
    assert len(set(selected_indices.tolist())) == 6
    assert np.all(selected_wavelengths >= wavelength_min + boundary_margin_nm)
    assert np.all(selected_wavelengths <= wavelength_max - boundary_margin_nm)
    assert np.all(np.diff(selected_wavelengths) > 0)

    with tempfile.TemporaryDirectory() as temporary_directory:
        output_path = Path(temporary_directory) / "figure1_spectrum.png"
        fig, ax = module.plot_spectrum(
            wavelengths,
            intensities,
            output_path=output_path,
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
            peak_indices=selected_indices,
            vertical_line_color="#123456",
            top_line_color=module.DEFAULT_TOP_LINE_COLOR,
            spectrum_alpha=0.35,
            horizontal_extension_nm=7.5,
            show=False,
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0
        assert ax.get_xlim() == (wavelength_min, wavelength_max)
        assert ax.get_title() == ""
        assert ax.get_xlabel() == "Wavelength (nm)"
        assert ax.get_ylabel() == "Intensity"
        assert ax.axison is True
        assert ax.lines[0].get_color() == "gray"
        assert ax.lines[0].get_alpha() == 0.35
        assert ax.lines[0].get_linewidth() == 1.2

        peak_markers = [
            collection
            for collection in ax.collections
            if isinstance(collection, PathCollection)
        ]
        line_collections = _line_collections(ax)
        vertical_lines = [collection for collection in line_collections if _is_vertical(collection)]
        horizontal_lines = [collection for collection in line_collections if _is_horizontal(collection)]
        assert len(peak_markers) == 2
        assert len(vertical_lines) == 12
        assert len(horizontal_lines) == 2
        regional_max = intensities[
            (wavelengths >= wavelength_min) & (wavelengths <= wavelength_max)
        ].max()
        lower_markers = [
            collection
            for collection in peak_markers
            if np.all(collection.get_offsets()[:, 1] <= regional_max)
        ]
        top_markers = [
            collection
            for collection in peak_markers
            if np.all(collection.get_offsets()[:, 1] > regional_max)
        ]
        assert len(lower_markers) == 1
        assert len(top_markers) == 1
        assert lower_markers[0].get_offsets().shape == (6, 2)
        assert top_markers[0].get_offsets().shape == (6, 2)
        assert to_rgba(module.DEFAULT_PEAK_MARKER_COLOR) == tuple(lower_markers[0].get_facecolors()[0])
        assert to_rgba(module.DEFAULT_TOP_LINE_COLOR) == tuple(top_markers[0].get_facecolors()[0])
        assert np.allclose(
            top_markers[0].get_sizes(),
            lower_markers[0].get_sizes(),
        )

        lower_lines = sorted(
            [collection for collection in vertical_lines if _line_y(collection) == 0.0],
            key=_line_x,
        )
        top_lines = sorted(
            [collection for collection in vertical_lines if _line_y(collection) > regional_max],
            key=_line_x,
        )
        lower_horizontal = [collection for collection in horizontal_lines if _line_y(collection) == 0.0]
        top_horizontal = [collection for collection in horizontal_lines if _line_y(collection) > regional_max]

        assert len(lower_lines) == 6
        assert len(top_lines) == 6
        assert len(lower_horizontal) == 1
        assert len(top_horizontal) == 1

        assert all(
            tuple(color) == to_rgba("#123456", alpha=module.DEFAULT_VERTICAL_LINE_ALPHA)
            for collection in lower_lines
            for color in collection.get_colors()
        )
        assert all(
            tuple(color) == to_rgba(module.DEFAULT_TOP_LINE_COLOR, alpha=module.DEFAULT_VERTICAL_LINE_ALPHA)
            for collection in top_lines
            for color in collection.get_colors()
        )
        assert tuple(lower_horizontal[0].get_colors()[0]) == to_rgba(
            "#123456", alpha=module.DEFAULT_VERTICAL_LINE_ALPHA
        )
        assert tuple(top_horizontal[0].get_colors()[0]) == to_rgba(
            module.DEFAULT_TOP_LINE_COLOR, alpha=module.DEFAULT_VERTICAL_LINE_ALPHA
        )
        assert lower_horizontal[0].get_linewidths()[0] == module.DEFAULT_VERTICAL_LINE_WIDTH
        assert top_horizontal[0].get_linewidths()[0] == module.DEFAULT_VERTICAL_LINE_WIDTH
        assert lower_horizontal[0].get_zorder() > 2.5
        assert top_horizontal[0].get_zorder() > 2.5

        assert all(
            segment[:, 1].min() == 0
            for collection in lower_lines
            for segment in collection.get_segments()
        )
        assert np.isclose(_line_y(lower_horizontal[0]), 0.0)
        lower_horizontal_segment = lower_horizontal[0].get_segments()[0]
        lower_positions = np.array([_line_x(collection) for collection in lower_lines])
        assert np.allclose(
            np.sort(lower_horizontal_segment[:, 0]),
            [lower_positions.min() - 7.5, lower_positions.max() + 7.5],
        )

        assert ax.get_ylim()[1] >= regional_max + intensities[selected_indices].max()
        assert all(
            segment[:, 1].min() > regional_max
            for collection in top_lines
            for segment in collection.get_segments()
        )
        top_line_tops = np.array([
            segment[:, 1].max()
            for collection in top_lines
            for segment in collection.get_segments()
        ])
        assert np.allclose(top_line_tops, top_line_tops[0])
        top_line_bottoms = np.array([
            segment[:, 1].min()
            for collection in top_lines
            for segment in collection.get_segments()
        ])
        assert np.allclose(
            top_markers[0].get_offsets()[:, 0],
            np.array([_line_x(collection) for collection in top_lines]),
        )
        assert np.allclose(
            top_markers[0].get_offsets()[:, 1],
            top_line_bottoms,
        )
        assert np.isclose(_line_y(top_horizontal[0]), top_line_tops[0])
        top_horizontal_segment = top_horizontal[0].get_segments()[0]
        top_positions = np.array([_line_x(collection) for collection in top_lines])
        assert np.allclose(
            np.sort(top_horizontal_segment[:, 0]),
            [top_positions.min() - 7.5, top_positions.max() + 7.5],
        )
        assert ax.get_ylim()[1] > top_line_tops.max()

        lower_lengths = np.array([
            np.ptp(collection.get_segments()[0][:, 1])
            for collection in lower_lines
        ])
        top_lengths = np.array([
            np.ptp(collection.get_segments()[0][:, 1])
            for collection in top_lines
        ])
        assert np.allclose(top_lengths, lower_lengths)
        assert np.all(np.diff(top_positions) >= module.DEFAULT_TOP_LINE_MIN_GAP_NM)

        with Image.open(output_path) as image:
            assert image.info["dpi"][0] > 500
            assert image.info["dpi"][1] > 500
            assert image.width > 3000
            assert image.height > 2000
        module.plt.close(fig)
