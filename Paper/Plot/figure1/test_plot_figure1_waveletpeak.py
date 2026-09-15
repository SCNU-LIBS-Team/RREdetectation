from pathlib import Path
import importlib.util
import sys
import tempfile

import matplotlib as mpl
import numpy as np
from matplotlib import font_manager
from matplotlib.collections import PathCollection
from matplotlib.colors import to_rgba
from PIL import Image


FIGURE_DIR = Path(__file__).resolve().parent
MODULE_PATH = FIGURE_DIR / "plot_figure1_waveletpeak.py"
SOURCE_MODULE_PATH = FIGURE_DIR / "plot_figure1.py"
DATA_PATH = FIGURE_DIR / "data3.csv"


def load_module():
    assert MODULE_PATH.exists(), "plot_figure1_waveletpeak.py must exist"
    spec = importlib.util.spec_from_file_location(
        "plot_figure1_waveletpeak",
        str(MODULE_PATH),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_source_module():
    assert SOURCE_MODULE_PATH.exists(), "plot_figure1.py must exist"
    spec = importlib.util.spec_from_file_location(
        "plot_figure1_source_for_wavelet_test",
        str(SOURCE_MODULE_PATH),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_reference_plot(source):
    wavelengths, intensities = source.load_spectrum_data(DATA_PATH)
    peak_indices = source.select_manual_peak_indices(
        wavelengths,
        intensities,
        source.DEFAULT_MANUAL_PEAK_WAVELENGTHS,
    )
    return source.plot_spectrum(
        wavelengths,
        intensities,
        output_path=None,
        peak_indices=peak_indices,
        show=False,
    )


def test_import_preserves_figure1_semibold_font_resolution():
    with mpl.rc_context(rc=mpl.rcParamsDefault):
        expected_family = list(mpl.rcParams["font.family"])
        expected_sans_serif = list(mpl.rcParams["font.sans-serif"])
        expected_font_path = font_manager.findfont(
            font_manager.FontProperties(
                family=expected_family,
                weight="semibold",
            )
        )

        sys.modules.pop("Wavelet_peakfinding", None)
        module = load_module()
        wavelengths, intensities = module.load_spectrum_region(DATA_PATH)
        peak_indices, _, _ = module.detect_wavelet_peaks(wavelengths, intensities)
        fig, ax = module.plot_wavelet_peaks(
            wavelengths,
            intensities,
            peak_indices,
            output_path=None,
            show=False,
        )

        actual_font_path = font_manager.findfont(
            ax.xaxis.label.get_fontproperties()
        )
        assert list(mpl.rcParams["font.family"]) == expected_family
        assert list(mpl.rcParams["font.sans-serif"]) == expected_sans_serif
        assert Path(actual_font_path).resolve() == Path(expected_font_path).resolve()
        module.plt.close(fig)


def test_load_figure1_region_uses_current_data_and_limits():
    module = load_module()

    wavelengths, intensities = module.load_spectrum_region(DATA_PATH)

    assert module.DEFAULT_DATA_PATH == DATA_PATH
    assert module.DEFAULT_OUTPUT_PATH == FIGURE_DIR / "figure1_waveletpeak.png"
    assert module.DEFAULT_WAVELENGTH_MIN == 220.0
    assert module.DEFAULT_WAVELENGTH_MAX == 330.0
    assert wavelengths.size == intensities.size == 2254
    assert wavelengths[0] == 220.02
    assert wavelengths[-1] == 329.98
    assert intensities.max() == 323700.0
    assert np.all(np.diff(wavelengths) > 0)


def test_detect_wavelet_peaks_uses_project_parameters_and_finds_known_peaks():
    module = load_module()
    wavelengths, intensities = module.load_spectrum_region(DATA_PATH)

    peak_indices, peak_wavelengths, peak_intensities = module.detect_wavelet_peaks(
        wavelengths,
        intensities,
    )

    assert module.DEFAULT_WAVELET == "mexh"
    assert np.array_equal(module.DEFAULT_SCALES, np.arange(1, 11))
    assert module.DEFAULT_NEIGHBOR == 4
    assert module.DEFAULT_MIN_LENGTH == 3
    assert module.DEFAULT_COEFFICIENT_THRESHOLD == 700.0
    assert module.DEFAULT_CORRECTION_WINDOW == 5
    assert peak_indices.size == 62
    assert np.array_equal(peak_indices, np.unique(peak_indices))
    assert np.array_equal(peak_wavelengths, wavelengths[peak_indices])
    assert np.array_equal(peak_intensities, intensities[peak_indices])

    known_figure1_peaks = np.array(
        [234.37, 248.96, 259.90, 274.83, 282.84, 302.07]
    )
    for known_peak in known_figure1_peaks:
        assert np.any(np.isclose(peak_wavelengths, known_peak))


def test_plot_wavelet_peaks_saves_png_and_marks_every_detected_peak():
    module = load_module()
    wavelengths, intensities = module.load_spectrum_region(DATA_PATH)
    peak_indices, _, _ = module.detect_wavelet_peaks(wavelengths, intensities)

    with tempfile.TemporaryDirectory() as temporary_directory:
        output_path = Path(temporary_directory) / "wavelet_result.png"
        fig, ax = module.plot_wavelet_peaks(
            wavelengths,
            intensities,
            peak_indices,
            output_path=output_path,
            show=False,
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0
        with Image.open(str(output_path)) as image:
            assert image.format == "PNG"
            assert image.width > image.height

        peak_markers = [
            collection
            for collection in ax.collections
            if isinstance(collection, PathCollection)
        ]
        assert len(peak_markers) == 1
        assert len(peak_markers[0].get_offsets()) == peak_indices.size
        assert ax.get_xlim() == (
            module.DEFAULT_WAVELENGTH_MIN,
            module.DEFAULT_WAVELENGTH_MAX,
        )
        assert ax.get_xlabel() == "Wavelength (nm)"
        assert ax.get_ylabel() == "Intensity"
        assert ax.lines[0].get_label() == "Original Spectrum"
        assert peak_markers[0].get_label() == "Detected Peaks"
        module.plt.close(fig)


def test_plot_matches_figure1_spectrum_axes_line_and_output_style():
    module = load_module()
    source = load_source_module()
    wavelengths, intensities = module.load_spectrum_region(DATA_PATH)
    peak_indices, _, _ = module.detect_wavelet_peaks(wavelengths, intensities)

    fig, ax = module.plot_wavelet_peaks(
        wavelengths,
        intensities,
        peak_indices,
        output_path=None,
        show=False,
    )
    reference_fig, reference_ax = make_reference_plot(source)

    assert tuple(fig.get_size_inches()) == tuple(reference_fig.get_size_inches())
    assert module.SAVE_DPI == source.SAVE_DPI
    assert ax.get_xlim() == reference_ax.get_xlim()
    assert ax.get_ylim() == reference_ax.get_ylim()
    assert ax.get_title() == reference_ax.get_title() == ""
    assert reference_ax.get_legend() is None
    assert ax.xaxis.label.get_fontsize() == reference_ax.xaxis.label.get_fontsize()
    assert ax.xaxis.label.get_fontweight() == reference_ax.xaxis.label.get_fontweight()
    assert ax.yaxis.label.get_fontsize() == reference_ax.yaxis.label.get_fontsize()
    assert ax.yaxis.label.get_fontweight() == reference_ax.yaxis.label.get_fontweight()
    assert ax.lines[0].get_color() == reference_ax.lines[0].get_color()
    assert ax.lines[0].get_linewidth() == reference_ax.lines[0].get_linewidth()
    assert ax.lines[0].get_alpha() == reference_ax.lines[0].get_alpha()
    for name in ("left", "right", "bottom", "top"):
        assert ax.spines[name].get_linewidth() == reference_ax.spines[name].get_linewidth()

    module.plt.close(fig)
    source.plt.close(reference_fig)


def test_plot_adds_unframed_semibold_legend_in_upper_right_without_peak_count():
    module = load_module()
    source = load_source_module()
    wavelengths, intensities = module.load_spectrum_region(DATA_PATH)
    peak_indices, _, _ = module.detect_wavelet_peaks(wavelengths, intensities)

    fig, ax = module.plot_wavelet_peaks(
        wavelengths,
        intensities,
        peak_indices,
        output_path=None,
        show=False,
    )

    legend = ax.get_legend()
    assert legend is not None
    assert legend._loc == 1  # Matplotlib location code for upper right.
    assert not legend.get_frame_on()
    assert [text.get_text() for text in legend.get_texts()] == [
        "Original Spectrum",
        "Detected Peaks",
    ]
    assert all(
        text.get_fontsize() == source.TICK_LABELSIZE
        for text in legend.get_texts()
    )
    assert all(
        text.get_fontweight() == source.FONT_WEIGHT
        for text in legend.get_texts()
    )
    module.plt.close(fig)


def test_peak_markers_are_red_and_match_figure1_marker_size_and_opacity():
    module = load_module()
    source = load_source_module()
    wavelengths, intensities = module.load_spectrum_region(DATA_PATH)
    peak_indices, _, _ = module.detect_wavelet_peaks(wavelengths, intensities)

    fig, ax = module.plot_wavelet_peaks(
        wavelengths,
        intensities,
        peak_indices,
        output_path=None,
        show=False,
    )
    reference_fig, reference_ax = make_reference_plot(source)

    peak_marker = next(
        collection
        for collection in ax.collections
        if isinstance(collection, PathCollection)
    )
    reference_marker = next(
        collection
        for collection in reference_ax.collections
        if isinstance(collection, PathCollection)
    )

    assert np.allclose(peak_marker.get_facecolors()[0], to_rgba("#D73027"))
    assert np.allclose(peak_marker.get_edgecolors(), peak_marker.get_facecolors())
    assert np.array_equal(peak_marker.get_sizes(), reference_marker.get_sizes())
    assert peak_marker.get_alpha() == reference_marker.get_alpha()
    assert peak_marker.get_facecolors()[0, 3] == reference_marker.get_facecolors()[0, 3]

    module.plt.close(fig)
    source.plt.close(reference_fig)


def test_cli_generates_requested_output_and_reports_peak_wavelengths(
    monkeypatch,
    capsys,
):
    module = load_module()

    with tempfile.TemporaryDirectory() as temporary_directory:
        output_path = Path(temporary_directory) / "cli_wavelet.png"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "plot_figure1_waveletpeak.py",
                "--data",
                str(DATA_PATH),
                "--output",
                str(output_path),
            ],
        )

        module.main()

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    stdout = capsys.readouterr().out
    assert "Detected peaks: 62" in stdout
    assert "234.37" in stdout
    assert "302.07" in stdout
    assert "Saved wavelet peak figure to:" in stdout
