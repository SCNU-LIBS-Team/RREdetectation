from pathlib import Path
import importlib.util
import sys
import tempfile

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image


FIGURE_DIR = Path(__file__).resolve().parent
MODULE_PATH = FIGURE_DIR / "plot_figure1_normalized_comb.py"
SOURCE_MODULE_PATH = FIGURE_DIR / "plot_figure1.py"
DATA_PATH = FIGURE_DIR / "data3.csv"


def load_module():
    assert MODULE_PATH.exists(), "plot_figure1_normalized_comb.py must exist"
    spec = importlib.util.spec_from_file_location(
        "plot_figure1_normalized_comb", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_source_module():
    assert SOURCE_MODULE_PATH.exists(), "plot_figure1.py must exist"
    spec = importlib.util.spec_from_file_location("plot_figure1_source", SOURCE_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def line_collections(ax):
    return [
        collection
        for collection in ax.collections
        if isinstance(collection, LineCollection)
    ]


def line_paths(ax):
    return [line for line in ax.lines if isinstance(line, Line2D)]


def is_vertical(collection):
    segment = collection.get_segments()[0]
    return np.isclose(np.ptp(segment[:, 0]), 0.0)


def segment(collection):
    return collection.get_segments()[0]


def test_new_comb_uses_current_figure1_settings_and_square_canvas():
    module = load_module()
    source = load_source_module()

    assert module.FIGSIZE == (4, 4)
    assert module.DEFAULT_TOP_LINE_COLOR == source.DEFAULT_TOP_LINE_COLOR
    assert module.DEFAULT_VERTICAL_LINE_COLOR == source.DEFAULT_VERTICAL_LINE_COLOR
    assert module.DEFAULT_TOP_LINE_RANDOM_SEED == source.DEFAULT_TOP_LINE_RANDOM_SEED
    assert module.DEFAULT_TOP_LINE_X_JITTER_NM == source.DEFAULT_TOP_LINE_X_JITTER_NM
    assert module.DEFAULT_TOP_LINE_MIN_GAP_NM == source.DEFAULT_TOP_LINE_MIN_GAP_NM
    assert module.DEFAULT_COMB_JOINSTYLE == "bevel"
    assert module.DEFAULT_COMB_CAPSTYLE == "butt"
    assert module.TOP_LABEL == "Theoratical normalized comb"
    assert module.BOTTOM_LABEL == "Expentional normalized comb"


def test_comb_geometry_is_normalized_aligned_and_keeps_correspondence():
    module = load_module()
    wavelengths, intensities, peak_indices = module.load_current_selection(DATA_PATH)

    geometry = module.build_normalized_comb_geometry(
        wavelengths,
        intensities,
        peak_indices,
    )

    assert geometry.lower_positions.size == 6
    assert geometry.upper_positions.size == 6
    assert np.isclose(np.max(geometry.lower_lengths), 1.0)
    assert np.isclose(np.max(geometry.upper_lengths), 1.0)
    assert np.allclose(geometry.lower_lengths, geometry.upper_lengths)
    assert np.all(np.diff(geometry.upper_positions) >= module.DEFAULT_TOP_LINE_MIN_GAP_NM)

    upper_tops = geometry.upper_top_y * np.ones_like(geometry.upper_lengths)
    assert np.allclose(upper_tops, geometry.upper_top_y)
    assert np.all(geometry.upper_top_y - geometry.upper_lengths > geometry.lower_base_y)

    assert np.allclose(
        geometry.lower_horizontal_limits,
        [geometry.lower_positions.min(), geometry.lower_positions.max()],
    )
    assert np.allclose(
        geometry.upper_horizontal_limits,
        [geometry.upper_positions.min(), geometry.upper_positions.max()],
    )


def test_comb_plot_uses_continuous_beveled_endpoint_paths():
    module = load_module()

    with tempfile.TemporaryDirectory() as temporary_directory:
        output_path = Path(temporary_directory) / "figure1_normalized_comb.png"
        fig, ax, geometry = module.plot_normalized_comb(
            data_path=DATA_PATH,
            output_path=output_path,
            show=False,
        )

        assert output_path.exists()
        assert ax.axison is False
        assert ax.get_title() == ""
        assert ax.get_xlabel() == ""
        assert ax.get_ylabel() == ""
        assert {text.get_text() for text in ax.texts} == {
            module.TOP_LABEL,
            module.BOTTOM_LABEL,
        }

        collections = line_collections(ax)
        vertical_lines = [collection for collection in collections if is_vertical(collection)]
        assert len(vertical_lines) == 8

        paths = line_paths(ax)
        assert len(paths) == 2
        assert all(
            line.get_solid_joinstyle() == module.DEFAULT_COMB_JOINSTYLE
            for line in paths
        )
        assert all(
            line.get_solid_capstyle() == module.DEFAULT_COMB_CAPSTYLE
            for line in paths
        )
        assert all(
            np.isclose(line.get_linewidth(), module.DEFAULT_VERTICAL_LINE_WIDTH)
            for line in paths
        )
        assert all(
            np.isclose(line.get_alpha(), module.DEFAULT_VERTICAL_LINE_ALPHA)
            for line in paths
        )
        assert all(line.get_zorder() > 2 for line in paths)

        lower_path = next(
            line
            for line in paths
            if np.isclose(line.get_xydata()[:, 1].min(), geometry.lower_base_y)
            and line.get_color() == module.DEFAULT_VERTICAL_LINE_COLOR
        )
        upper_path = next(

            line
            for line in paths
            if np.isclose(line.get_xydata()[:, 1].max(), geometry.upper_top_y)
        )

        expected_lower_path = np.array(
            [
                [geometry.lower_positions[0], geometry.lower_lengths[0]],
                [geometry.lower_positions[0], geometry.lower_base_y],
                [geometry.lower_positions[-1], geometry.lower_base_y],
                [geometry.lower_positions[-1], geometry.lower_lengths[-1]],
            ]
        )
        expected_upper_path = np.array(
            [
                [geometry.upper_positions[0], geometry.upper_top_y - geometry.upper_lengths[0]],
                [geometry.upper_positions[0], geometry.upper_top_y],
                [geometry.upper_positions[-1], geometry.upper_top_y],
                [geometry.upper_positions[-1], geometry.upper_top_y - geometry.upper_lengths[-1]],
            ]
        )
        assert np.allclose(lower_path.get_xydata(), expected_lower_path)
        assert np.allclose(upper_path.get_xydata(), expected_upper_path)
        assert lower_path.get_color() == module.DEFAULT_VERTICAL_LINE_COLOR
        assert upper_path.get_color() == module.DEFAULT_TOP_LINE_COLOR

        # The endpoint paths replace the old horizontal LineCollections.
        assert all(collection.get_segments() for collection in vertical_lines)

        with Image.open(output_path) as image:
            assert image.width == image.height
            assert image.width >= 4000
            assert image.height >= 4000
        module.plt.close(fig)


def test_comb_has_full_figure_border_matching_figure1_spine():
    module = load_module()
    source = load_source_module()

    assert module.DEFAULT_BORDER_COLOR == "black"
    assert np.isclose(module.DEFAULT_BORDER_LINEWIDTH, source.SPINE_WIDTH)

    with tempfile.TemporaryDirectory() as temporary_directory:
        output_path = Path(temporary_directory) / "figure1_normalized_comb.png"
        fig, ax, _ = module.plot_normalized_comb(
            data_path=DATA_PATH,
            output_path=output_path,
            show=False,
        )

        figure_artists = list(getattr(fig, "artists", [])) + list(
            getattr(fig, "patches", [])
        )
        borders = [artist for artist in figure_artists if isinstance(artist, Rectangle)]
        assert len(borders) == 1
        border = borders[0]

        assert border.get_visible()
        assert border.get_fill() is False
        assert border.get_edgecolor()[:3] == (0.0, 0.0, 0.0)
        assert np.isclose(border.get_linewidth(), source.SPINE_WIDTH)
        assert border.get_transform().contains_branch(fig.transFigure)
        assert np.allclose(border.get_xy(), (0.0, 0.0))
        assert np.isclose(border.get_width(), 1.0)
        assert np.isclose(border.get_height(), 1.0)
        assert border.get_zorder() > module.DEFAULT_COMB_ENDPOINT_PATH_ZORDER
        assert np.allclose(fig.get_size_inches(), module.FIGSIZE)

        module.plt.close(fig)
