from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "Paper" / "Plot" / "figure8" / "plot_multipeak_error_scatter.py"


def _load_module():
    assert MODULE_PATH.is_file(), "The standalone figure8 error plotting program is missing."
    spec = importlib.util.spec_from_file_location("figure8_error_plotter", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_finite_metric_values_excludes_summary_and_invalid_rows():
    plotter = _load_module()
    table = pd.DataFrame(
        {
            "粒子种类": ["TmII", "CeII", "PrII", "SmII", np.nan, np.nan],
            "RMSE/nm": [1.0, np.inf, 3.0, 30000.1, np.nan, 2.0],
            "强度误差比例(%)": [10.0, 20.0, np.nan, 10000.1, 15.0, np.nan],
        }
    )

    assert plotter.finite_metric_values(table, "RMSE/nm", maximum=30000.0).tolist() == [1.0, 3.0]
    assert plotter.finite_metric_values(
        table,
        "强度误差比例(%)",
        maximum=10000.0,
    ).tolist() == [10.0, 20.0]


def test_plot_metric_scatter_uses_reference_style_and_mean_line(tmpdir):
    plotter = _load_module()
    output_path = Path(str(tmpdir)) / "metric.png"

    figure, axes, mean_value = plotter.plot_metric_scatter(
        np.array([1.0, 2.0, 6.0]),
        ylabel="RMSE/nm",
        title="RMSE/nm Scatter Plot",
        output_path=output_path,
        x_tick_interval=25.0,
        y_tick_interval=2.0,
    )

    assert output_path.is_file()
    assert output_path.stat().st_size > 0
    assert np.allclose(figure.get_size_inches(), [7.0, 5.0])
    assert len(axes.collections) == 2
    assert len(axes.collections[0].get_offsets()) == 2
    assert len(axes.collections[1].get_offsets()) == 1
    assert np.allclose(axes.collections[0].get_facecolors()[0], to_rgba("tab:green", 0.85))
    assert np.allclose(axes.collections[1].get_facecolors()[0], to_rgba("tab:red", 0.85))
    assert np.allclose(axes.collections[0].get_edgecolors()[0], to_rgba("black", 0.85))
    assert np.isclose(mean_value, 3.0)
    assert np.allclose(axes.lines[0].get_ydata(), [3.0, 3.0])
    assert axes.lines[0].get_color() == "tab:orange"
    assert axes.lines[0].get_linestyle() == "--"
    assert axes.get_xlabel() == "Index"
    assert axes.get_ylabel() == "RMSE/nm"
    assert axes.get_title() == "RMSE/nm Scatter Plot"
    assert axes.spines["left"].get_linewidth() == 1.8
    assert [text.get_text() for text in axes.get_legend().get_texts()] == [
        "Mean = 3.0000",
        "Std = 2.1602",
    ]
    assert axes.get_legend().get_frame_on()
    figure.canvas.draw()
    assert np.allclose(np.diff(axes.get_xticks()), 25.0)
    assert np.allclose(np.diff(axes.get_yticks()), 2.0)
    assert len(axes.xaxis.get_minorticklocs()) == 0
    assert len(axes.yaxis.get_minorticklocs()) == 0
    plt.close(figure)


def test_generate_error_scatter_plots_writes_two_separate_images(tmpdir):
    plotter = _load_module()
    tmp_path = Path(str(tmpdir))
    input_path = tmp_path / "multipeak_fit_results.csv"
    output_dir = tmp_path / "plots"
    pd.DataFrame(
        {
            "光谱名称": [
                "a.csv",
                "b.csv",
                "outlier.csv",
                "强度误差比例平均值",
                "RMSE/nm平均值",
            ],
            "粒子种类": ["TmII", "CeII", "SmII", np.nan, np.nan],
            "RMSE/nm": [1.0, 3.0, 31000.0, np.nan, 2.0],
            "强度误差比例(%)": [10.0, 30.0, 10000.1, 20.0, np.nan],
        }
    ).to_csv(input_path, index=False, encoding="utf-8-sig")

    results = plotter.generate_error_scatter_plots(input_path, output_dir)

    assert plotter.X_TICK_INTERVAL == 25.0
    assert plotter.RMSE_Y_TICK_INTERVAL == 2500.0
    assert plotter.ERROR_RATIO_Y_TICK_INTERVAL == 1000.0
    assert plotter.MAX_INTENSITY_ERROR_RATIO == 10000.0
    assert set(results) == {"RMSE/nm", "强度误差比例(%)"}
    assert np.isclose(results["RMSE/nm"]["mean"], 2.0)
    assert np.isclose(results["强度误差比例(%)"]["mean"], 20.0)
    assert results["RMSE/nm"]["point_count"] == 2
    assert results["RMSE/nm"]["excluded_outlier_count"] == 1
    assert results["强度误差比例(%)"]["point_count"] == 2
    assert results["强度误差比例(%)"]["excluded_outlier_count"] == 1
    for result in results.values():
        image_path = result["path"]
        assert image_path.is_file()
        assert image_path.stat().st_size > 0
