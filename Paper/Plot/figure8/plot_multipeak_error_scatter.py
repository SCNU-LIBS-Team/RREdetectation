"""Plot figure8 multi-peak fitting errors as two standalone scatter figures.

The visual style follows ``error_evaluation.py``: 7 x 5 inch figures,
semibold labels, inward ticks on all four sides, thick axes, a light grid,
and a frameless legend.  Each point represents one retained fitting record.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = SCRIPT_DIR / "evaluation_results" / "multipeak_fit_results.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "evaluation_results" / "plots"
MAX_RMSE_PER_NM = 30000.0
MAX_INTENSITY_ERROR_RATIO = 10000.0
X_TICK_INTERVAL = 25.0
RMSE_Y_TICK_INTERVAL = 2500.0
ERROR_RATIO_Y_TICK_INTERVAL = 1000.0


def finite_metric_values(
    table: pd.DataFrame,
    metric_column: str,
    maximum: float | None = None,
) -> np.ndarray:
    """Return finite detail values, optionally excluding values above a limit."""
    required_columns = {"粒子种类", metric_column}
    missing_columns = required_columns.difference(table.columns)
    if missing_columns:
        raise ValueError("CSV 缺少必要列：" + ", ".join(sorted(missing_columns)))

    detail_rows = table.loc[table["粒子种类"].notna()]
    numeric_values = pd.to_numeric(detail_rows[metric_column], errors="coerce")
    values = numeric_values.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if maximum is not None:
        values = values[values <= float(maximum)]
    return values


def plot_metric_scatter(
    values: np.ndarray,
    ylabel: str,
    title: str,
    output_path: Path,
    x_tick_interval: float,
    y_tick_interval: float,
):
    """Create one styled scatter plot and return its figure, axes, and mean."""
    finite_values = np.asarray(values, dtype=float).reshape(-1)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        raise ValueError(f"{ylabel} 没有有限的明细数据，无法绘图。")

    mean_value = float(np.mean(finite_values))
    std_value = float(np.std(finite_values))
    record_index = np.arange(1, finite_values.size + 1)
    within_mean_mask = finite_values <= mean_value
    above_mean_mask = finite_values > mean_value

    figure, axes = plt.subplots(figsize=(7, 5))
    if np.any(within_mean_mask):
        axes.scatter(
            record_index[within_mean_mask],
            finite_values[within_mean_mask],
            color="tab:green",
            s=40,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.5,
        )
    if np.any(above_mean_mask):
        axes.scatter(
            record_index[above_mean_mask],
            finite_values[above_mean_mask],
            color="tab:red",
            s=38,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.5,
        )
    axes.axhline(
        mean_value,
        color="tab:orange",
        linestyle="--",
        linewidth=2,
        alpha=0.8,
    )

    for spine in axes.spines.values():
        spine.set_linewidth(1.8)

    axes.set_xlabel("Index", fontsize=15, fontweight="semibold")
    axes.set_ylabel(ylabel, fontsize=15, fontweight="semibold")
    axes.set_title(title, fontsize=15, fontweight="semibold")
    axes.xaxis.set_major_locator(MultipleLocator(x_tick_interval))
    axes.yaxis.set_major_locator(MultipleLocator(y_tick_interval))
    axes.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        width=2.0,
        length=6,
        labelsize=12,
    )
    for label in axes.get_xticklabels():
        label.set_fontweight("semibold")
    for label in axes.get_yticklabels():
        label.set_fontweight("semibold")

    axes.grid(alpha=0.3)
    invisible_handles = [
        Line2D([], [], linestyle="None", color="none"),
        Line2D([], [], linestyle="None", color="none"),
    ]
    legend = axes.legend(
        invisible_handles,
        [f"Mean = {mean_value:.4f}", f"Std = {std_value:.4f}"],
        loc="upper right",
        bbox_to_anchor=(0.98, 0.90),
        prop={"weight": "semibold", "size": 12},
        frameon=True,
        handlelength=0,
        handletextpad=0,
    )
    for text in legend.get_texts():
        text.set_color("black")
    figure.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    return figure, axes, mean_value


def generate_error_scatter_plots(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, dict[str, object]]:
    """Read the combined evaluation CSV and save two separate PNG figures."""
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"找不到多峰拟合评价结果：{input_path}")

    table = pd.read_csv(input_path, encoding="utf-8-sig")
    plot_specs = (
        (
            "RMSE/nm",
            "RMSE/nm",
            "RMSE/nm Scatter Plot",
            "rmse_per_nm_scatter.png",
            MAX_RMSE_PER_NM,
            RMSE_Y_TICK_INTERVAL,
        ),
        (
            "强度误差比例(%)",
            "Intensity Error Ratio (%)",
            "Intensity Error Ratio Scatter Plot",
            "intensity_error_ratio_scatter.png",
            MAX_INTENSITY_ERROR_RATIO,
            ERROR_RATIO_Y_TICK_INTERVAL,
        ),
    )

    results: dict[str, dict[str, object]] = {}
    for metric_column, ylabel, title, filename, maximum, y_tick_interval in plot_specs:
        all_finite_values = finite_metric_values(table, metric_column)
        values = finite_metric_values(table, metric_column, maximum=maximum)
        output_path = output_dir / filename
        figure, _axes, mean_value = plot_metric_scatter(
            values,
            ylabel=ylabel,
            title=title,
            output_path=output_path,
            x_tick_interval=X_TICK_INTERVAL,
            y_tick_interval=y_tick_interval,
        )
        plt.close(figure)
        results[metric_column] = {
            "path": output_path,
            "mean": mean_value,
            "std": float(np.std(values)),
            "point_count": int(values.size),
            "excluded_outlier_count": int(all_finite_values.size - values.size),
            "maximum_included": float(maximum),
        }

    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="绘制 figure8 多峰拟合的 RMSE/nm 和强度误差比例散点图。",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="合并评价 CSV；默认读取 evaluation_results/multipeak_fit_results.csv。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="图片输出目录；默认是 evaluation_results/plots。",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    results = generate_error_scatter_plots(args.input, args.output_dir)
    for metric, result in results.items():
        print(
            f"[{metric}] 点数={result['point_count']}, "
            f"排除离群值={result['excluded_outlier_count']}, "
            f"平均值={result['mean']:.12g}, 图片={result['path']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
