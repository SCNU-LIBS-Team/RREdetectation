from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


DEFAULT_CSV_PATH = Path(__file__).resolve().parents[1] / "RandomSpectrum_av2" / "Temperature_Bar.csv"
DEFAULT_HEADER = None


def read_csv_with_fallback(file_path, header=None, encodings=None):
    if encodings is None:
        encodings = ["utf-8", "utf-8-sig", "gbk", "gb18030", "latin1"]

    last_error = None
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, header=header, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
        except Exception as exc:
            last_error = exc

    raise ValueError(f"无法读取 CSV 文件: {file_path}; 最后一次错误: {last_error}")


def load_abs_residuals_from_csv(csv_path, header=None):
    df = read_csv_with_fallback(csv_path, header=header)
    if df.shape[1] == 0:
        raise ValueError(f"CSV 文件没有可读取的列: {csv_path}")

    residuals = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().to_numpy(dtype=float)
    residuals = np.abs(residuals)
    if residuals.size == 0:
        raise ValueError(f"CSV 第一列没有可绘图的数值: {csv_path}")

    return residuals


def apply_plot_style(ax):
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        length=6,
        width=2.0,
        labelsize=12,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=6,
        width=2.0,
        labelsize=12,
    )
    ax.grid(alpha=0.3)

    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
    for label in ax.get_xticklabels():
        label.set_fontweight("semibold")
    for label in ax.get_yticklabels():
        label.set_fontweight("semibold")


def format_residual_stats(residuals):
    mean_value = float(np.mean(residuals))
    std_value = float(np.std(residuals))
    return f"Mean = {mean_value:.4f} K", f"Std = {std_value:.4f} K"


def plot_residual_abs_scatter(
    csv_path,
    header=None,
    show_plot=True,
    save_path=None,
    title="Temperature iteration residual plot",
):
    residuals = load_abs_residuals_from_csv(csv_path, header=header)
    x_values = np.arange(1, len(residuals) + 1)
    mean_value = float(np.mean(residuals))
    within_mean_mask = residuals <= mean_value
    above_mean_mask = residuals > mean_value

    fig, ax = plt.subplots(figsize=(7, 5))
    if np.any(within_mean_mask):
        ax.scatter(
            x_values[within_mean_mask],
            residuals[within_mean_mask],
            color="tab:green",
            s=40,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.5,
        )
    if np.any(above_mean_mask):
        ax.scatter(
            x_values[above_mean_mask],
            residuals[above_mean_mask],
            color="tab:red",
            s=38,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.5,
        )
    ax.axhline(
        mean_value,
        color="tab:orange",
        linestyle="--",
        linewidth=2,
        alpha=0.8,
    )

    apply_plot_style(ax)
    ax.set_xlabel("Index", fontsize=15, fontweight="semibold")
    ax.set_ylabel("Absolute Residual (K)", fontsize=15, fontweight="semibold")
    ax.set_title(title, fontsize=15, fontweight="semibold")
    mean_text, std_text = format_residual_stats(residuals)
    invisible_handles = [
        Line2D([], [], linestyle="None", color="none"),
        Line2D([], [], linestyle="None", color="none"),
    ]
    legend = ax.legend(
        invisible_handles,
        [mean_text, std_text],
        loc="upper right",
        bbox_to_anchor=(0.98, 0.90),
        prop={"weight": "semibold", "size": 12},
        frameon=True,
        handlelength=0,
        handletextpad=0,
    )
    for text in legend.get_texts():
        text.set_color("black")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()

    return fig, ax, x_values, residuals


def parse_args():
    parser = ArgumentParser(description="Plot absolute residual scatter from the first CSV column.")
    parser.add_argument("csv_path", nargs="?", default=DEFAULT_CSV_PATH, help="CSV 文件路径")
    header_group = parser.add_mutually_exclusive_group()
    header_group.add_argument("--header", action="store_true", help="CSV 第一行是表头时使用")
    header_group.add_argument("--no-header", action="store_true", help="CSV 没有表头时使用（默认）")
    parser.add_argument("--save", default=None, help="保存图片路径，例如 residual_scatter.png")
    parser.add_argument("--no-show", action="store_true", help="只保存或测试，不弹出图窗")
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv_path)
    header = 0 if args.header else DEFAULT_HEADER
    plot_residual_abs_scatter(
        csv_path,
        header=header,
        show_plot=not args.no_show,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
