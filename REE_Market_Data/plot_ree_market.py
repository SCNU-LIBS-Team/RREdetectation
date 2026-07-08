"""
稀土元素市场数据可视化 - 柱状图
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


OUTPUT_DIR = Path(__file__).resolve().parent

COLORS = {
    "text": "#1F2933",
    "muted": "#52616B",
    "grid": "#D9E2EC",
    "axis": "#AAB6C4",
    "panel": "#FFFFFF",
    "canvas": "#FFFFFF",
    "blue": "#4F86C6",
    "sky": "#72A7C9",
    "teal": "#2F9C95",
    "green": "#67A961",
    "amber": "#E6A23C",
    "coral": "#D6604D",
    "violet": "#8D6FB7",
    "slate": "#7D8790",
}


def configure_chinese_font() -> None:
    """Prefer Microsoft YaHei/SimHei so Chinese labels render cleanly."""
    font_candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]

    for font_path in font_candidates:
        path = Path(font_path)
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(path)).get_name()
            break

    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "figure.facecolor": COLORS["canvas"],
            "axes.facecolor": COLORS["panel"],
            "axes.edgecolor": COLORS["axis"],
            "axes.labelcolor": COLORS["text"],
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "text.color": COLORS["text"],
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "font.weight": "bold",
            "savefig.facecolor": COLORS["canvas"],
        }
    )


def style_axis(ax, *, grid_axis: str = "y") -> None:
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=COLORS["grid"], linewidth=1.1, alpha=0.95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["axis"])
    ax.spines["bottom"].set_color(COLORS["axis"])
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis="both", labelsize=13, length=0, pad=7)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")


def annotate_vertical_bars(ax, bars, values, *, offset: float, fmt: str = "{:.1f}") -> None:
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color=COLORS["text"],
        )


def annotate_horizontal_bars(ax, bars, values, *, offset: float = 0.45) -> None:
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{value:g}%",
            ha="left",
            va="center",
            fontsize=13,
            fontweight="bold",
            color=COLORS["text"],
        )


def draw_market_size(ax) -> None:
    years_market = ["2022", "2023", "2024", "2025", "2030(预测)"]
    market_size = [64.2, 83.1, 64.7, 90.3, 134.9]  # 亿美元(取中间值)
    market_colors = [
        COLORS["sky"],
        COLORS["blue"],
        "#6FB0B4",
        COLORS["green"],
        COLORS["amber"],
    ]

    bars = ax.bar(
        years_market,
        market_size,
        width=0.62,
        color=market_colors,
        edgecolor="white",
        linewidth=2,
    )

    ax.set_title("全球稀土金属市场规模", fontsize=19, pad=15, fontweight="bold")
    ax.set_xlabel("年份", fontsize=15, labelpad=10, fontweight="bold")
    ax.set_ylabel("市场规模（亿美元）", fontsize=15, labelpad=10, fontweight="bold")
    ax.set_ylim(0, 155)
    ax.set_yticks(np.arange(0, 151, 30))
    ax.margins(x=0.04)
    annotate_vertical_bars(ax, bars, market_size, offset=3.0)
    style_axis(ax, grid_axis="y")


def draw_production(ax) -> None:
    countries = ["中国", "美国", "缅甸", "澳大利亚", "泰国", "其他"]
    production_2022 = [21.0, 4.3, 1.2, 1.8, 0.7, 1.0]  # 万吨REO
    colors = [
        COLORS["coral"],
        COLORS["blue"],
        COLORS["green"],
        COLORS["amber"],
        COLORS["violet"],
        COLORS["slate"],
    ]

    bars = ax.bar(
        countries,
        production_2022,
        width=0.62,
        color=colors,
        edgecolor="white",
        linewidth=2,
    )

    ax.set_title("2022年全球稀土产量分布", fontsize=19, pad=15, fontweight="bold")
    ax.set_xlabel("国家/地区", fontsize=15, labelpad=10, fontweight="bold")
    ax.set_ylabel("产量（万吨REO）", fontsize=15, labelpad=10, fontweight="bold")
    ax.set_ylim(0, 24)
    ax.set_yticks(np.arange(0, 25, 4))
    annotate_vertical_bars(ax, bars, production_2022, offset=0.45)
    style_axis(ax, grid_axis="y")


def draw_price_comparison(ax) -> None:
    elements = [
        r"La$_2$O$_3$",
        r"CeO$_2$",
        r"Nd$_2$O$_3$",
        r"Pr$_6$O$_{11}$",
        r"Dy$_2$O$_3$",
        r"Tb$_4$O$_7$",
    ]
    price_2020 = [2.5, 1.75, 45, 45, 275, 700]  # 美元/kg 取中间值
    price_2022 = [4, 3, 80, 75, 330, 1200]

    x = np.arange(len(elements))
    width = 0.34

    ax.bar(
        x - width / 2,
        price_2020,
        width,
        label="2020年",
        color=COLORS["blue"],
        edgecolor="white",
        linewidth=1.5,
    )
    ax.bar(
        x + width / 2,
        price_2022,
        width,
        label="2022年",
        color=COLORS["coral"],
        edgecolor="white",
        linewidth=1.5,
    )

    ax.set_title("主要稀土氧化物价格对比", fontsize=19, pad=15, fontweight="bold")
    ax.set_xlabel("稀土氧化物", fontsize=15, labelpad=10, fontweight="bold")
    ax.set_ylabel("价格（美元/kg）", fontsize=15, labelpad=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(elements, fontsize=13)
    ax.set_yscale("log")
    ax.legend(
        loc="upper left",
        frameon=True,
        framealpha=1,
        facecolor="white",
        edgecolor=COLORS["grid"],
        fontsize=13,
    )
    style_axis(ax, grid_axis="y")


def draw_applications(ax) -> None:
    applications = ["永磁材料", "催化剂", "抛光粉", "玻璃添加剂", "冶金合金", "荧光粉", "陶瓷", "其他"]
    percentages = [37.5, 22.5, 11, 7.5, 9, 4, 4, 6.5]
    colors = [
        COLORS["coral"],
        COLORS["blue"],
        COLORS["green"],
        COLORS["amber"],
        COLORS["violet"],
        COLORS["teal"],
        "#F08A5D",
        COLORS["slate"],
    ]

    bars = ax.barh(
        applications[::-1],
        percentages[::-1],
        color=colors[::-1],
        edgecolor="white",
        linewidth=2,
        height=0.72,
    )

    ax.set_title("稀土下游应用市场分布（2022年）", fontsize=19, pad=15, fontweight="bold")
    ax.set_xlabel("市场占比（%）", fontsize=15, labelpad=10, fontweight="bold")
    ax.set_xlim(0, 42)
    ax.set_xticks(np.arange(0, 41, 5))
    annotate_horizontal_bars(ax, bars, percentages[::-1])
    style_axis(ax, grid_axis="x")


def save_standalone_market_chart() -> None:
    fig, ax = plt.subplots(figsize=(10.8, 5.8), dpi=300)
    draw_market_size(ax)
    fig.tight_layout(pad=1.2)
    fig.savefig(OUTPUT_DIR / "全球稀土金属市场规模柱状图.png", bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def main() -> None:
    configure_chinese_font()

    fig, axes = plt.subplots(2, 2, figsize=(18, 13), dpi=300)
    fig.suptitle("稀土元素（REE）市场数据概览", fontsize=27, fontweight="bold", y=0.985)

    draw_market_size(axes[0, 0])
    draw_production(axes[0, 1])
    draw_price_comparison(axes[1, 0])
    draw_applications(axes[1, 1])

    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=3.0, w_pad=2.8)
    fig.savefig(OUTPUT_DIR / "稀土市场数据柱状图.png", bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)

    save_standalone_market_chart()
    print(f"图表已保存至: {OUTPUT_DIR / '稀土市场数据柱状图.png'}")
    print(f"单图已保存至: {OUTPUT_DIR / '全球稀土金属市场规模柱状图.png'}")


if __name__ == "__main__":
    main()
