"""
全球稀土储量分布饼图
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager


OUTPUT_DIR = Path(__file__).resolve().parent

COLORS = {
    "text": "#1F2933",
    "muted": "#52616B",
    "grid": "#D9E2EC",
    "canvas": "#FFFFFF",
    "china": "#D6604D",
    "brazil": "#67A961",
    "india": "#E6A23C",
    "australia": "#4F86C6",
    "russia": "#8D6FB7",
    "vietnam": "#2F9C95",
    "usa": "#F08A5D",
    "other": "#7D8790",
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
            "axes.facecolor": COLORS["canvas"],
            "text.color": COLORS["text"],
            "font.weight": "bold",
            "savefig.facecolor": COLORS["canvas"],
        }
    )


def autopct_with_threshold(total: int):
    def _format(pct: float) -> str:
        if pct < 3.0:
            return ""
        value = int(round(pct / 100 * total))
        return f"{pct:.1f}%\n({value}万吨)"

    return _format


def main() -> None:
    configure_chinese_font()

    countries = ["中国", "巴西", "印度", "澳大利亚", "俄罗斯", "越南", "美国", "其他"]
    reserves = [4400, 2100, 690, 570, 380, 350, 190, 520]  # 万吨
    total = sum(reserves)
    colors = [
        COLORS["china"],
        COLORS["brazil"],
        COLORS["india"],
        COLORS["australia"],
        COLORS["russia"],
        COLORS["vietnam"],
        COLORS["usa"],
        COLORS["other"],
    ]
    explode = [0.035, 0.015, 0.01, 0.01, 0.01, 0.01, 0.012, 0.01]

    fig, ax = plt.subplots(figsize=(12.5, 8), dpi=300)
    wedges, texts, autotexts = ax.pie(
        reserves,
        labels=countries,
        autopct=autopct_with_threshold(total),
        explode=explode,
        colors=colors,
        startangle=92,
        counterclock=True,
        pctdistance=0.66,
        labeldistance=1.08,
        textprops={"fontsize": 14, "fontweight": "bold", "color": COLORS["text"]},
        wedgeprops={"linewidth": 2.4, "edgecolor": COLORS["canvas"]},
    )

    for text in texts:
        text.set_fontsize(14)
        text.set_fontweight("bold")

    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight("bold")
        autotext.set_color("white")

    ax.set_title(
        "全球稀土储量分布\n（数据来源：USGS）",
        fontsize=24,
        fontweight="bold",
        pad=26,
        color=COLORS["text"],
    )
    ax.axis("equal")

    legend_labels = [
        f"{country}  -  {reserve}万吨（{reserve / total * 100:.1f}%）"
        for country, reserve in zip(countries, reserves)
    ]
    legend = ax.legend(
        wedges,
        legend_labels,
        title="国家 - 储量",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=13,
        title_fontsize=14,
        frameon=True,
        framealpha=1,
        facecolor="white",
        edgecolor=COLORS["grid"],
        borderpad=0.9,
        labelspacing=0.8,
        handlelength=1.3,
        handletextpad=0.8,
    )
    legend.get_title().set_fontweight("bold")
    for label in legend.get_texts():
        label.set_fontweight("bold")

    fig.tight_layout(pad=1.5)
    fig.savefig(OUTPUT_DIR / "全球稀土储量分布饼图.png", bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(f"饼图已保存至: {OUTPUT_DIR / '全球稀土储量分布饼图.png'}")


if __name__ == "__main__":
    main()
