from pathlib import Path
from html import escape

OUT_DIR = Path(__file__).resolve().parent
SVG_PATH = OUT_DIR / "温度迭代流程_优化版.svg"

W, H = 2600, 3100
FONT = "Microsoft YaHei,微软雅黑,SimSun,宋体,Noto Serif SC,sans-serif"


def attrs(**kwargs):
    return " ".join(f'{k.replace("_", "-")}="{escape(str(v), quote=True)}"' for k, v in kwargs.items())


def text_block(cx, cy, lines, size=34, weight=400, fill="#222222", line_height=None, anchor="middle", family=FONT):
    if isinstance(lines, str):
        lines = [lines]
    line_height = line_height or int(size * 1.35)
    first_y = cy - (len(lines) - 1) * line_height / 2 + size * 0.34
    parts = [
        f'<text x="{cx}" y="{first_y:.1f}" text-anchor="{anchor}" '
        f'font-family="{family}" font-size="{size}px" font-weight="{weight}" fill="{fill}" '
        f'paint-order="stroke" stroke="#ffffff" stroke-width="10" stroke-linejoin="round">'
    ]
    for i, line in enumerate(lines):
        y = first_y + i * line_height
        parts.append(f'<tspan x="{cx}" y="{y:.1f}">{escape(line)}</tspan>')
    parts.append('</text>')
    return "".join(parts)


def box(x, y, w, h, lines, *, kind="process", size=34, weight=500, rx=None, fill=None, stroke=None):
    if kind == "terminator":
        rx = rx if rx is not None else min(h / 2, 48)
        fill = fill or "#ffffff"
        stroke = stroke or "#2f2f2f"
        sw = 4
    elif kind == "process":
        rx = rx if rx is not None else 8
        fill = fill or "#f7f7f7"
        stroke = stroke or "#2f2f2f"
        sw = 4
    else:
        rx = rx if rx is not None else 8
        fill = fill or "#ffffff"
        stroke = stroke or "#2f2f2f"
        sw = 4
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" ry="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
        + text_block(x + w / 2, y + h / 2, lines, size=size, weight=weight)
    )


def diamond(cx, cy, w, h, lines, *, size=32, weight=500, fill="#ffffff", stroke="#2f2f2f"):
    pts = [
        (cx, cy - h / 2),
        (cx + w / 2, cy),
        (cx, cy + h / 2),
        (cx - w / 2, cy),
    ]
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    return (
        f'<polygon points="{points}" fill="{fill}" stroke="{stroke}" stroke-width="4" stroke-linejoin="round"/>'
        + text_block(cx, cy, lines, size=size, weight=weight, line_height=int(size * 1.25))
    )


def connector(d, *, width=4, dash=None):
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<path d="{d}" fill="none" stroke="#3a3a3a" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round" marker-end="url(#arrow)"{extra}/>'


def label(x, y, value, *, size=28, anchor="middle", weight=600):
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{FONT}" font-size="{size}px" '
        f'font-weight="{weight}" fill="#2f2f2f" paint-order="stroke" stroke="#ffffff" stroke-width="12" stroke-linejoin="round">'
        f'{escape(value)}</text>'
    )


svg = []
svg.append(f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{W}px" height="{H}px" viewBox="0 0 {W} {H}">
  <title>温度迭代识别算法流程</title>
  <desc>外层循环遍历多个温度起点，内层循环对每个温度起点迭代至收敛。</desc>
  <defs>
    <marker id="arrow" viewBox="0 0 12 12" refX="10.5" refY="6" markerWidth="12" markerHeight="12" orient="auto-start-reverse">
      <path d="M 0 0 L 12 6 L 0 12 z" fill="#3a3a3a"/>
    </marker>
    <filter id="softShadow" x="-10%" y="-10%" width="120%" height="120%">
      <feDropShadow dx="0" dy="3" stdDeviation="4" flood-color="#000000" flood-opacity="0.08"/>
    </filter>
  </defs>
  <rect width="{W}" height="{H}" fill="#ffffff"/>
''')

# Title and an outer-loop frame.
svg.append(text_block(W / 2, 60, "温度迭代识别算法流程", size=50, weight=700, fill="#1f1f1f"))
svg.append(text_block(W / 2, 103, "外层遍历多个温度起点；内层对每个起点迭代至温度收敛", size=27, weight=400, fill="#666666"))
svg.append(
    f'<rect x="40" y="125" width="2520" height="2830" rx="30" ry="30" fill="none" stroke="#8a8a8a" stroke-width="4" stroke-dasharray="18 14"/>'
)
svg.append(f'<rect x="95" y="135" width="590" height="55" rx="14" fill="#ffffff"/>')
svg.append(text_block(115, 171, "外层循环：遍历多个温度起点", size=28, weight=700, fill="#595959", anchor="start"))

# Outer-level nodes.
svg.append(box(1060, 210, 480, 90, ["外层温度迭代"], kind="terminator", size=36, weight=650))
svg.append(box(940, 360, 720, 125, ["生成多个温度起点", "（5000–20000 K，均匀采样）"], size=34, weight=500))
svg.append(box(980, 520, 640, 100, ["初始化全局最优温度与评分", "（T_best，S_best）"], size=32, weight=500))
svg.append(diamond(1300, 750, 760, 210, ["是否还有未遍历的", "温度起点？"], size=34, weight=550))
svg.append(box(980, 930, 640, 100, ["选取当前未遍历的温度起点"], size=34, weight=550))
svg.append(box(80, 695, 700, 110, ["输出全局最优温度与评分", "结束"], kind="terminator", size=32, weight=600))

# Outer-level connectors.
svg.append(connector("M 1300 300 L 1300 360"))
svg.append(connector("M 1300 485 L 1300 520"))
svg.append(connector("M 1300 620 L 1300 645"))
svg.append(connector("M 1300 855 L 1300 930"))
svg.append(label(1348, 898, "是", size=28, anchor="start"))
svg.append(connector("M 920 750 L 780 750"))
svg.append(label(850, 718, "否", size=28))

# Inner-loop module.
svg.append(
    f'<rect x="360" y="1080" width="1880" height="1260" rx="26" ry="26" fill="#f8f8f8" stroke="#777777" stroke-width="5"/>'
)
svg.append(
    f'<rect x="360" y="1080" width="1880" height="94" rx="26" ry="26" fill="#e9e9e9" stroke="#777777" stroke-width="5"/>'
)
svg.append(f'<rect x="360" y="1140" width="1880" height="34" fill="#e9e9e9"/>')
svg.append(text_block(1300, 1126, "内层循环：对当前温度起点迭代至收敛", size=34, weight=700, fill="#3d3d3d"))

svg.append(box(1000, 1195, 600, 80, ["输入当前温度起点"], size=31, weight=550, fill="#ffffff"))
svg.append(box(950, 1320, 700, 90, ["获取基体元素谱线库"], size=32, weight=500, fill="#ffffff"))
svg.append(box(900, 1450, 800, 100, ["小波寻峰检测", "并输出峰值置信度"], size=32, weight=500, fill="#ffffff"))
svg.append(box(830, 1600, 940, 110, ["选取 TOP3 峰值", "Softmax 加权获得目标温度"], size=32, weight=500, fill="#ffffff"))
svg.append(box(980, 1770, 640, 90, ["阻尼更新当前温度"], size=32, weight=500, fill="#ffffff"))
svg.append(diamond(1300, 2040, 780, 200, ["温度是否收敛？", "连续两轮 |ΔT| < tolerance"], size=31, weight=550))
svg.append(box(900, 2190, 800, 90, ["输出当前起点最终温度与评分"], kind="terminator", size=31, weight=600))

# Inner-loop connectors.
svg.append(connector("M 1300 1030 L 1300 1080"))
svg.append(connector("M 1300 1174 L 1300 1195"))
svg.append(connector("M 1300 1275 L 1300 1320"))
svg.append(connector("M 1300 1410 L 1300 1450"))
svg.append(connector("M 1300 1550 L 1300 1600"))
svg.append(connector("M 1300 1710 L 1300 1770"))
svg.append(connector("M 1300 1860 L 1300 1940"))
svg.append(connector("M 1300 2140 L 1300 2190"))
svg.append(label(1340, 2171, "是", size=27, anchor="start"))
# Inner-loop return path: no -> repeat peak finding.
svg.append(connector("M 1690 2040 L 1950 2040 L 1950 1500 L 1700 1500"))
svg.append(label(1840, 2008, "否", size=27))
svg.append(label(1980, 1780, "返回重新寻峰", size=25, anchor="start", weight=500))

# Result comparison and update/keep branches.
svg.append(diamond(1300, 2480, 780, 200, ["当前结果是否优于", "全局最优记录？"], size=32, weight=550))
svg.append(box(540, 2640, 550, 110, ["更新全局最优", "温度与评分"], size=31, weight=600, fill="#f1f1f1"))
svg.append(box(1510, 2640, 550, 110, ["保持全局最优记录"], size=31, weight=600, fill="#f1f1f1"))
svg.append(box(1000, 2820, 600, 90, ["完成当前温度起点评估"], size=31, weight=550, fill="#ffffff"))

svg.append(connector("M 1300 2280 L 1300 2380"))
svg.append(connector("M 910 2480 L 815 2480 L 815 2640"))
svg.append(label(880, 2448, "是", size=27))
svg.append(connector("M 1690 2480 L 1785 2480 L 1785 2640"))
svg.append(label(1720, 2448, "否", size=27))
svg.append(connector("M 815 2750 L 815 2865 L 1000 2865"))
svg.append(connector("M 1785 2750 L 1785 2865 L 1600 2865"))

# Outer-loop return path. It stays outside the inner module and re-enters the outer decision.
svg.append(connector("M 1300 2910 L 2360 2910 L 2360 750 L 1680 750"))
svg.append(label(2300, 915, "下一温度起点", size=26, anchor="end", weight=500))

# A few concise annotations to make the hierarchy explicit without adding nodes.

svg.append('</svg>')
SVG_PATH.write_text("\n".join(svg), encoding="utf-8")
print(SVG_PATH)
