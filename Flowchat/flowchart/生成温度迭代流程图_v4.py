# -*- coding: utf-8 -*-
"""生成第四版温度迭代识别算法流程图。

本图严格对应 Elements_detectation.py 中 T_iteration / T_iteration_single 的实际逻辑：
- 外层遍历 10 个温度起点；
- 内层每一轮都按当前温度更新基体元素谱线库；
- Top-3 按候选元素置信度排序，再进行 Softmax 加权；
- 收敛判据为相对温度变化率连续两轮低于 tolerance；
- 达到 max_iterations 上限时同样输出当前起点结果。
"""

from html import escape
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
SVG_PATH = OUT_DIR / "温度迭代流程_优化版_v4.svg"

# 采用横向双模块版式，降低纵向压缩时的文字缩小程度。
W, H = 3600, 3200
FONT = "Microsoft YaHei,微软雅黑,SimSun,宋体,Noto Serif SC,sans-serif"
STROKE = "#2f2f2f"
LINE = "#3b3b3b"
TEXT = "#202020"
MUTED = "#5b5b5b"
LIGHT = "#f7f7f7"
GROUP = "#8b8b8b"


def text_block(
    cx,
    cy,
    lines,
    *,
    size=42,
    weight=500,
    fill=TEXT,
    line_height=None,
    anchor="middle",
    stroke_width=None,
):
    """输出居中的多行 SVG 文本，所有文本均通过 UTF-8 写入。"""
    if isinstance(lines, str):
        lines = [lines]
    line_height = line_height or int(size * 1.28)
    stroke_width = stroke_width if stroke_width is not None else max(6, int(size * 0.20))
    first_y = cy - (len(lines) - 1) * line_height / 2 + size * 0.34
    parts = [
        f'<text x="{cx:.1f}" y="{first_y:.1f}" text-anchor="{anchor}" '
        f'font-family="{FONT}" font-size="{size}px" font-weight="{weight}" fill="{fill}" '
        f'paint-order="stroke" stroke="#ffffff" stroke-width="{stroke_width}" stroke-linejoin="round">'
    ]
    for index, line in enumerate(lines):
        y = first_y + index * line_height
        parts.append(f'<tspan x="{cx:.1f}" y="{y:.1f}">{escape(line)}</tspan>')
    parts.append("</text>")
    return "".join(parts)


def label(x, y, value, *, size=32, anchor="middle", weight=650):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="{FONT}" font-size="{size}px" font-weight="{weight}" fill="{TEXT}" '
        f'paint-order="stroke" stroke="#ffffff" stroke-width="12" stroke-linejoin="round">'
        f"{escape(value)}</text>"
    )


def box(
    x,
    y,
    width,
    height,
    lines,
    *,
    kind="process",
    size=42,
    weight=520,
    fill=None,
    rx=None,
):
    if kind == "terminator":
        rx = rx if rx is not None else min(height / 2, 54)
        fill = fill or "#ffffff"
    else:
        rx = rx if rx is not None else 12
        fill = fill or LIGHT
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{rx}" ry="{rx}" '
        f'fill="{fill}" stroke="{STROKE}" stroke-width="5"/>'
        + text_block(x + width / 2, y + height / 2, lines, size=size, weight=weight)
    )


def diamond(cx, cy, width, height, lines, *, size=38, weight=540):
    points = " ".join(
        f"{x:.1f},{y:.1f}"
        for x, y in (
            (cx, cy - height / 2),
            (cx + width / 2, cy),
            (cx, cy + height / 2),
            (cx - width / 2, cy),
        )
    )
    return (
        f'<polygon points="{points}" fill="#ffffff" stroke="{STROKE}" stroke-width="5" '
        f'stroke-linejoin="round"/>'
        + text_block(cx, cy, lines, size=size, weight=weight, line_height=int(size * 1.22))
    )


def connector(path, *, width=5, dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<path d="{path}" fill="none" stroke="{LINE}" stroke-width="{width}" '
        f'stroke-linecap="round" stroke-linejoin="round" marker-end="url(#arrow)"{dash_attr}/>'
    )


svg = [
    f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{W}px" height="{H}px" viewBox="0 0 {W} {H}">
  <title>温度迭代识别算法流程（第四版）</title>
  <desc>外层遍历十个温度起点，内层在每轮迭代中更新谱线库并通过候选元素置信度计算目标温度，直到收敛或达到最大迭代次数。</desc>
  <defs>
    <marker id="arrow" viewBox="0 0 14 14" refX="12" refY="7" markerWidth="14" markerHeight="14" orient="auto">
      <path d="M 0 0 L 14 7 L 0 14 z" fill="{LINE}"/>
    </marker>
  </defs>
  <rect width="{W}" height="{H}" fill="#ffffff"/>
''',
]

# 论文图标题与分组框。标题、副标题、模块标题之间保留足够留白。
svg.append(text_block(W / 2, 74, "温度迭代识别算法流程", size=64, weight=720, fill="#161616", stroke_width=12))
svg.append(text_block(W / 2, 142, "外层遍历多个温度起点；内层对每个起点迭代更新温度并进行全局最优比较", size=38, weight=430, fill=MUTED, stroke_width=9))

outer_x, outer_y, outer_w, outer_h = 90, 230, 1370, 2840
inner_x, inner_y, inner_w, inner_h = 1540, 230, 1970, 2840

svg.append(
    f'<rect x="{outer_x}" y="{outer_y}" width="{outer_w}" height="{outer_h}" rx="30" ry="30" '
    f'fill="#ffffff" stroke="{GROUP}" stroke-width="5" stroke-dasharray="20 16"/>'
)
svg.append(
    f'<rect x="{inner_x}" y="{inner_y}" width="{inner_w}" height="{inner_h}" rx="30" ry="30" '
    f'fill="#fafafa" stroke="{GROUP}" stroke-width="5"/>'
)
svg.append(
    f'<rect x="{outer_x}" y="{outer_y}" width="{outer_w}" height="105" rx="30" ry="30" fill="#ededed" stroke="{GROUP}" stroke-width="5"/>'
)
svg.append(f'<rect x="{outer_x}" y="{outer_y + 68}" width="{outer_w}" height="37" fill="#ededed"/>')
svg.append(
    f'<rect x="{inner_x}" y="{inner_y}" width="{inner_w}" height="105" rx="30" ry="30" fill="#e4e4e4" stroke="{GROUP}" stroke-width="5"/>'
)
svg.append(f'<rect x="{inner_x}" y="{inner_y + 68}" width="{inner_w}" height="37" fill="#e4e4e4"/>')
svg.append(text_block(outer_x + outer_w / 2, outer_y + 62, "外层循环：遍历 10 个温度起点", size=40, weight=700, fill="#333333", stroke_width=8))
svg.append(text_block(inner_x + inner_w / 2, inner_y + 62, "内层循环：对当前温度起点迭代更新温度", size=40, weight=700, fill="#333333", stroke_width=8))

# 几何位置：外层流程（左列）。
outer_cx = 775
outer_start = (470, 405, 610, 105)
outer_generate = (300, 570, 950, 150)
outer_init = (355, 740, 840, 105)
outer_check = (outer_cx, 975, 930, 230)
outer_select = (400, 1195, 750, 120)
outer_end = (140, 1390, 520, 160)
outer_compare = (outer_cx, 1955, 900, 230)
outer_update = (190, 2110, 520, 145)
outer_keep = (840, 2110, 520, 145)
outer_join = (475, 2335, 600, 120)
outer_failure = (760, 1380, 600, 170)

# 几何位置：内层流程（右列）。
inner_cx = 2360
inner_input = (1870, 470, 980, 130)
inner_library = (1850, 660, 1020, 145)
inner_wavelet = (1810, 850, 1100, 145)
inner_validity = (inner_cx, 1065, 1250, 210)
inner_top3 = (1750, 1190, 1220, 210)
inner_damping = (2010, 1430, 700, 125)
inner_convergence = (inner_cx, 1650, 1350, 300)
inner_output = (1870, 1880, 980, 150)

# 连接线先绘制，使箭头不会覆盖节点边框。
connectors = [
    # 外层主干。
    connector("M 775 510 L 775 570"),
    connector("M 775 720 L 775 740"),
    connector("M 775 845 L 775 860"),
    connector("M 775 1090 L 775 1195"),
    # 外层判断：否 -> 全局输出。
    connector("M 310 975 L 165 975 L 165 1470 L 140 1470"),
    # 外层选择 -> 调用内层模块（从间隙上行进入内层输入）。
    connector("M 1150 1255 L 1490 1255 L 1490 535 L 1870 535"),
    # 内层主干。
    connector("M 2360 600 L 2360 660"),
    connector("M 2360 805 L 2360 850"),
    connector("M 2360 995 L 2360 1065"),
    connector("M 2360 1170 L 2360 1190"),
    connector("M 2360 1400 L 2360 1430"),
    connector("M 2360 1555 L 2360 1500"),
    connector("M 2360 1800 L 2360 1880"),
    # 内层判断：否 -> 根据当前 T 更新谱线库。
    connector("M 3035 1650 L 3240 1650 L 3240 732 L 2870 732"),
    # 无有效候选元素 -> 当前起点提前结束，汇入输出节点。
    # invalid-candidate branch -> dedicated failure node
    connector("M 1735 1065 L 1600 1065 L 1600 1380 L 1060 1380"),
    # failure node -> merge at evaluation-complete node, then continue outer loop
    connector("M 1060 1550 L 1060 1600 L 180 1600 L 180 2395 L 475 2395"),
    # 内层输出 -> 外层当前起点结果比较。
    connector("M 1870 1955 L 1225 1955"),
    # 外层比较分支。
    connector("M 325 1955 L 450 1955 L 450 2110"),
    connector("M 1225 1955 L 1100 1955 L 1100 2110"),
    # 更新/保持两路合流。
    connector("M 450 2255 L 450 2300 L 475 2300 L 475 2335"),
    connector("M 1100 2255 L 1100 2300 L 1075 2300 L 1075 2335"),
    # 外层回路：经最左侧专用通道返回外层判断，避开内层调用线和结果比较线。
    connector("M 775 2455 L 120 2455 L 120 350 L 1300 350 L 1300 975 L 1240 975"),
]
svg.extend(connectors)

# 外层节点。
svg.append(box(*outer_start, ["外层温度迭代"], kind="terminator", size=46, weight=700))
svg.append(box(*outer_generate, ["生成 10 个温度起点", "（5000–20000 K，均匀采样）"], size=44, weight=540))
svg.append(box(*outer_init, ["初始化全局最优温度与评分", "（T_best，S_best）"], size=41, weight=540))
svg.append(diamond(outer_cx, outer_check[1], outer_check[2], outer_check[3], ["是否还有未遍历的", "温度起点？"], size=42, weight=570))
svg.append(box(*outer_select, ["选取当前未遍历的温度起点 T₀"], size=42, weight=570))
svg.append(box(*outer_end, ["输出全局最优温度", "结束"], kind="terminator", size=40, weight=650))
svg.append(box(*outer_failure, ["当前起点评估失败", "跳过该温度起点"], size=38, weight=620, fill="#ffffff"))
svg.append(diamond(outer_cx, outer_compare[1], outer_compare[2], outer_compare[3], ["当前结果评分是否优于", "全局最优评分？"], size=40, weight=570))
svg.append(box(*outer_update, ["更新全局最优", "温度与评分"], size=39, weight=650, fill="#eeeeee"))
svg.append(box(*outer_keep, ["保持全局最优记录"], size=39, weight=650, fill="#eeeeee"))
svg.append(box(*outer_join, ["完成当前温度起点评估"], size=39, weight=570, fill="#ffffff"))

# 内层节点。
svg.append(box(*inner_input, ["输入当前温度起点 T₀"], size=42, weight=570, fill="#ffffff"))
svg.append(box(*inner_library, ["根据当前温度 T 更新", "基体元素标准谱线库"], size=40, weight=540, fill="#ffffff"))
svg.append(box(*inner_wavelet, ["小波寻峰检测", "计算候选元素置信度"], size=40, weight=540, fill="#ffffff"))
svg.append(diamond(inner_cx, inner_validity[1], inner_validity[2], inner_validity[3], ["是否获得有效", "候选元素？"], size=41, weight=570))
svg.append(box(*inner_top3, ["选取置信度最高的 Top-3 候选元素", "基于置信度与 R² 评分进行 Softmax 加权", "获得目标温度"], size=35, weight=540, fill="#ffffff", rx=12))
svg.append(box(*inner_damping, ["采用阻尼策略更新当前温度"], size=40, weight=540, fill="#ffffff"))
svg.append(diamond(inner_cx, inner_convergence[1], inner_convergence[2], inner_convergence[3], ["温度是否收敛，或", "达到最大迭代次数？", "相对温度变化率连续两轮 < tolerance"], size=36, weight=570))
svg.append(box(*inner_output, ["输出当前起点最终温度及内层最优评分"], kind="terminator", size=37, weight=650))

# 分支与回路标签。
svg.append(label(275, 940, "否", size=34))
svg.append(label(825, 1135, "是", size=34, anchor="start"))
svg.append(label(2390, 1188, "是", size=34, anchor="start"))
svg.append(label(3140, 1615, "否", size=34))
svg.append(label(1685, 1030, "否", size=34))
svg.append(label(375, 1915, "是", size=34))
svg.append(label(1175, 1915, "否", size=34))
svg.append(label(2390, 1840, "是", size=34, anchor="start"))
svg.append(label(1300, 820, "下一温度起点", size=28, anchor="middle", weight=550))

svg.append("</svg>")
SVG_PATH.write_text("\n".join(svg), encoding="utf-8")
print(SVG_PATH)
