# SPEC：用 matplotlib 复现 `example1.png`（figure6 主图样式）

> Agent-1（需求/协调）产出。所有数值均来自对原图的像素级勘测，可直接使用。
> **禁止修改现有文件**：`plot_figure6.py`、`test_plot_figure6.py`、`data.csv`、`example1.png`。
> 所有新代码放在 `repro_example1/` 目录内。

## 1. 目标

用 Python(matplotlib) 从 `data.csv` 出发，1:1 复现 `D:/LIBS/RREdetectation/Paper/Plot/figure6/example1.png`
（1253×800 px）。曲线数据**不是**手绘，经定量验证原图就是由 data.csv 绘制的：

| 曲线 | 颜色 | data.csv 列 | 验证 |
|---|---|---|---|
| 蓝 | `#3366CC` | `Sum(calc)` | 大峰/肩部/次峰位置全部吻合 |
| 红 | `#DC3912` | `Eu II (8.6e-3)` | 中心线 RMS 0.67px，完全吻合 |

> **用户确认（2026-09-04）：只画蓝(SUM)和红(Eu II)两条曲线；原图中的橙色 Tm II 曲线故意不画。**
> 视口（xlim/ylim/网格/标签）仍与原图保持一致。检验时复现图中**没有**橙线才是正确的。

绘图顺序：蓝 → 红（红在上层）。
无图例、无标题、无轴标题、无 y 轴刻度标签、无 spines、无刻度短线（tick marks）。

## 2. 画布与坐标几何（实测）

- 图像尺寸：**1253 × 800 px**，建议 `figsize=(12.53, 8.0), dpi=100`。
- 坐标轴区域铺满图宽：left=0, right=1252；**y=0 基线在第 727 行**，轴顶在第 0 行。
  即 `fig.add_axes([0, 1 - 727.5/800, 1, 727.5/800])`。
- x 映射：1 nm = 200.8 px；主网格线中心 x = 197.5(380nm), 397.5, 599.5, 799.5, 1001.5, 1201.5(385nm)。
  → `xlim = (379.016, 385.254)`（左边缘恰为 379.0 附近，右边缘 385.25）。
- y 映射：`ymax = Sum(calc) 在窗口内的最大值 29440`（蓝峰顶端正好贴住/轻微削顶，已验证）；
  `ylim = (0, 29440)`。比例 40.5 数据单位/px。
- x 主刻度：379, 380, …, 385。379 的刻度在 px≈-3.6，故其标签被图边裁掉只剩 "79"（复现时自然发生）。
- y 主刻度：0, 5000, …, 25000（仅用于画网格，**不显示标签**）；本次勘测中 y 无标签。
- x 刻度标签：颜色 `#444444`，字高约 22px（含AA，行 751–772），"380" 三字符簇宽约 46px，
  字重偏 semibold（暗像素填充率 0.25）。标签顶部距轴底约 23.5px（即 xtick pad ≈ 17pt @100dpi）。
- 字体：优先尝试 Arial / Segoe UI / DejaVu Sans × (normal, semibold, bold)，用 compare 度量
  （簇宽 46±3px、高 22±2px、填充率 0.22–0.28）自动选最接近的组合，并写进 DEFAULTS。
  注意：原图文字有 ClearType 彩色边缘（浏览器渲染痕迹），**不要模仿**。
- 蓝峰顶部削顶属正常现象（ylim 上界=Sum最大值 + 线条半宽所致），复现后应自然出现，不要人为修饰。

## 3. 网格系统（实测）

| 层 | 颜色 | 线宽 | x 间隔 | y 间隔 | 实测位置 |
|---|---|---|---|---|---|
| 主网格 | `#CCCCCC` | 2px 核（lw≈1.5pt@100dpi） | 1 nm | 5000 | v: 197.5,397.5,…,1201.5；h: 111.5,233.5,357.5,481.5,603.5（y=0 线被橙线盖住） |
| 次网格 | `#EBEBEB` | 2px 核 | 0.5 nm | 2500 | v: 97,297,499,699,899,1101；h: ≈49.5,172.5,295.5,418.5,542.5,665 |

- 背景**纯白**（#EBEBEB 不是背景色，是次网格线颜色）。
- matplotlib 2.2.3 实现要点：`ax.set_xticks(major)` + `ax.set_xticks(minor, minor=True)`
  （y 同理），然后分别 `ax.grid(True, which='major', ...)` 和 `which='minor'`；
  `ax.tick_params(length=0)`，y 轴 `labelleft=False`。

## 4. 线条（实测+需标定）

- 纯色核宽约 3px（原图基线处可见 3 行纯色），含 AA 总宽 4–5px。
  matplotlib 起点：两条线 `linewidth=2.3`（@100dpi ≈ 3.2px），按 compare 指标微调。
- 标定目标：蓝线纯色像素数 6562（±12%，原图中蓝线无遮挡，可直接比）。
- 红线像素数**不做数量门限**：原图中红色基线段被橙色曲线盖住，而复现图没有橙色，
  红色可见像素必然更多；红线只做中心线 RMSE 比对（仅在原图存在红色像素的列上）。

## 5. 数据窗口

绘图数据取 `data.csv` 中波长 ∈ [378.5, 385.6] 的行（144 点，步长 0.04–0.05nm），
直接 `ax.plot` 折线（原图即逐点折线，无平滑），xlim 裁剪负责两侧边界。
列名含空格/括号，用精确字符串：`'Sum(calc)'`、`'Eu II (8.6e-3)'`、`'Tm II (6.2e-3)'`。

## 6. 交付物（全部在 `repro_example1/`）

1. `plot_example1.py` — 主脚本。**参数接口（重要，用户后续要改线宽等）**：
   - 顶部 `DEFAULTS` dict（仿照现有 `plot_figure6.py` 的风格），至少包含：
     `data_path, output_path, figsize, dpi, xlim, ylim, xticks, yticks,
     grid_major(color/linewidth/visible), grid_minor(color/linewidth/visible),
     font(family/size/weight/color/xtick_pad),
     curves: {blue/red: {column, color, linewidth, visible}}（结构可扩展，便于以后加曲线）`
   - `plot_example1(config=None)` 函数：`config` dict 递归深合并覆盖 DEFAULTS，返回 `(fig, ax)`；`--show` 在保存 PNG 的同时显示窗口。
   - CLI 覆盖参数：`--blue-lw --red-lw --grid-lw --grid-minor-lw
     --font-size --font-family --font-weight --figsize --dpi --out --show` 等。
   - 路径基于 `__file__`，不依赖 CWD。运行 `python plot_example1.py` 生成
     `output/example1_repro.png`（1253×800, dpi=100，**禁止 tight_layout / bbox_inches='tight'**）。
2. `compare.py` — 客观比对脚本：`python compare.py [repro.png] [ref.png]`，输出逐项 PASS/FAIL：
   - 尺寸一致；
   - 蓝线纯色像素数 6562 ±12%；
   - 复现图中橙色像素数必须为 0（行<740 区域；用户要求不画 Tm II）；
   - 蓝线中心线 RMSE（排除陡峭列：列内游程>10px 的列、行≥740 的文字区）≤2.5px，p95≤6px；
   - 红线中心线 RMSE 同上，但仅在原图存在红色像素的列上计算；
   - 主/次网格线中心位置与原图检测值偏差 ≤2px（同一检测器跑两张图自比对）；
   - x 刻度标签簇中心偏差 ≤3px（7 个簇，首个为被裁剪的 "379"）；
   - 背景抽查为白色；
   - 末行输出 `RESULT: PASS` 或 `RESULT: FAIL`。
3. `README.md` — 中文说明：参数表 + 常用改参示例（改线宽、改颜色、改字号、换输出名）。

## 7. 环境约束

- Windows + Git Bash；`python` = 3.7.0，matplotlib **2.2.3**（旧版，勿用 3.x-only API）、
  numpy 1.21.6、pandas（旧版，参考现有 plot_figure6.py 的读取写法）、PIL 5.2.0。
- 无网络。字体用系统自带（C:\Windows\Fonts）。

## 8. 验收流程

程序员 agent 完成 → compare.py 全部 PASS → 交检验员 agent（judge）把
`output/example1_repro.png` 与 `example1.png` 做视觉验收 → 不过关则按意见迭代。
