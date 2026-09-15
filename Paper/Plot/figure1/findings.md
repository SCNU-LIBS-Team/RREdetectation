# Figure 1 小波寻峰发现

## Requirements

- 对 Figure 1 当前使用的数据执行现有小波寻峰算法。
- 输出结果图名为 `figure1_waveletpeak.png`。
- 结果应可通过脚本重复生成。

## Research Findings

- Figure 1 当前默认数据为 `Paper/Plot/figure1/data3.csv`。
- 当前 Figure 1 展示区间为 220–330 nm。
- `data3.csv` 包含 `wavelength,intensity` 两列，完整范围 198.79–603.55 nm。
- 仓库根目录已有与 Figure 6 相关的计划文件，本任务不可覆盖。
- `Wavelet_peakfinding.py` 的核心接口是 `wavelet_peak_detection(signal, wl, ...)`，使用 PyWavelets 的 Mexican hat (`mexh`) 连续小波变换、跨尺度脊线跟踪和局部极大值校正。
- 项目主检测流程当前使用尺度 1–10、`neighbor=4`、`min_length=3`、`coeffi_threshold=700`、`window=5`。
- 对 `data3.csv` 的 220–330 nm 共 2254 点试算，阈值 700 得到 62 个唯一峰且没有重复索引；Figure 1 原先手选的六个峰全部被识别。
- 现有模块导入时会额外读取硬编码的 `RREs/03116_95.csv` 并计算一次无关 CWT；本次先不改旧模块，以缩小变更范围。

## Technical Decisions

| Decision | Rationale |
|---|---|
| 新脚本导入并调用 `wavelet_peak_detection` | 复用现有项目算法，不复制脊线逻辑 |
| 默认输出单面板光谱加峰点 | 结果直观且与用户指定的单张结果图一致 |
| 后处理峰索引为唯一、升序整数数组 | 防止未来参数变化产生重复或乱序峰点 |

## Issues Encountered

| Issue | Resolution |
|---|---|
| 默认 Python 为 3.7，planning 恢复脚本使用了较新语法 | 手工恢复上下文；实现需保持 Python 3.7 兼容 |

## Resources

- `Wavelet_peakfinding.py`
- `Paper/Plot/figure1/data3.csv`
- `Paper/Plot/figure1/plot_figure1.py`

## Visual Findings

### 2026-09-15 style-alignment follow-up

- `figure1_spectrum.png` uses a much taller 0-to-about-620000 y span because its upper mirrored lines determine the limit.
- The reference image has no visible title and no legend; its axis labels, ticks, frame, and canvas are styled by `plot_figure1.apply_spectrum_style`.
- Reference peak markers use size 20, no contrasting white edge, and visually full opacity.
- The existing wavelet result differs in y span, title/legend, spectrum alpha (0.65 vs 0.5), marker size (22 vs 20), marker edge, and save DPI (600 vs 1200).
- The requested exception is the wavelet peak face color `#D73027`; all other visible style details should follow `figure1_spectrum`.
- Final property verification: x limits `(220.0, 330.0)`, y limits `(0.0, 639024.75)`, 15 pt semibold axis labels, gray spectrum line at alpha 0.5, and no title or legend.
- Final peak collection: 62 markers, size 20, face/edge RGBA `(0.843137, 0.188235, 0.152941, 1.0)`, which is opaque `#D73027` and matches the reference marker opacity.
- Regenerated PNG metadata: 11360 x 5841 px at approximately 1200 DPI.

### Font-rendering correction

- A `semibold` property alone did not guarantee the same rendered glyphs.
- Root cause: importing `Wavelet_peakfinding.py` prepended Chinese font candidates to global Matplotlib `font.sans-serif`; the wavelet figure resolved labels to `simhei.ttf`, while a clean `figure1_spectrum` process resolved them to `DejaVuSans-Bold.ttf`.
- `plot_figure1_waveletpeak.py` now snapshots and restores the affected Matplotlib font rcParams around the legacy detector import.
- Final verification resolves every axis label and tick in both figures to `DejaVuSans-Bold.ttf`, with weight `semibold`.
- The corrected PNG is exactly 11386 x 5876 px at approximately 1200 DPI, matching `figure1_spectrum.png`.

- 正式图已生成，原始尺寸 5682×2929，横向版式、灰色光谱和 62 个红色峰点均清晰。
- 右上角图例与 323.59 nm 的最高峰重叠；需要把图例移到上方左侧空白区域后重新验证。
- 根因是图例固定为 `upper right`，而全局最高峰也位于波段右侧；左上区域在相同高度没有高峰。单一修复假设：只把图例位置改为 `upper left` 即可解除遮挡。
- 修复后重新渲染并查看：图例位于左上空白区，不再遮挡任何高峰；红色检测点与对应局部峰顶对齐，横纵轴、标题和边框均完整。
