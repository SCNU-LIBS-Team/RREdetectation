# Figure 1 小波寻峰计划

## Goal

复用项目现有小波寻峰方法，对 `data3.csv` 的 Figure 1 波段执行寻峰，并生成可复现的 `figure1_waveletpeak.png`。

## Current Phase

Phase 4

## Phases

### Phase 1: 需求与现有实现检查

- [x] 确认目标数据、输出文件名和不应修改的已有文件
- [x] 读取并理解 `Wavelet_peakfinding.py` 的算法和依赖
- [x] 记录发现
- **Status:** complete

### Phase 2: TDD RED

- [x] 定义 Figure 1 小波寻峰脚本的可测试接口
- [x] 先写测试并确认因实现缺失而失败
- **Status:** complete

### Phase 3: TDD GREEN 与重构

- [x] 编写最小实现使测试通过
- [x] 保留现有小波寻峰算法语义并集中配置参数
- **Status:** complete

### Phase 4: 出图与验证

- [ ] 生成 `figure1_waveletpeak.png`
- [ ] 运行局部测试、原 Figure 1 回归测试和语法检查
- [ ] 视觉检查结果图并核对输入文件未变化
- **Status:** in_progress

### Phase 5: 交付

- [ ] 汇总改动、寻峰结果、边缘情况和复现命令
- **Status:** pending

## Key Questions

1. 现有程序使用哪种 CWT 尺度、阈值和脊线判据？
2. Figure 1 应在全数据还是实际显示的 220–330 nm 区间寻峰？
3. 输出图如何同时清楚显示原光谱和被识别的峰？

## Decisions Made

| Decision | Rationale |
|---|---|
| 在 `Paper/Plot/figure1/` 内新增独立脚本和测试 | 隔离功能，避免改坏项目根目录旧脚本和现有 Figure 1 绘图 |
| 默认输入 `data3.csv`、分析范围 220–330 nm | 与当前 Figure 1 的真实数据源和展示范围一致 |
| 默认使用 `mexh`、尺度 1–10、邻域 4、最短脊线 3、系数阈值 700、校正窗口 5 | 与项目主流程 `Elements_detectation.py` 的现行调用一致；在 Figure 1 波段检测到稳定的 62 个峰 |
| 新图采用单面板原光谱加红色峰点 | 直接回答“寻峰结果图”，避免把未请求的 CWT 中间诊断图混入论文图 |

## Errors Encountered

| Error | Attempt | Resolution |
|---|---:|---|
| `session-catchup.py` 在 Python 3.7 因 `:=` 语法失败 | 1 | 不重复执行；人工检查现有计划与 `git diff --stat`，并使用 Figure 1 局部计划文件 |
| 首次正式图的右上图例遮挡 323.59 nm 最高峰 | 1 | 新增图例位置回归测试，计划改到左上空白区域后重新生成 |

## Notes

## 2026-09-15 style-alignment completion

- [x] Compare `figure1_waveletpeak` against the rendered and programmatic properties of `figure1_spectrum`.
- [x] Add failing regression tests for reference styling and marker appearance.
- [x] Reuse `plot_figure1.py` style constants and y-span rule.
- [x] Keep peak color `#D73027` while matching reference size, edge, and opacity.
- [x] Regenerate the 1200 DPI PNG and run full verification.
- **Status:** complete

- 不覆盖仓库根目录已有的 Figure 6 `task_plan.md`、`findings.md`、`progress.md`。
- 不修改 `data3.csv`。
