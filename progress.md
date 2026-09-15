# Figure 6 光谱图复现进度

## 2026-09-04

- 已读取 `planning-with-files`、`test-driven-development`、`verification-before-completion` 技能说明。
- 已尝试运行 planning-with-files 的 session catchup；因系统 Python 3.7 不支持脚本使用的 `:=` 语法而失败。
- 已检查仓库状态：存在大量任务外的用户改动，后续仅操作本任务新建文件。
- 已将工作拆为检查、TDD 实现、生成与验证三个阶段。
- 已检查 CSV 表头、依赖版本、现有绘图脚本及参考图；确认目标列为 `Wavelength (nm)`、`Sum(calc)`、`Eu II (8.6e-3)`。
- 为保存用户已有尝试，决定新建独立的局部光谱复现脚本与测试，不覆盖 `plot_figure6.py` 等已有文件。
- 阶段 1 完成。发现 `repro_example1/plot_example1.py` 已是本任务的一份未跟踪复现尝试，因此不另起重复脚本；在该隔离目录内补齐集中配置和测试。
- TDD RED：新增 4 个回归测试后运行，结果为 2 failures + 1 error；失败明确来自缺少 smoothing/spines/legend/labels 集中配置、嵌套覆盖丢失列定义、未实现平滑。
- TDD GREEN：实现递归配置合并、可选移动平均、可调边框/图例/标签和曲线线型/标签；复跑 4 个测试全部通过（旧版 Matplotlib 在 Python 3.7 下输出 2 条依赖级弃用警告）。
- 阶段 2 完成，进入默认图生成和视觉核对阶段。
- 已运行默认脚本生成 `repro_example1/output/example1_repro.png`（1253×800，57,432 bytes）。
- 已运行像素级比较：8/8 检查通过，结果 `RESULT: PASS`。
- 已人工查看生成图：仅有蓝色 Sum 和红色 Eu II，两条曲线及整体版式符合参考图；未出现橙线。
- 已复核输入 SHA-256 与开始时相同，`data.csv` 未被修改。
- 最终新鲜验证：`unittest discover` 4/4 通过；`py_compile` 通过；默认脚本重新生成成功；像素比较 8/8 通过；PNG 为 1253×800 RGBA；输入哈希不变。
- 全部三个阶段完成，未提交或推送任何内容。
- 收到独立评审后重新打开任务。已核实 compare 覆盖门槛、CSV schema 验证、DPI/figsize 缩放语义、PNG 输出约束和 rcParams 显式化问题与本任务有关；进入 Phase 4，继续按 receiving-code-review + TDD 修复。
- Phase 4 RED（第一轮）：低红线覆盖和额外红列三个反例均错误返回 PASS；10×10 画布触发固定坐标 `IndexError`。网格反例第一次参数方向写反而意外通过，已纠正测试方向后重跑。
- Phase 4 GREEN：5/5 compare 反例测试通过；默认输出仍通过 compare 8/8。覆盖门槛、一一对应网格匹配及画布尺寸早停已生效。
- Phase 5 RED：12 个绘图测试中 6 failures + 3 errors，覆盖缺失列、错误列选择、空窗口、点数不足、非数值/无限值、重复/非递增波长、DPI/figsize 布局漂移和 rcParams 隐式样式。
- Phase 5 GREEN：12/12 绘图测试通过。CSV 在建图前完成严格验证；列名、参考画布、axes/网格/曲线关键样式全部集中；axes 归一化布局不再随 DPI/figsize 漂移。
- Phase 6 RED：真实 PNG/目录/尺寸/输入不变集成测试已通过现有行为，非 `.png` 输出反例按预期失败（旧实现生成了 JPEG）。
- Phase 6 GREEN：14/14 绘图测试通过；非 PNG 路径在建图/建目录前明确拒绝，`savefig(format="png")` 已显式固定格式，深合并与 `--show` 文档已更正。
