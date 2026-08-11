"""
2025年全球稀土矿产量分布饼图
数据来源：USGS, Mining-Technology/GlobalData
"""
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 2025年全球稀土产量数据（万吨REO）
countries = ['中国', '美国', '澳大利亚', '缅甸', '泰国', '其他']
production = [27.0, 4.8, 2.9, 2.65, 0.48, 1.17]  # 总计约39万吨

# 颜色
colors = ["#EE322F", '#1565C0', '#F9A825', '#43A047', '#8E24AA', '#9E9E9E']

# 突出显示中国
explode = [0.05, 0, 0, 0, 0, 0]

fig, ax = plt.subplots(figsize=(10, 8))

wedges, texts, autotexts = ax.pie(
    production,
    labels=countries,
    autopct=lambda pct: f'{pct:.1f}%\n({pct/100*sum(production):.2f}万吨)',
    explode=explode,
    colors=colors,
    startangle=90,
    textprops={'fontsize': 11},
    pctdistance=0.75
)

# 设置百分比文字样式
for autotext in autotexts:
    autotext.set_fontsize(9)
    autotext.set_color('white')
    autotext.set_fontweight('bold')

ax.set_title('2025年全球稀土矿产量分布\n(总产量约39万吨REO)\n数据来源: USGS, Mining-Technology/GlobalData',
             fontsize=14, fontweight='bold', pad=20)

# 添加图例
ax.legend(
    wedges, [f'{c} - {p}万吨' for c, p in zip(countries, production)],
    title="国家 - 产量",
    loc="center left",
    bbox_to_anchor=(1, 0, 0.5, 1),
    fontsize=10
)

plt.tight_layout()
plt.savefig('REE_Market_Data/2025年全球稀土产量分布饼图.png', dpi=150, bbox_inches='tight')
print("饼图已保存至: REE_Market_Data/2025年全球稀土产量分布饼图.png")
plt.show()
