"""
全球稀土储量分布饼图
"""
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据
countries = ['中国', '巴西', '印度', '澳大利亚', '俄罗斯', '越南', '美国', '其他']
reserves = [4400, 2100, 690, 570, 380, 350, 190, 520]  # 万吨
# 其他 = 总量约9200 - 已知国家总和

# 颜色
colors = ['#E53935', '#43A047', '#FF9800', '#1565C0', '#8E24AA', '#00897B', '#F44336', '#9E9E9E']

# 突出显示中国
explode = [0.05, 0, 0, 0, 0, 0, 0, 0]

fig, ax = plt.subplots(figsize=(10, 8))

wedges, texts, autotexts = ax.pie(
    reserves,
    labels=countries,
    autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(reserves))}万吨)',
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

ax.set_title('全球稀土储量分布\n(数据来源: USGS)', fontsize=16, fontweight='bold', pad=20)

# 添加图例
ax.legend(
    wedges, [f'{c} - {r}万吨' for c, r in zip(countries, reserves)],
    title="国家 - 储量",
    loc="center left",
    bbox_to_anchor=(1, 0, 0.5, 1),
    fontsize=10
)

plt.tight_layout()
plt.savefig('REE_Market_Data/全球稀土储量分布饼图.png', dpi=150, bbox_inches='tight')
plt.show()
print("饼图已保存至: REE_Market_Data/全球稀土储量分布饼图.png")
