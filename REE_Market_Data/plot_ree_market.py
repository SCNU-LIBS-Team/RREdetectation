"""
稀土元素市场数据可视化 - 柱状图
"""
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建图形
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('稀土元素(REE)市场数据概览', fontsize=16, fontweight='bold')

# ============ 图1: 全球稀土市场规模 ============
ax1 = axes[0, 0]
years_market = ['2022', '2023', '2024', '2025', '2030(预测)']
market_size = [64.2, 83.1, 64.7, 90.3, 134.9]  # 亿美元(取中间值)

bars1 = ax1.bar(years_market, market_size, color=['#2196F3', '#2196F3', '#2196F3', '#4CAF50', '#FF9800', '#FF9800'])
ax1.set_title('全球稀土金属市场规模', fontsize=13, fontweight='bold')
ax1.set_xlabel('年份')
ax1.set_ylabel('市场规模 (亿美元)')
ax1.set_ylim(0, 210)
for bar, val in zip(bars1, market_size):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 3,
             f'{val}', ha='center', va='bottom', fontsize=10)
ax1.grid(axis='y', alpha=0.3)

# ============ 图2: 各国稀土产量(2022年) ============
ax2 = axes[0, 1]
countries = ['中国', '美国', '缅甸', '澳大利亚', '泰国', '其他']
production_2022 = [21.0, 4.3, 1.2, 1.8, 0.7, 1.0]  # 万吨REO
colors2 = ['#E53935', '#1565C0', '#43A047', '#F9A825', '#8E24AA', '#757575']

bars2 = ax2.bar(countries, production_2022, color=colors2)
ax2.set_title('2022年全球稀土产量分布', fontsize=13, fontweight='bold')
ax2.set_xlabel('国家/地区')
ax2.set_ylabel('产量 (万吨REO)')
ax2.set_ylim(0, 25)
for bar, val in zip(bars2, production_2022):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
             f'{val}', ha='center', va='bottom', fontsize=10)
ax2.grid(axis='y', alpha=0.3)

# ============ 图3: 主要稀土氧化物价格对比(2020 vs 2022) ============
ax3 = axes[1, 0]
elements = ['La₂O₃', 'CeO₂', 'Nd₂O₃', 'Pr₆O₁₁', 'Dy₂O₃', 'Tb₄O₇']
price_2020 = [2.5, 1.75, 45, 45, 275, 700]  # 美元/kg 取中间值
price_2022 = [4, 3, 80, 75, 330, 1200]

x = np.arange(len(elements))
width = 0.35

bars3a = ax3.bar(x - width/2, price_2020, width, label='2020年', color='#42A5F5')
bars3b = ax3.bar(x + width/2, price_2022, width, label='2022年', color='#EF5350')
ax3.set_title('主要稀土氧化物价格对比', fontsize=13, fontweight='bold')
ax3.set_xlabel('稀土氧化物')
ax3.set_ylabel('价格 (美元/kg)')
ax3.set_xticks(x)
ax3.set_xticklabels(elements)
ax3.legend()
ax3.set_yscale('log')  # 使用对数坐标以便显示差异较大的数据
ax3.grid(axis='y', alpha=0.3)

# ============ 图4: 稀土下游应用市场分布 ============
ax4 = axes[1, 1]
applications = ['永磁材料', '催化剂', '抛光粉', '玻璃添加剂', '冶金合金', '荧光粉', '陶瓷', '其他']
percentages = [37.5, 22.5, 11, 7.5, 9, 4, 4, 6.5]
colors4 = ['#E53935', '#1565C0', '#43A047', '#F9A825', '#8E24AA', '#00897B', '#FF7043', '#757575']

bars4 = ax4.barh(applications, percentages, color=colors4)
ax4.set_title('稀土下游应用市场分布 (2022年)', fontsize=13, fontweight='bold')
ax4.set_xlabel('市场占比 (%)')
for bar, val in zip(bars4, percentages):
    ax4.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2.,
             f'{val}%', ha='left', va='center', fontsize=10)
ax4.set_xlim(0, 45)
ax4.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('REE_Market_Data/稀土市场数据柱状图.png', dpi=150, bbox_inches='tight')
plt.show()
print("图表已保存至: REE_Market_Data/稀土市场数据柱状图.png")
