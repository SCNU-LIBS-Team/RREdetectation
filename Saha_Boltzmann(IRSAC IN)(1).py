
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#本程序用于IRSAC 并且可用于绘制Saha-Boltzmann平面图
#参数
k_B=8.617333262e-5#玻尔兹曼常量 单位ev/K
m_e=9.109382154e-31#电子质量 单位Kg
N_e=5.216e+17 #电子数密度 单位m^-3（此处为钛合金样品的参考值）
h=6.62607015e-34 #普朗克常数 单位Js


data=pd.read_excel('Saha_Boltzmann_PointValue.xlsx')#更换为待处理表格名
data = data.to_numpy()#数组预处理

# 提取各参数
λ1 = data[0:12, 0] *1e-9 # 原子线波长 (nm)转化为（m）
I1 = data[0:12, 1] # 原子线谱线强度 (任意单位)
A1 = data[0:12, 2] *1e+8 # 原子线跃迁概率 (1/s)
g1 = data[0:12, 3] # 原子线简并度
E1 = data[0:12, 4] # 原子线上能级能量 (eV)

λ2 = data[:, 5] *1e-9# 离子线波长 (nm)转化为（m）
I2 = data[:, 6] # 离子线谱线强度 (任意单位)
A2 = data[:, 7] *1e+8# 离子线跃迁概率 (1/s)
g2 = data[:, 8] # 离子线简并度
E2 = data[:, 9] # 离子线上能级能量 (eV)
E2_ion = data[:,10] #离子线的第一电离能

def SB_PointCalu(T):
    # 计算散点
    y1 = np.log((I1 * λ1) / (g1 * A1))
    y2 = np.log((I2 * λ2) / (g2 * A2)) - np.log((2 * (2 * np.pi * m_e * k_B * T) ** 1.5) / (N_e * h ** 3))
    x1 = E1
    x2 = E2 + E2_ion
    # 合并散点
    y = np.concatenate((y1, y2))
    x = np.concatenate((x1, x2))
    # 线性拟合
    coefficients = np.polyfit(x, y, 1)  # slope斜率 intercpet截距 拟合
    slope, intercept = coefficients
    T_new = -1 / (slope * k_B)  # 温度计算
    return T_new,coefficients,x,y

def IRSAC_circle(λ, I, E, A, g, T):
    data_list = []
    for i in range(len(λ)):
        I_R, λ_R, E_R, A_R, g_R = I[i], λ[i], E[i], A[i], g[i]
        I_J = np.delete(I, i)
        λ_J = np.delete(λ, i)
        E_J = np.delete(E, i)
        A_J = np.delete(A, i)
        g_J = np.delete(g, i)

        # 初始自吸收系数
        SA_R = 1.0
        min_SA_R = 0.0
        step = 0.01
        max_iter = 1000
        iter_count = 0

        while iter_count < max_iter:
            numerator = (I_J * λ_J) * (A_R * g_R) * np.exp(-E_R / (k_B * T))
            denominator = (I_R * λ_R) * (A_J * g_J * np.exp(-E_J / (k_B * T)))
            SA_J = SA_R * numerator / denominator

            if np.all(SA_J < 1) or SA_R <= min_SA_R:
                break
            SA_R -= step
            iter_count += 1
        summary= np.sum(SA_J)
        data_list.append([i, λ_R, SA_R,summary,SA_J])
    # 找到SA_J的最大值所在的行，并输出该行
    SA_J_array = np.array([row[3] for row in data_list], dtype=object)
    max_index = np.argmax([np.max(saj) for saj in SA_J_array])
    print("SA_J最大值所在的行：", data_list[max_index])

    return np.array(data_list, dtype=object),data_list[max_index]


def T_iteration(tolerance,max_iteration,T_initial):
    iteration = 0
    T_current=T_initial
    T_previous=T_initial+2*tolerance

    while abs(T_current - T_previous) > tolerance and iteration < max_iteration:
        T_previous = T_current
        T_current,coefficients,x,y = SB_PointCalu(T_previous)
        iteration += 1
    if iteration == max_iteration:
        print(f"警告：未在{max_iteration}次内收敛，当前差值{abs(T_current-T_previous):.2f} K")
    return iteration,T_current,coefficients,x,y

def calculate_r2(y_true, y_pred):
    # 将输入转换为numpy数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 计算实际值的均值
    mean_actual = np.mean(y_true)

    # 计算总平方和（SS_tot）
    ss_total = np.sum((y_true - mean_actual) ** 2)

    # 计算残差平方和（SS_res）
    ss_residual = np.sum((y_true - y_pred) ** 2)

    # 处理分母为零的特殊情况
    if ss_total == 0:
        # 如果所有实际值相同，且预测完全正确则返回1，否则返回0
        return 1.0 if ss_residual == 0 else 0.0

    # 计算R平方
    r2 = 1 - (ss_residual / ss_total)

    return r2

#IRSAC
IRSAC,SA_Jmax=IRSAC_circle(λ1,I1,E1,A1,g1,8000)
print(IRSAC)
print(SA_Jmax)







# iteration,T_final,coefficients,x,y=T_iteration(0.00001,1000,10000)
# slope,intercept=coefficients

# #拟合直线的数据点计算
# fit_x = np.linspace(min(x), max(x), 1000)
# fit_y = np.polyval(coefficients, fit_x)

# #r^2计算
# y_pred=slope*x+intercept
# r2_score = calculate_r2(y, y_pred)

# #文本编辑
# text = (
#     f"R²=: {r2_score:.4f}\n"
#     f"T= : {T_final:.4f}\n"
#     f"Iteration= : {iteration:.4f}\n"
# )

# #绘制Saha-Bboltzmann平面图
# plt.figure(figsize=(10,6))
# plt.scatter(x,y,color='blue',label='port',zorder=2)
# plt.plot(fit_x, fit_y, color='red', linestyle='--',label=f'\n$y={slope:.2f}x+{intercept:.2f}$\n$T={T_final:.0f}$ K', zorder=1)

# #添加图例和标签
# plt.xlabel('E(eV)',fontsize=12)
# plt.ylabel('Fitpoint', fontsize=12)
# plt.title('Saha-Boltzmann Plot', fontsize=14)
# plt.text(0.9, 1.1, text, horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12)#显示R^2
# plt.legend(frameon=True, loc='best', fontsize=10)
# plt.grid(True, alpha=0.3)

# #显示图标
# plt.tight_layout()
# plt.show()
