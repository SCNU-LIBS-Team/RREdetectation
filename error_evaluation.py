#此文件用来衡量计算得出温度的误差容许范围
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

kB=8.617330350e-5 #eV/K

file_path = r'D:\LIBS\RREdetectation\Elements_database\FeI.csv'

df = pd.read_csv(file_path, header=1, encoding="gbk")

df = df.iloc[1::2].copy()

# 只保留计算所需列，并将空值/非法值统一转为 NaN
needed_col_idx = [1, 2, 3, 7]
needed_cols = [df.columns[i] for i in needed_col_idx]
for col in needed_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 去除关键列中包含 NaN 或无穷值的行
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=needed_cols).reset_index(drop=True)

wl = df.iloc[:, 1].to_numpy(dtype=float)
A  = df.iloc[:, 2].to_numpy(dtype=float)
E  = df.iloc[:, 3].to_numpy(dtype=float)
g  = df.iloc[:, 7].to_numpy(dtype=float)

wl = wl * 0.1              
E  = E  * 1.2398e-4  


T_true=10000
def U_Calculate(g,A,E,T):
    if T <= 0:
        U = np.zeros(len(g), dtype=float)
        return U, 0.0
    U = g * np.exp(-E / (kB * T))
    U = np.where(np.isfinite(U), U, 0.0)
    return U, float(np.sum(U))

def rel_intensity(wl,A,E,g,T):
    _, U_T_sum = U_Calculate(g, A, E, T)
    rel_intensity = np.zeros(len(wl), dtype=float)


    if T <= 0:
        return rel_intensity

    # 仅在分母有效时计算，避免除零和无效值告警
    denominator = U_T_sum * wl
    valid = np.isfinite(denominator) & (np.abs(denominator) > 1e-30)
    if np.any(valid):
        numerator = A * g * np.exp(-E / (kB * T))
        valid = valid & np.isfinite(numerator)
        rel_intensity[valid] = numerator / denominator[valid]

    return rel_intensity/sum(rel_intensity)  # 归一化处理，确保总和为1



def error_evaluation(T_calculated,T_true):
    error = np.abs(rel_intensity(wl,A,E,g,T_calculated) - rel_intensity(wl,A,E,g,T_true))
    confidence_error=np.exp(-1.5*np.sum(error**2)/0.85)
    return confidence_error


def plot_confidence_error_curve(T_true=10000, t_min=5000, t_max=20000, num=500, save_path='confidence_error_curve.png'):
    """绘制 T_calculated 在指定区间内时 confidence_error 的变化曲线。"""
    T_values = np.linspace(t_min, t_max, num)
    confidence_mean = np.zeros(num)

    for i, T_calculated in enumerate(T_values):
        confidence_mean[i] = error_evaluation(T_calculated, T_true)

    plt.figure(figsize=(9, 5))
    plt.plot(T_values, confidence_mean, color='tab:blue', linewidth=2, label='confidence_error')
    plt.axvline(T_true, color='tab:red', linestyle='--', linewidth=1.5, label=f'T_true={T_true}')
    plt.xlabel('T_calculated')
    plt.ylabel('confidence_error')
    plt.title('Confidence Error vs T_calculated')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


# if __name__ == '__main__':
#     plot_confidence_error_curve(T_true=8000, t_min=5000, t_max=20000)

#导数原始定义
def derivative_P_T(i,T):
    p=rel_intensity(wl,A,E,g,T_true)/sum(rel_intensity(wl,A,E,g,T_true))
    global_derivative=-p[i]**2/kB/T**2
    discrete_derivative=(-E[i]-sum((-E*p)))
    derivative=-global_derivative*discrete_derivative
    return derivative

def derivative_curve_value(T):
    """独立于 derivative_P_T 的可绘图导数指标：mean(|dp/dT|)。"""
    if T <= 0:
        return 0.0
    p = rel_intensity(wl, A, E, g, T)
    E_mean = np.sum(-E * p)
    dp_dT = -p * (-E - E_mean) / (kB * T**2)
    return float(np.mean(np.abs(dp_dT)))

def plot_derivative_vs_T(k,t_min=500, t_max=20000, num=2000, save_path='derivative_vs_T.png'):
    """绘制导数指标随温度 T 的变化曲线。"""
    T_values = np.linspace(t_min, t_max, num)
    y_values = np.zeros(num, dtype=float)

    for i, T in enumerate(T_values):
        # y_values[i] = derivative_curve_value(T)
        y_values[i] = derivative_P_T(k, T)

    plt.figure(figsize=(9, 5))
    plt.plot(T_values, y_values, color='tab:green', linewidth=2, label='mean(|dp/dT|)')
    plt.axvline(T_true, color='tab:red', linestyle='--', linewidth=1.5, label=f'T_true={T_true}')
    plt.xlabel('T')
    plt.ylabel('Derivative Indicator')
    plt.title('Derivative vs Temperature')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


# 测试单个点的导数计算
# print(E[1])
# print(derivative_P_T(1, 10000))

# plot_derivative_vs_T(12,t_min=1000, t_max=20000, num=2000)
plot_confidence_error_curve(T_true=10000, t_min=5000, t_max=20000, num=500)