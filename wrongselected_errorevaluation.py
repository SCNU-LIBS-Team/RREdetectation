#此文件用来计算选错线时的置信度或者相对强度误差
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from error_evaluation import read_csv_with_fallback, U_Calculate, plot_U_sum_vs_T_from_df

kB=8.617330350e-5 #eV/K



def load_line_data(file_path):
    """读取并清洗谱线数据，返回 wl, A, E, g（已完成单位换算）。"""
    df_raw = read_csv_with_fallback(file_path, header=1)
    df = df_raw.copy()

    # 仅保留奇数行（与原脚本保持一致）
    df = df.iloc[1::2].copy()

    # 第9列若为 N，则该行不参与后续读取与计算
    if df.shape[1] >= 9:
        col9 = df.iloc[:, 8]
        keep_mask = col9.isna() | (col9.astype(str).str.strip().str.upper() != 'N')
        df = df.loc[keep_mask].reset_index(drop=True)

    # 只保留计算所需列，并将空值/非法值统一转为 NaN
    needed_col_idx = [1, 2, 3, 7]
    max_idx = max(needed_col_idx)
    if df.shape[1] <= max_idx:
        raise ValueError(f'列数不足，至少需要到第{max_idx + 1}列，当前仅有{df.shape[1]}列。')

    needed_cols = [df.columns[i] for i in needed_col_idx]
    for col in needed_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 去除关键列中包含 NaN 或无穷值的行
    df[needed_cols] = df[needed_cols].mask(~np.isfinite(df[needed_cols]), np.nan)
    df = df.dropna(subset=needed_cols).reset_index(drop=True)

    wl = df.iloc[:, 1].to_numpy(dtype=float)
    A = df.iloc[:, 2].to_numpy(dtype=float)
    E = df.iloc[:, 3].to_numpy(dtype=float)
    g = df.iloc[:, 7].to_numpy(dtype=float)

    wl = wl * 0.1
    E = E * 1.2398e-4
    return wl, A, E, g

#谱线数据读取
file_path_right = r'D:\LIBS\RREdetectation\Rareearth_pt3\YII.csv'
file_path_wrong = r'D:\LIBS\RREdetectation\Elements_database\FeII.csv'
wl_right, A_right, E_right, g_right = load_line_data(file_path_right)
wl_wrong, A_wrong, E_wrong, g_wrong = load_line_data(file_path_wrong)


def select_lines_by_count(wl, A, E, g, n_lines=None):
    """按数量选择谱线：优先取 A 最大的 n_lines 条；None 表示全取，0 表示不取。"""
    total = len(wl)
    if n_lines is None:
        return wl, A, E, g

    n = int(n_lines)
    if n < 0:
        raise ValueError('n_lines 不能为负数。')
    if n == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty, empty
    if n >= total:
        return wl, A, E, g

    # 选取 A 最大的 n 条线，并按原顺序返回，便于后续对照
    idx = np.argsort(A)[::-1][:n]
    idx = np.sort(idx)
    return wl[idx], A[idx], E[idx], g[idx]


#第三步部分开发不同元素的配分函数比较
#配分函数数据读取
df_right=read_csv_with_fallback(r'D:\LIBS\RREdetectation\Ucalculation\0406_Y.csv', header=0)
df_wrong=read_csv_with_fallback(r'D:\LIBS\RREdetectation\Ucalculation\0406_FeI.csv', header=0)


#比较部分
#Attention:density_ratio=number_wrong/number_right
def plot_wrongselected(density_ratio=10, T_true=10000, t_min=5000, t_max=20000, num=500, show_plot=True, n_right=None, n_wrong=None):

    wl_right_sel, A_right_sel, E_right_sel, g_right_sel = select_lines_by_count(wl_right, A_right, E_right, g_right, n_right)
    wl_wrong_sel, A_wrong_sel, E_wrong_sel, g_wrong_sel = select_lines_by_count(wl_wrong, A_wrong, E_wrong, g_wrong, n_wrong)

    if len(wl_right_sel) == 0: 
        raise ValueError('right 线数量为 0，无法计算 p_T_values。请将 n_right 设为 None 或正整数。')

    T_values, U_sum_values_right = plot_U_sum_vs_T_from_df(
    df_right,
    t_min=t_min,
    t_max=t_max,
    num=num,
    show_plot=False,
    return_values=True
)
    _, U_sum_values_wrong = plot_U_sum_vs_T_from_df(
    df_wrong,
    t_min=t_min,
    t_max=t_max,
    num=num,
    show_plot=False,
    return_values=True
)   
    T_values = np.linspace(t_min, t_max, num)
    p_T_values = np.zeros((len(wl_right_sel), num))
    

    for i, T in enumerate(T_values):
        U_right, _ = U_Calculate(g_right_sel, A_right_sel, E_right_sel, T)
        right_term = np.sum(U_right * A_right_sel)

        wrong_term = 0.0
        partition_ratio = 0.0
        if len(wl_wrong_sel) > 0:
            U_wrong, _ = U_Calculate(g_wrong_sel, A_wrong_sel, E_wrong_sel, T)
            wrong_term = np.sum(U_wrong * A_wrong_sel)
            if np.abs(U_sum_values_wrong[i]) > 1e-30:
                partition_ratio = U_sum_values_right[i] / U_sum_values_wrong[i]

        denominator = right_term + density_ratio * partition_ratio * wrong_term
        if np.isfinite(denominator) and np.abs(denominator) > 1e-30:
            p_T_values[:, i] = (U_right * A_right_sel) / denominator
        # print(U_sum_values_right[i]/U_sum_values_wrong[i])
    if show_plot:
        plt.figure(figsize=(9, 5))
        for i in range(len(wl_right_sel)):
            plt.plot(T_values, p_T_values[i, :], label=f'wl={wl_right_sel[i]:.1f}nm')
        plt.xlabel('T')
        plt.ylabel('p')
        plt.title('wrongselected')
        plt.minorticks_on()
        plt.tick_params(axis='both', which='major', direction='in', top=True, right=True)
        plt.tick_params(axis='both', which='minor', direction='in', top=True, right=True)
        plt.grid(alpha=0.3)
        if len(wl_right_sel) > 0:
            plt.legend()
        plt.tight_layout()
        plt.show()

    return T_values, p_T_values

if __name__ == "__main__":
    plot_wrongselected(
    density_ratio=100.0,
    T_true=10000,
    t_min=5000,
    t_max=20000,
    num=500,
    show_plot=True,
    n_right=3,
    n_wrong=3)