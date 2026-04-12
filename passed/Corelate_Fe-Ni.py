import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from sklearn.linear_model import LinearRegression
from Wavelet_peakfinding import wavelet_peak_detection


def extract_Ni_value(name):
    try:
        return float(name[2:])
    except:
        print(f"警告：无法从文件名解析 Ni 含量：{name}")
        return None
    

# ================================
# 读取所有文件
# ================================
folder_path = r'D:\LIBS\ElementDetectation\11.10\Fe-Ni_Spec'
file_list = glob.glob(os.path.join(folder_path, "*.csv"))
elements_list = [os.path.splitext(os.path.basename(f))[0] for f in file_list]
Ni_contents = [extract_Ni_value(name) for name in elements_list]

# 把光谱读入内存
spectra = {}
for name in elements_list:
    df = pd.read_csv(os.path.join(folder_path, name + ".csv"))
    wl = df.iloc[:, 0].values
    intensity = df.iloc[:, 1].values
    spectra[name] = (wl, intensity)


if "Ni100" not in spectra:
    raise ValueError("缺少 Ni100.csv")

wl_100, int_100 = spectra["Ni100"]

peak_idx, peak_wl_100, peak_int_100 = wavelet_peak_detection(
    int_100, wl_100,
    scales=np.arange(1, 11), neighbor=3,
    min_length=3, coeffi_threshold=100, window=5
)

print(f"Ni100 找到 {len(peak_wl_100)} 个基准峰")
print("峰位：", peak_wl_100)

search_width = 0.07   # 阈值

def find_peak_near(wl_arr, int_arr, target_wl, width=0.1):
    """在 target_wl ± width 内寻找局部最大强度"""
    mask = (wl_arr >= target_wl - width) & (wl_arr <= target_wl + width)
    if not np.any(mask):
        return None  # 没数据点
    sub_wl = wl_arr[mask]
    sub_int = int_arr[mask]
    idx = np.argmax(sub_int)
    return sub_int[idx]   # 返回峰强


fit_results = []

for base_wl in peak_wl_100:

    Ni_vals = []
    peak_vals = []

    for Ni_value, name in zip(Ni_contents, elements_list):
        wl, intensity = spectra[name]

        peak_int = find_peak_near(wl, intensity, base_wl, width=search_width)

        # 若找不到任何点，则跳过（也可以设为0，看需求）
        if peak_int is None:
            continue

        Ni_vals.append(Ni_value)
        peak_vals.append(peak_int)

    # 至少 3 个不同 Ni 才拟合
    if len(np.unique(Ni_vals)) < 3:
        continue

    X = np.array(Ni_vals).reshape(-1, 1)
    Y = np.array(peak_vals)

    model = LinearRegression()
    model.fit(X, Y)

    slope = model.coef_[0]
    intercept = model.intercept_
    r2 = model.score(X, Y)
    count = len(X)

    fit_results.append([base_wl, slope, intercept, r2, count])

    # if base_wl ==303.78:
    #     # 绘图查看拟合效果
    #     plt.scatter(X, Y, label="Data Points")
    #     plt.plot(X, model.predict(X), color='red', label="Fit Line")
    #     plt.xlabel("Ni Content")
    #     plt.ylabel("Peak Intensity")
    #     plt.title(f"Peak at {base_wl} nm: slope={slope:.3f}, R2={r2:.3f}")
    #     plt.legend()
    #     plt.show()



# ================================
# 3. 结果整理并排序
# ================================
result_df = pd.DataFrame(
    fit_results,
    columns=["peak_wl", "slope", "intercept", "R2", "count"]
)

# positive_df = result_df[result_df["slope"] > 0].copy()
# positive_df["slope_dist_to_1"] = (positive_df["slope"] - 1).abs()

# best_slope_df = positive_df.sort_values("slope_dist_to_1")

# print(best_slope_df.head(40))

bad_lines = result_df[result_df["R2"] < 0.95]
print(bad_lines)
