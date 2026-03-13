#本文件用于验证谱线选择Branch
#SpectalConflictFinder.py 顾名思义，就是用来找稀土元素离子线与岩石基体元素之间的冲突的

import numpy as np
import pandas as pd
import glob
import os
import re
import pywt
import matplotlib.pyplot as plt
from Wavelet_peakfinding import find_peaks_ridge,wavelet_peak_detection

THRESHOLD = 0.2  # nm distance allowed between catalog line and detected peak

folder_path = r'D:\LIBS\RREdetectation\Rareearth' #元素谱线库的路径
folder_path2=r'D:\LIBS\RREdetectation\PureMainElems' #冲突谱线的输出路径
file_list = glob.glob(os.path.join(folder_path, "*II.csv")) # 只处理离子态谱线文�?
file_list2 = glob.glob(os.path.join(folder_path2, "*.csv")) # 只处理离子态谱线文档
elements_list = [os.path.splitext(os.path.basename(f))[0] for f in file_list]
PureElem_list=[os.path.splitext(os.path.basename(f))[0] for f in file_list2]
PureElem_base = [re.sub(r"\d+$", "", name) for name in PureElem_list] #主元素
elements = {}
conflicts = []  # store near-peak matches for later use

for element_name in elements_list: 
    #数据预处理
    file_path = os.path.join(folder_path, element_name + ".csv")
    df = pd.read_csv(file_path, header=1, encoding="gbk")
    df = df.iloc[1::2].copy()
    wl=df.iloc[:,1]
    #仅检测未启用的（N）的谱线
    if df.shape[1] > 8:
        enable_flag = df.iloc[:, 8]
        enable_mask = enable_flag.astype(str).str.strip().str.upper().eq("N")
    else:
        enable_mask = pd.Series(False, index=df.index)

    wl = pd.to_numeric(wl, errors="coerce")
    wl = wl * 0.1 
    valid_mask = (enable_mask &np.isfinite(wl))
    wl = wl[valid_mask]
    band_mask = (wl >= 200) & (wl <= 900)
    wl = wl[band_mask]
    wl = wl.to_numpy(dtype=float)
    elements[element_name] = wl #谱线库创建完成


    #遍历岩石基体元素
    for PureElem_name in PureElem_list:
        PureElem_path = os.path.join(folder_path2, PureElem_name + ".csv")
        df2 = pd.read_csv(PureElem_path, header=0, encoding="gbk")
        wl_Pure=df2.iloc[:,0]
        wl_Pure = pd.to_numeric(wl_Pure, errors="coerce")
        int_Pure=df2.iloc[:,1]
        int_Pure = pd.to_numeric(int_Pure, errors="coerce")
        true_peak_idx, peak_wl, peak_int = wavelet_peak_detection(int_Pure,wl_Pure,wavelet='mexh', scales=np.arange(1, 11), 
                               neighbor=4, min_length=3, coeffi_threshold=700, window=5)#峰值校正
        

        # 找到所有距离任意参考谱线小于 THRESHOLD 的峰值
        if peak_wl.size and wl.size:
            diff = np.abs(peak_wl[:, None] - wl[None, :])
            min_diff = diff.min(axis=1)
            close_mask = min_diff < THRESHOLD
            if close_mask.any():
                nearest_ref_idx = diff.argmin(axis=1)
                for i in np.where(close_mask)[0]:
                    conflicts.append({
                        "rareearth": element_name,
                        "pure_elem": PureElem_name,
                        "ref_wl": wl[nearest_ref_idx[i]],
                        "peak_wl": peak_wl[i],
                        "delta": float(min_diff[i]),
                    })
                # print(f"{PureElem_name} 与 {element_name} 存在 {close_mask.sum()} 条距离<{THRESHOLD}nm 的冲突峰")  

# for c in conflicts:
#     print(c)
    # print(c["rareearth"], c["peak_wl"],c["ref_wl"], c["delta"], c["pure_elem"])
    

#遍历稀土元素，更改谱线库中对应元素的谱线文件
for elements_name in elements_list:
    element_conflicts = [c for c in conflicts if c["rareearth"] == elements_name]
    folder_path = r'D:\LIBS\RREdetectation\Rareearth'
    file_list = glob.glob(os.path.join(folder_path, "*II.csv")) 
    df=pd.read_csv(os.path.join(folder_path, elements_name + ".csv"), header=1, encoding="gbk")
    if element_conflicts:
        conf_df = pd.DataFrame(element_conflicts)
        cols = ["peak_wl", "ref_wl", "delta", "pure_elem"]

        # 准备波长列：第1列转换为 nm（*0.1），并保证存在写入列（末尾新增）
        df_wl = pd.to_numeric(df.iloc[:, 1], errors="coerce") * 0.1
        if df.shape[1] <= 9:
            df.insert(loc=df.shape[1], column="conflict_elem", value=np.nan)
        target_col = df.columns[-1]  # 新增列

        # 将冲突峰值对应的基体元素写回谱线表
        for _, row in conf_df.iterrows():
            peak_wl = float(row["ref_wl"])
            pure_elem = row["pure_elem"]
            # 按波长精确匹配，无需容差
            exact_match = df_wl == peak_wl
            if exact_match.any():
                df.loc[exact_match, target_col] = pure_elem
        print(df)
        break
   


# ---- 输出 Tm 的全部谱线及其冲突谱线 ----
# tm_key = next((k for k in elements if k.lower().startswith("tm")), None)
# if tm_key:
#     tm_lines = np.sort(elements[tm_key])
#     print(f"{tm_key} 共 {tm_lines.size} 条参考谱线（nm）:")  # noqa: T201
#     print(tm_lines)  # noqa: T201

#     tm_conflicts = [c for c in conflicts if c["rareearth"] == tm_key]
#     if tm_conflicts:
#         tm_conf_df = pd.DataFrame(tm_conflicts)
#         cols = ["peak_wl", "ref_wl", "delta", "pure_elem"]
#         print(f"{tm_key} 冲突谱线:")  # noqa: T201
#         print(tm_conf_df[cols])  # noqa: T201
#         tm_conf_df.to_csv("Tm_conflicts.csv", index=False, encoding="utf-8-sig")
#     else:
#         print(f"{tm_key} 未发现距离<{THRESHOLD}nm 的冲突峰")  # noqa: T201
# else:
#     print("未在元素库中找到 Tm 的谱线文件")  # noqa: T201
