import numpy as np
import pandas as pd
import glob
import os
import re
import pywt
import matplotlib.pyplot as plt
from wavelet_peakfinding import find_peaks_ridge,wavelet_peak_detection

folder_path = r'D:\LIBS\RREdetectation\Rareearth' #元素谱线库的路径
folder_path2=r'D:\LIBS\RREdetectation\PureMainElems' #冲突谱线的输出路径
file_list = glob.glob(os.path.join(folder_path, "*II.csv")) # 只处理离子态谱线文�?
file_list2 = glob.glob(os.path.join(folder_path2, "*.csv")) # 只处理离子态谱线文档
elements_list = [os.path.splitext(os.path.basename(f))[0] for f in file_list]
PureElem_list=[os.path.splitext(os.path.basename(f))[0] for f in file_list2]
PureElem_base = [re.sub(r"\d+$", "", name) for name in PureElem_list] #主元素
elements = {}

for element_name in elements_list: 
    file_path = os.path.join(folder_path, element_name + ".csv")
    df = pd.read_csv(file_path, header=1, encoding="gbk")
    df = df.iloc[1::2].copy()
    wl=df.iloc[:,1]
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
    elements[element_name] = wl
    for PureElem_name in PureElem_list:
        PureElem_path = os.path.join(folder_path2, PureElem_name + ".csv")
        df2 = pd.read_csv(PureElem_path, header=0, encoding="gbk")
        wl_Pure=df2.iloc[:,0]
        wl_Pure = pd.to_numeric(wl_Pure, errors="coerce")
        int_Pure=df2.iloc[:,1]
        int_Pure = pd.to_numeric(int_Pure, errors="coerce")
        true_peak_idx, peak_wl, peak_int = wavelet_peak_detection(int_Pure,wl_Pure,wavelet='mexh', scales=np.arange(1, 11), 
                               neighbor=4, min_length=3, coeffi_threshold=700, window=5)#峰值校正
        