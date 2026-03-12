import numpy as np
import pandas as pd
import glob
import os
import pywt
import matplotlib.pyplot as plt

folder_path = r'D:\LIBS\RREdetectation\Rareearth' #元素谱线库的路径
folder_path2=r'D:\LIBS\RREdetectation\PureMainElems' #冲突谱线的输出路径
file_list = glob.glob(os.path.join(folder_path, "*II.csv")) # 只处理离子态谱线文�?
elements_list = [os.path.splitext(os.path.basename(f))[0] for f in file_list]
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
    print(element_name, ":", wl)
