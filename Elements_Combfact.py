import numpy as np
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt 
kB=8.617330350e-5 #eV/K
#计算U（T） 返回U和U总和
def U_Calculate(g,A,E,T):
    U=np.zeros(len(g))
    for i in range(len(g)):
        U[i]=g[i]*np.exp(-E[i]/(kB*T))
    return U,np.sum(U)

#计算相对强度 返回相对强度
#遗留问题1：模拟强度到底要不要wl
def rel_intensity(wl,A,E,g,T):
    U_T,U_T_sum=U_Calculate(g,A,E,T)
    rel_intensity=np.zeros(len(wl))
    for i in range(len(wl)):
        rel_intensity[i]=(A[i]*g[i]*np.exp(-E[i]/(kB*T)))/(U_T_sum*wl[i])  
    return rel_intensity

#元素库制作 返回elements字典和elements_list元素列表
def elements_database(folder_path,T):
    file_list = glob.glob(os.path.join(folder_path, "*.csv"))
    elements_list = [os.path.splitext(os.path.basename(f))[0] for f in file_list]
    #元素特征光谱制作
    elements={}
    for element_name in elements_list: 
        file_path = os.path.join(folder_path, element_name + ".csv")  # 拼接完整路径
        df = pd.read_csv(file_path,header=1,encoding="gbk")  # 读取该元素的csv
        df=df.to_numpy()
        even_rows = df[1::2]
        wl=even_rows[:,1]*0.1
        A=even_rows[:,2]
        E=even_rows[:,3]*1.2398*10**(-4) #eV
        g=even_rows[:,7]
        # 强制转换为 float
        wl = wl.astype(float)
        A  = A.astype(float)
        E  = E.astype(float)
        g  = g.astype(float)
        #波段过滤 200-900nm
        mask = (wl >= 200) & (wl <= 900)
        wl = wl[mask]
        A = A[mask]
        E = E[mask]
        g = g[mask]
        
        relative_intensity=rel_intensity(wl,A,E,g,T)
        matrix = np.column_stack((wl, relative_intensity,A,E,g))
        elements[element_name] = { "data": matrix}
    return elements,elements_list

def elements_database_pt2(folder_path, T):
    file_list = glob.glob(os.path.join(folder_path, "*.csv"))
    elements_list = [os.path.splitext(os.path.basename(f))[0] for f in file_list]

    elements = {}
    for element_name in elements_list: 
        file_path = os.path.join(folder_path, element_name + ".csv")

        # 读取 CSV
        df = pd.read_csv(file_path, header=1, encoding="gbk")

        # 只取偶数行
        df = df.iloc[1::2].copy()

        # ===============================
        # ① 先取列（不要做任何运算）
        # ===============================
        wl = df.iloc[:, 1]
        A  = df.iloc[:, 2]
        E  = df.iloc[:, 3]
        g  = df.iloc[:, 7]
        if df.shape[1] > 8:
            enable_flag = df.iloc[:, 8]
            enable_mask = enable_flag.isna() | (
                enable_flag.astype(str).str.strip().str.upper().isin(["", "Y"])
            )
         
  
        else:
            enable_mask = pd.Series(True, index=df.index) #problem
            

        # ===============================
        # ② 强制转数值（核心）
        # ===============================
        wl = pd.to_numeric(wl, errors="coerce")
        A  = pd.to_numeric(A,  errors="coerce")
        E  = pd.to_numeric(E,  errors="coerce")
        g  = pd.to_numeric(g,  errors="coerce")

        # ===============================
        # ③ 单位换算（现在才安全）
        # ===============================
        wl = wl * 0.1                # Å → nm
        E  = E  * 1.2398e-4          # cm⁻¹ → eV（按你原公式）

        # ===============================
        # ④ 物理合法性过滤
        # ===============================
        valid_mask = (
            enable_mask &
            np.isfinite(wl) &
            np.isfinite(A) & (A > 0) &
            np.isfinite(E) &
            np.isfinite(g)
        )

        wl = wl[valid_mask]
        A  = A[valid_mask]
        E  = E[valid_mask]
        g  = g[valid_mask]

        # ===============================
        # ⑤ 波段过滤
        # ===============================
        band_mask = (wl >= 200) & (wl <= 900)

        wl = wl[band_mask]
        A  = A[band_mask]
        E  = E[band_mask]
        g  = g[band_mask]

        # ===============================
        # ⑥ 转 numpy（现在 100% 安全）
        # ===============================
        wl = wl.to_numpy(dtype=float)
        A  = A.to_numpy(dtype=float)
        E  = E.to_numpy(dtype=float)
        g  = g.to_numpy(dtype=float)

        # 相对强度
        relative_intensity = rel_intensity(wl, A, E, g, T)

        matrix = np.column_stack((wl, relative_intensity, A, E, g))
        elements[element_name] = {"data": matrix}

    return elements, elements_list

#目前的想法是main_elements是一个列表，里面包含了岩石基体元素
def elements_database_lineswitch(folder_path, T, main_elements, LineSwitchMode=False):
    file_list = glob.glob(os.path.join(folder_path, "*.csv"))
    elements_list = [os.path.splitext(os.path.basename(f))[0] for f in file_list]
    elements = {}
    for element_name in elements_list: 
        file_path = os.path.join(folder_path, element_name + ".csv")

        # 读取 CSV
        df = pd.read_csv(file_path, header=0, encoding="gbk")

        # 只取偶数行
        df = df.iloc[1::2].copy()
        wl = df.iloc[:, 1]
        A  = df.iloc[:, 2]
        E  = df.iloc[:, 3]
        g  = df.iloc[:, 7]
        
        if df.shape[1] > 9:
            enable_flag = df.iloc[:, 8]
            pure_element_flag = df.iloc[:, 9]
            main_elements_normalized = {str(m).strip().upper() for m in main_elements} #基体元素全集
            normalized_pure_element = pure_element_flag.astype(str).str.strip().str.upper()
            base_mask = (
                enable_flag.isna()
                | enable_flag.astype(str).str.strip().str.upper().isin(["", "Y"])
            )
            has_pure_flag = pure_element_flag.notna() & normalized_pure_element.ne("")
            non_matrix_pure = ~normalized_pure_element.isin(main_elements_normalized)
            if LineSwitchMode:
                enable_mask = base_mask | (has_pure_flag & non_matrix_pure)
            else:
                enable_mask = base_mask
        else:
            enable_mask = pd.Series(True, index=df.index)


        wl = pd.to_numeric(wl, errors="coerce")
        A  = pd.to_numeric(A,  errors="coerce")
        E  = pd.to_numeric(E,  errors="coerce")
        g  = pd.to_numeric(g,  errors="coerce")


        wl = wl * 0.1                # Å → nm
        E  = E  * 1.2398e-4          # cm⁻¹ → eV（按你原公式）
        valid_mask = (
            enable_mask &
            np.isfinite(wl) &
            np.isfinite(A) & (A > 0) &
            np.isfinite(E) &
            np.isfinite(g)
        )

        wl = wl[valid_mask]
        A  = A[valid_mask]
        E  = E[valid_mask]
        g  = g[valid_mask]

        # ===============================
        # ⑤ 波段过滤
        # ===============================
        band_mask = (wl >= 200) & (wl <= 900)

        wl = wl[band_mask]
        A  = A[band_mask]
        E  = E[band_mask]
        g  = g[band_mask]

        # ===============================
        # ⑥ 转 numpy（现在 100% 安全）
        # ===============================
        wl = wl.to_numpy(dtype=float)
        A  = A.to_numpy(dtype=float)
        E  = E.to_numpy(dtype=float)
        g  = g.to_numpy(dtype=float)

        # 相对强度
        relative_intensity = rel_intensity(wl, A, E, g, T)

        matrix = np.column_stack((wl, relative_intensity, A, E, g))
        elements[element_name] = {"data": matrix}

    return elements, elements_list

# folder_path2 =r'D:\LIBS\ElementDetectation\11.10\Rareearth' #稀土元素光谱路径
# a,b=elements_database_pt2(folder_path2,10000)


# data=pd.read_csv(r'D:\LIBS\ElementDetectation\11.10\Rareearth\Spectrum\All.csv',header=0,skipinitialspace=True)
# data = data.fillna(0).to_numpy()
# data = np.nan_to_num(data, nan=0.0)
# x = data[:, 0]
# signal=data[:,1]


#debug
# folder_path =r'D:\LIBS\ElementDetectation\11.10\Rareearth' #稀土元素光谱路径
# file_list = glob.glob(os.path.join(folder_path, "*.csv"))
# # elements_list = [os.path.splitext(os.path.basename(f))[0] for f in file_list]
# elements,elements_list=elements_database_pt2(folder_path,T=3000) #元素库制作
# print(elements[elements_list[1]]['data'][:,1])
# elements_list={}
# folder_path = r'D:\LIBS\ElementDetectation\11.10\Rareearth' #稀土元素光谱路径 
# for element_name in elements_list: 
#     file_path = os.path.join(folder_path, element_name + ".csv")

#     # 读取 CSV
#     df = pd.read_csv(file_path, header=1, encoding="gbk")
#     # 只取偶数行
#     df = df.iloc[1::2].copy()
#     wl = df.iloc[:, 1]
#     wl = pd.to_numeric(wl, errors="coerce")
#     wl = wl * 0.1                # Å → nm

#     valid_mask = (np.isfinite(wl) )
#     wl = wl[valid_mask]
#     band_mask = (wl >= 200) & (wl <= 900)
#     wl = wl[band_mask]
#     wl = wl.to_numpy(dtype=float)




# plot=plt.figure(figsize=(10,6))
# plt.plot(x,signal,label='Total Signal',color='blue')
# plt.show()
