#本文件用于Nist谱线模拟
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os   


folder_path = r'D:\LIBS\ElementDetectation\11.10\Elements_database' #元素库路径
data=pd.read_csv(r'D:\LIBS\ElementDetectation\11.10\SpecSimuDatabase\Cr100_10000K.csv',header=0)
data=data.to_numpy()
wl=data[:,0]
intensity_sum=data[:,1]
intensity_atom=data[:,2]
intensity_ion=data[:,3]
#-----预备-----
#参数设置
T=10000 
kB=8.617330350e-5 #eV/K


#----必备函数定义----
#计算U（T） 返回U和U总和
def U_Calculate(g,A,E):
    U=np.zeros(len(g))
    for i in range(len(g)):
        U[i]=g[i]*np.exp(-E[i]/(kB*T))
    return U,np.sum(U)

#计算相对强度 返回相对强度
def rel_intensity(wl,A,E,g):
    U_T,U_T_sum=U_Calculate(g,A,E)
    rel_intensity=np.zeros(len(wl))
    for i in range(len(wl)):
        rel_intensity[i]=(A[i]*g[i]*np.exp(-E[i]/(kB*T)))/U_T_sum
    return rel_intensity

#元素库制作 返回elements字典和elements_list元素列表
def elements_database(folder_path):
    folder_path = r'D:\LIBS\ElementDetectation\10.10\Automatic-Elements-Recognition-for-LIBS-main\Elements_database' #元素库路径
    # 获取所有Excel文件路径
    file_list = glob.glob(os.path.join(folder_path, "*.csv"))
    # 获取元素名字（去掉路径和后缀）
    elements_list = [os.path.splitext(os.path.basename(f))[0] for f in file_list]
    #元素特征光谱制作
    elements={}
    for element_name in elements_list: 
        dfs = []
        file_path = os.path.join(folder_path, element_name + ".csv")  # 拼接完整路径
        df = pd.read_csv(file_path,header=1,encoding="gbk")  # 读取该元素的csv
        df=df.to_numpy()
        even_rows = df[1::2]
        wl=even_rows[:,1]*0.1
        A=even_rows[:,2]
        E=even_rows[:,3]*1.2398*10**(-4) #eV
        g=even_rows[:,7]
    #波段过滤 200-900nm
        mask = (wl >= 200) & (wl <= 900)
        wl = wl[mask]
        A = A[mask]
        E = E[mask]
        g = g[mask]
        
        relative_intensity=rel_intensity(wl,A,E,g)
        matrix = np.column_stack((wl, relative_intensity))
        elements[element_name] = { "data": matrix}
    return elements,elements_list



def Boltzmann_fit(I, A, g, E):
    y = np.log(I/ (g * A))

    # 线性拟合
    coefficients = np.polyfit(E, y, 1) #slope斜率 intercpet截距 拟合
    slope, intercept = coefficients
    T = -1/(slope * kB)  # 温度计算
    return coefficients, slope, intercept, T, y


#-----主程序-----
elements,elements_list=elements_database(folder_path)

element_name = "CrI"  # 元素名需与文件名一致，例如 "CrI.csv"
if element_name in elements:
    element_data = elements[element_name]["data"]
    wl_theo = element_data[:, 0]
    rel_intensity_theo = element_data[:, 1]

    
    print(f"\n=== {element_name} 理论谱线 ===")
    for i in range(len(wl_theo)):
        print(f"λ = {wl_theo[i]:.3f} nm,  相对强度 = {rel_intensity_theo[i]:.4e}")

    # ===== 绘制 CrI 模拟光谱 =====
    plt.figure(figsize=(10, 5))
    for i in range(len(wl_theo)):
        # 在每个谱线波长位置画一条垂直于x轴的线
        plt.vlines(x=wl_theo[i], ymin=0, ymax=rel_intensity_theo[i],
                color='r', linewidth=1.2, alpha=0.8)

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Relative Intensity (a.u.)")
    plt.title(f"NIST Simulated Spectrum of {element_name}")
    plt.xlim(200, 900)
    plt.ylim(0, 1.1 * max(rel_intensity_theo))
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.show()



# plt.plot(wl,intensity_atom)
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('Intensity (a.u.)')
# plt.title('NIST Spectrum Simulation of Cr')
# # plt.xlim(200,900)
# plt.ylim(0,1.1*max(intensity_sum))
# # plt.grid()
# plt.xticks(np.arange(200, 901, 100))
# plt.show()