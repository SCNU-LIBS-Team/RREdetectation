import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

#constant
kB=8.617330350e-5 #eV/K
T=10000

#function
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

def Boltzmann_fit(I,wl, A, g, E):
    y = np.log(I*wl/ (g * A))

    # 线性拟合
    coefficients = np.polyfit(E, y, 1) #slope斜率 intercpet截距 拟合
    slope, intercept = coefficients
    T = -1/(slope * kB)  # 温度计算
    return coefficients, slope, intercept, T, y


def Saha_Boltzmann_fit(I_ion, I_atom, wl_ion, wl_atom, A_ion, A_atom, g_ion, g_atom, E_ion, E_atom, chi):
    y = np.log((I_ion * wl_atom) / (I_atom * wl_ion) * (g_atom * A_atom) / (g_ion * A_ion))

    # 线性拟合
    coefficients = np.polyfit(E_ion - E_atom + chi, y, 1)
    slope, intercept = coefficients
    T = -1/(slope * kB)  # 温度计算
    return coefficients, slope, intercept, T, y


data=pd.read_csv(r'E:\工作文件\课题组激光诱导击穿光谱学习\LIBS-ElementRecogonise\10.22\Elements_database\CrI.csv',header=1,encoding="gbk")
df=data.to_numpy()
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
relative_intensity_copy=[140184,110430,86529,40022,30531,22560,8957,15061,20575]
relative_intensity_copy_Norm=relative_intensity_copy / np.sum(relative_intensity_copy)  # 归一化
relative_intensity=rel_intensity(wl,A,E,g)
relative_intensity_Norm=relative_intensity / np.sum(relative_intensity)  # 归一化
print(relative_intensity_Norm)
print(relative_intensity_copy_Norm)
#Boltzmann拟合
coefficients, slope, intercept, T, y = Boltzmann_fit(relative_intensity_copy_Norm,wl, A, g, E)
print(f"Fitted Temperature: {T} K")


#显示
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(8,10))


# Boltzmann拟合图
ax1.scatter(E, y, color='blue', label='Data Points')
ax1.plot(E, slope * E + intercept, color='red', label='Fitted Line')
ax1.set_xlabel('Excitation Energy (eV)')
ax1.set_ylabel('ln(I / (g * A))')
ax1.set_title('Boltzmann Plot')
ax1.legend()

# 用 vline 绘制每条谱线
for i in range(len(wl)):
    ax2.vlines(wl[i], 0, relative_intensity_Norm[i], color='blue', linewidth=1)

ax2.set_xlabel("Wl(nm)")
ax2.set_ylabel("RI(a.u.)")
ax2.set_title("CrI(T = 10000 K)")
ax2.grid(True, linestyle='--', alpha=0.5)


plt.tight_layout()
plt.show()

