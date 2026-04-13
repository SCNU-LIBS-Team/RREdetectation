# 本文件用于模拟光谱数据，并且生成csv文件
# 目前只是考量Cr元素的模拟光谱数据

#待处理任务
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#数据导入
data = pd.read_csv(r'D:\LIBS\ElementDetectation\11.10\Elements_database\CrII.csv',header=1,skipinitialspace=True,encoding="gbk")
data=data.to_numpy()
even_rows = data[1::2]

#参数设置
T=10000 #开尔文温度
kB=8.617330350e-5 #eV/K

# 数据处理
wl=even_rows[:,1]*0.1
A=even_rows[:,2]
E=even_rows[:,3]*1.2398*10**(-4) #eV
g=even_rows[:,7]
print(g)

#-----光谱模拟------

#计算U（T）
def U_Calculate(g,E):
    U=np.zeros(len(g))
    for i in range(len(g)):
        U[i]=g[i]*np.exp(-E[i]/(kB*T))
    return U,np.sum(U)

#计算相对强度
def rel_intensity(wl,A,E,g):
    U_T,U_T_sum=U_Calculate(g,E)
    rel_intensity=np.zeros(len(wl))
    for i in range(len(wl)):
        rel_intensity[i]=(A[i]*g[i]*np.exp(-E[i]/(kB*T)))/U_T_sum
    return rel_intensity



#调用
simulate_intensity=rel_intensity(wl,A,E,g)
print(simulate_intensity)

#显示
plt.scatter(wl,simulate_intensity,s=5,c='r')
print(np.sum(simulate_intensity))
plt.vlines(wl, 0, simulate_intensity, colors='r', linestyles='-', linewidth=1)
plt.xlabel('Wavelength(nm)')
plt.ylabel('Relative Intensity')
plt.title('Cr Persistent Lines')
plt.xlim(200, 900)
plt.show()



