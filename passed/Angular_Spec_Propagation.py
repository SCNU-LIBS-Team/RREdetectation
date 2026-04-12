import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#对照片进行灰度处理和傅里叶变换（变换核为书上二维傅里叶变换核）

#定义二维傅里叶变换函数
def two_dimensional_fourier_transform(f):
    x_len, y_len = f.shape
    x = np.arange(x_len)
    y = np.arange(y_len)
    u = np.arange(x_len)
    v = np.arange(y_len)
    # 构建指数矩阵
    exp_x = np.exp(-2j * np.pi * np.outer(u, x) / x_len)
    exp_y = np.exp(-2j * np.pi * np.outer(v, y) / y_len)
    # 公式：F(u,v) = Σ_x Σ_y f(x,y) e^{-j2π(ux/Nx + vy/Ny)}
    F = exp_x @ f @ exp_y.T
    return F

#定义二维逆傅里叶变换函数
def inverse_two_dimensional_fourier_transform(F):

    x_len, y_len = F.shape
    x = np.arange(x_len)
    y = np.arange(y_len)
    u = np.arange(x_len)
    v = np.arange(y_len)
    exp_u = np.exp(2j * np.pi * np.outer(x, u) / x_len)
    exp_v = np.exp(2j * np.pi * np.outer(y, v) / y_len)
    f = exp_u @ F @ exp_v.T
    return f / (x_len * y_len)

#读取图片并转换为灰度图
img = Image.open("selfie.jpg").convert("L") # 读取图片并转换为灰度图
img_array = np.array(img, dtype=float)

#调整图片大小10mm*10mm
#设置物理尺寸概念（双三次插值重采样）
#最终输出img_array为512*512的二维数组   dx，dy采样间距mm
L = 10.0  # 物理边长，单位 mm
N = 256   # 重采样为 N x N
img_resized = img.resize((N, N), resample=Image.BICUBIC)
img_array = np.array(img_resized, dtype=np.float64) #转化为数组
#归一化
img_array = img_array - img_array.min()
img_array = img_array / img_array.max()
dx = L / N  # 采样间距，单位 mm
dy = dx


#角谱传播
def angularspec_propagation(Ui,l,dx,dy,wl):

    #复域场网格
    Ny,Nx=Ui.shape
    u=np.arange(Nx)
    v=np.arange(Ny)
    FX,FY=np.meshgrid(u,v)

    #传递函数
    k=2.0*np.pi/wl #波数mm^-1
    arg = 1.0 - (wl**2) * (FX**2 + FY**2)
    H = np.exp(1j * k * l * np.sqrt(arg.astype(np.complex128)))

    #傅里叶变换
    Ui_F=two_dimensional_fourier_transform(Ui) 

    #移频（低频在中间，高频在两边）
    Ui_F_shift=np.fft.fftshift(Ui_F)

    #传递函数*频谱
    Uo_F_shift=Ui_F_shift*H

    #逆移频
    Uo_F=np.fft.ifftshift(Uo_F_shift)
    #逆傅里叶变换
    Uo=inverse_two_dimensional_fourier_transform(Uo_F)
    return Uo


#调用
Uo_01=angularspec_propagation(img_array,0.1,dx,dy,0.000555) #传播距离2m，波长555nm可见光中心波长
Uo_1=angularspec_propagation(img_array,1,dx,dy,0.000555)
Uo_200=angularspec_propagation(img_array,200,dx,dy,0.000555)
#绘图
plt.figure(figsize=(12,8))
plt.subplot(1,4,1)
plt.imshow(img_array, cmap='gray')
plt.title("Original image")
plt.axis('off')

plt.subplot(1,4,2)
plt.imshow(np.abs(Uo_01), cmap='gray')
plt.title("l=0.1mm")
plt.axis('off')

plt.subplot(1,4,3)
plt.imshow(np.abs(Uo_1), cmap='gray')
plt.title("l=1mm")
plt.axis('off')

plt.subplot(1,4,4)
plt.imshow(np.abs(Uo_200), cmap='gray')
plt.title("l=200mm")
plt.axis('off')

plt.tight_layout()
plt.show()