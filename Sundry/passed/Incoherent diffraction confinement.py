#题目：衍射受限非相干成像系统的光瞳为在边长为10的正方形以四个角为圆心分别挖掉4个半径为3的扇形
# 求其光瞳的二维图像和光学传递函数的三维图像。
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (for 3D)


#定义光瞳函数
#参数介绍：采样点N，物理长度L，扇形半径
def pupil_function(N, L, r):
    dx = L / N
    dy = dx
    #设置表格采样点0为其中心
    x = np.linspace(0, L - dx, N)
    y = np.linspace(0, L - dy, N)
    X, Y = np.meshgrid(x, y)
    
    # 定义正方形光瞳，为N*N的全1矩阵
    pupil = np.ones((N, N))
    #找到四个角落坐标
    corners=[(0, 0), (0, L- dy), (L-dx, L-dy), (L- dx, 0)]

    # 挖掉四个扇形
    #实现方法：将对应挖去部分赋值为0
    for (xc, yc) in corners:
        dx = X - xc
        dy = Y - yc
        # 计算点到角的距离
        dist = np.sqrt(dx*dx + dy*dy)

        # 判断“在角内”的方向：点位于方形内部相对于该角的象限
        #通过向量和距离来判断
        cond_x = (dx >= 0) if xc == 0.0 else (dx <= 0)
        cond_y = (dy >= 0) if yc == 0.0 else (dy <= 0)
        #bool判据
        inward_quadrant = cond_x & cond_y
        # 在该扇形（四分之一圆）范围内的点设为 0（挖掉）
        mask_cut = (dist <= r) & inward_quadrant
        pupil[mask_cut] = 0.0
    
    return pupil,x,y



#计算光学传递函数OTF
def optical_transfer_function(pupil):
    # OTF 是光瞳函数的自相关归一化
    pupil_fft=np.fft.fft2(pupil)
    amplitude=np.abs(pupil_fft)
    #自相关
    OTF = np.fft.ifft2(amplitude**2)
    #移频
    OTF = np.fft.fftshift(np.abs(OTF))
    OTF /= OTF.sum()  # 归一化
    return OTF


pupil1,x,y=pupil_function(512, 10, 3)
OTF = optical_transfer_function(pupil1)

#显示部分
fig = plt.figure(figsize=(14,6))
ax1 = fig.add_subplot(1,2,1)
im = ax1.imshow(pupil1, extent=(x[0], x[-1], y[0], y[-1]), origin='lower', cmap='gray')
ax1.set_title('Pupil 2D')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

ax2 = fig.add_subplot(1,2,2, projection='3d')
X, Y = np.meshgrid(x, y)
ax2.plot_surface(X, Y, OTF, cmap='viridis')
ax2.set_title('Optical Transfer Function (OTF)')
ax2.set_xlabel('fx')
ax2.set_ylabel('fy')
ax2.set_zlabel('OTF Amplitude')


plt.tight_layout()
plt.show()