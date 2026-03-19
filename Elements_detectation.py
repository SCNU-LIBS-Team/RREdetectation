
import numpy as np
import pandas as pd
import glob
import os
import pywt
import matplotlib.pyplot as plt
from collections import defaultdict
from Wavelet_peakfinding import find_peaks_ridge,peak_correction,wavelet_peak_detection #寻峰
from Elements_Combfact import elements_database, elements_database_pt2,elements_database_lineswitch#元素库制作
from scipy.optimize import linear_sum_assignment

# terminal color helpers
RESET = "\033[0m"
BLUE = "\033[34m"
GREEN = "\033[32m"
YELLOW = "\033[33m"


def _enable_windows_ansi():
    if os.name != "nt":
        return True
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        if handle == 0 or handle == -1:
            return False

        mode = wintypes.DWORD()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)) == 0:
            return False

        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        if (mode.value & ENABLE_VIRTUAL_TERMINAL_PROCESSING) == 0:
            if kernel32.SetConsoleMode(handle, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING) == 0:
                return False
        return True
    except Exception:
        return False


COLOR_ENABLED = (
    os.getenv("NO_COLOR") is None
    and (
        bool(os.getenv("FORCE_COLOR"))
        or
        os.name != "nt"
        or _enable_windows_ansi()
        or bool(os.getenv("WT_SESSION"))
        or bool(os.getenv("ANSICON"))
        or bool(os.getenv("TERM"))
    )
)


def color_text(text, color):
    if not COLOR_ENABLED:
        return text
    return f"{color}{text}{RESET}"


#-----预备-----
#参数设置
T=10000
kB=8.617330350e-5 #eV/K


#----必备函数定义----
#玻尔兹曼图拟合 返回斜率，截距，温度，y
def Boltzmann_fit(I, wl, A, g, E):
    # Filter invalid / non-positive values to avoid log and fit failures
    mask = (
        np.isfinite(I) & np.isfinite(wl) & np.isfinite(A) & np.isfinite(g) & np.isfinite(E) &
        (I > 0) & (wl > 0) & (A > 0) & (g > 0)
    )
    I = I[mask]
    wl = wl[mask]
    A = A[mask]
    g = g[mask]
    E = E[mask]

    if len(E) < 2:
        return 0, 0, 0, 0, np.array([])

    y = np.log(I*wl / (g * A))

    try:
        slope, intercept = np.polyfit(E, y, 1)  # slope/intercept
    except np.linalg.LinAlgError:
        return 0, 0, 0, 0, y

    T = -1 / (slope * kB)

    y_fit = slope * E + intercept
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    R2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    return slope, intercept, T, R2, y

def Boltzmann_fit_iterative(I, wl, A, g, E,R2_threshold=1e-1,R2_start_threshold=0.97,max_iter=5,verbose=False):
    """Iterative Boltzmann fit with outlier removal."""
    I = np.array(I, float)
    wl = np.array(wl, float)
    A = np.array(A, float)
    g = np.array(g, float)
    E = np.array(E, float)

    mask = (
        np.isfinite(I) & np.isfinite(wl) & np.isfinite(A) & np.isfinite(g) & np.isfinite(E) &
        (I > 0) & (wl > 0) & (A > 0) & (g > 0)
    )
    I, wl, A, g, E = I[mask], wl[mask], A[mask], g[mask], E[mask]

    if len(E) < 2:
        return 0, 0, 0, 0, np.array([]), E, wl, I, A, g

    def _fit_once(Ev, yv):
        try:
            return np.polyfit(Ev, yv, 1)
        except np.linalg.LinAlgError:
            return None

    y = np.log(I * wl / (g * A))
    fit = _fit_once(E, y)
    if fit is None:
        return 0, 0, 0, 0, y, E, wl, I, A, g
    slope, intercept = fit
    y_pred = slope * E + intercept

    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    R2_init = 1 - ss_res / ss_tot if ss_tot != 0 else 0

    if verbose:
        print(f"[Init] R2={R2_init:.5f}")

    if R2_init >= R2_start_threshold:
        T = -1/(slope*kB)
        return slope, intercept, T, R2_init, y, E, wl, I, A, g

    R2_prev = R2_init

    for it in range(max_iter):
        y = np.log(I * wl / (g * A))
        y_pred = slope * E + intercept
        residuals = y - y_pred
        worst = np.argmax(np.abs(residuals))

        I = np.delete(I, worst)
        wl = np.delete(wl, worst)
        A = np.delete(A, worst)
        g = np.delete(g, worst)
        E = np.delete(E, worst)

        if len(E) < 2:
            break

        y = np.log(I * wl / (g * A))
        fit = _fit_once(E, y)
        if fit is None:
            break
        slope, intercept = fit
        y_pred = slope * E + intercept

        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        R2_new = 1 - ss_res / ss_tot if ss_tot != 0 else 0

        delta_R2 = abs(R2_new - R2_prev)
        if verbose:
            print(f"    R2={R2_new:.5f}, dR2={delta_R2:.6f}")

        if delta_R2 < R2_threshold:
            break

        R2_prev = R2_new

    T = -1/(slope*kB) if slope != 0 else 0
    return slope, intercept, T, R2_prev, np.log(I * wl / (g * A)), E, wl, I, A, g

def Boltzmann_plot(matched_i, matched_wl, element_A, element_E, element_g, element_wl,element_name,mode='normal'):

#参数说明:matched_theo匹配到的理论谱线  matched_exp匹配到的实验谱线  element_A元素的A  element_E元素的E  element_g元素的g  element_wl元素的波长列表  element_name元素名称
#用途说明：检测匹配点并且绘制玻尔兹曼图
    # ====== ② 玻尔兹曼图计算与绘制 ======
    if len(matched_wl) >= 2:  # 至少2个点才能线性拟合
        print(f"\n--- {element_name} 玻尔兹曼图 ---")
        
        # 提取匹配到的谱线参数（与 matched_exp 对应的理论参数）
        matched_wl = np.array([t[0] for t in matched_wl])
        matched_I = np.array([t[1] for t in matched_i])  # 实验强度
        # 从理论库中取对应的 A、E、g
        matched_idx = [np.argmin(np.abs(element_wl - wl)) for wl in matched_wl]
        A_sel = element_A[matched_idx]
        E_sel = element_E[matched_idx]
        g_sel = element_g[matched_idx]


        matched_I = np.array(matched_I, dtype=float)
        A_sel = np.array(A_sel, dtype=float)
        g_sel = np.array(g_sel, dtype=float)
        E_sel = np.array(E_sel, dtype=float)
        if mode=='normal':
        # 玻尔兹曼拟合
            slope, intercept, T_fit,R2, y_full = Boltzmann_fit(matched_I, matched_wl,A_sel, g_sel, E_sel)
            # slope, intercept, T_fit,R2, y_full, y_used = Boltzmann_fit_iterative(matched_I, matched_wl,A_sel, g_sel, E_sel,R2_start_threshold=0.97,max_iter=5,verbose=False)
            # print(f"拟合温度 T = {T_fit:.2f} K, 斜率 = {slope:.3f}")
            plt.figure(figsize=(6,4))
            plt.scatter(E_sel, y_full, c='r', label='Used Points')
            plt.plot(E_sel, slope * E_sel + intercept, 'b--',label=f'Fit T={T_fit:.1f} K, R2={R2:.3f}')
            plt.xlabel('E (eV)')
            plt.ylabel('ln(I / (g·A))')
            plt.title(f'{element_name} Boltzmann Plot')
            plt.legend()
            plt.grid(True, alpha=0.3)
            # plt.minorticks_on()
            plt.tick_params(axis='both', which='both', direction='in',top=True, right=True)
        if mode=='iterative':
        #绘图
            slope, intercept, T_fit, R2, y_used, E_used, wl_used, I_used, A_used, g_used = \
                Boltzmann_fit_iterative(matched_I, matched_wl, A_sel, g_sel, E_sel,
                                        R2_start_threshold=0.1, max_iter=1, verbose=False)

            plt.figure(figsize=(6,4))
            plt.scatter(E_used, y_used, c='r', label='Used Points')
            plt.plot(E_used, slope * E_used + intercept, 'b--',label=f'Fit T={T_fit:.1f} K, R2={R2:.3f}')
            plt.xlabel('E (eV)')
            plt.ylabel('ln(I / (g·A))')
            plt.title(f'{element_name} Boltzmann Plot')
            plt.legend()
            plt.grid(True, alpha=0.3)
            # plt.minorticks_on()
            plt.tick_params(axis='both', which='both', direction='in',top=True, right=True)
        
    else:
        print(f"{element_name} 匹配峰数不足，无法绘制玻尔兹曼图。")


        
#匈牙利算法线匹配策略
def match_spectral_lines(theo_wl, theo_int, exp_wl, exp_int, scope):

    T = len(theo_wl)
    E = len(exp_wl)
    N = max(T, E) 
    cost = np.zeros((N, N), dtype=float)
    BIG = 1e6
    cost[:] = BIG
    for i in range(T):
        for j in range(E):
            cost[i, j] = abs(theo_wl[i] - exp_wl[j])

    #  匈牙利算法
    row_ind, col_ind = linear_sum_assignment(cost)

    theo_vec = []
    exp_vec = []

    matched_theo = []
    matched_exp = []

    for i, j in zip(row_ind, col_ind):

        if i < T:            # 这是一个真实的理论峰
            if j < E:        # 实验峰也是真实的
                diff = abs(theo_wl[i] - exp_wl[j])

                if diff <= scope:
                    # 匹配成功
                    theo_vec.append(theo_int[i])
                    exp_vec.append(exp_int[j])

                    matched_theo.append((theo_wl[i], theo_int[i]))
                    matched_exp.append((exp_wl[j], exp_int[j]))
                else:
                    # 匹配距离太大 → 当作未匹配
                    theo_vec.append(0)
                    exp_vec.append(0)

            else:
                # 实验峰不存在（补的虚拟列）→ 未匹配
                theo_vec.append(0)
                exp_vec.append(0)

    theo_vec = np.array(theo_vec)
    exp_vec = np.array(exp_vec)

    return theo_vec, exp_vec, matched_theo, matched_exp

def match_spectral_lines_weighted(theo_wl, theo_int, exp_wl, exp_int, scope=0.2, max_high_intensity=None, alpha=1.0, beta=1.0):
    """
    匈牙利算法改进：优先匹配高强度实验峰
    
    Parameters:
        theo_wl, theo_int : 理论谱线波长和强度
        exp_wl, exp_int   : 实验谱线波长和强度
        scope             : 匹配容差 (nm)
        max_high_intensity: 限制参与匹配的高强度峰数量
        alpha, beta       : 成本权重，cost = alpha*|wl_diff| - beta*exp_intensity
    """
    exp_wl = np.array(exp_wl, dtype=float)
    exp_int = np.array(exp_int, dtype=float)
    theo_wl = np.array(theo_wl, dtype=float)
    theo_int = np.array(theo_int, dtype=float)

    # 按实验强度排序，选出前 N 个强峰
    exp_idx_use = np.arange(len(exp_wl))
    if max_high_intensity is not None and len(exp_wl) > max_high_intensity:
        exp_idx_use = np.argsort(-exp_int)[:max_high_intensity]
    
    exp_wl_sel = exp_wl[exp_idx_use]
    exp_int_sel = exp_int[exp_idx_use]
    
    T = len(theo_wl)
    E = len(exp_wl_sel)
    N = max(T, E)
    BIG = 1e6
    
    # 构建成本矩阵
    cost = np.full((N, N), BIG, dtype=float)
    for i in range(T):
        for j in range(E):
            diff = abs(theo_wl[i] - exp_wl_sel[j])
            if diff <= scope:
                cost[i, j] = alpha * diff - beta * exp_int_sel[j]  # 波长差减去强度加权
    
    # 匈牙利算法
    row_ind, col_ind = linear_sum_assignment(cost)
    
    matched_theo = []
    matched_exp = []
    theo_vec = []
    exp_vec = []
    
    for i, j in zip(row_ind, col_ind):
        if i < T and j < E and cost[i,j] < BIG:
            # 匹配成功
            matched_theo.append((theo_wl[i], theo_int[i]))
            matched_exp.append((exp_wl_sel[j], exp_int_sel[j]))
            theo_vec.append(theo_int[i])
            exp_vec.append(exp_int_sel[j])
        else:
            # 未匹配
            theo_vec.append(0)
            exp_vec.append(0)
    
    return np.array(theo_vec), np.array(exp_vec), matched_theo, matched_exp

#施工中的判断策略
def confidence_score(base_elem,element_distance,element_T,element_R2,element_linecounts,final_T,final_R2,final_lc,final_distance,elements_confidence):
    """
    使用说明:base_elem:元素名称列表 element_distance: 元素距离列表 element_T: 元素温度列表 element_R2: 元素R2列表 
    """
    for base_elem, T in element_T.items():
        valid_T = [t for t in T ] #初步验证
        if valid_T:
            TRCD_pairs = []
            R2=element_R2[base_elem]
            LC=element_linecounts[base_elem]
            D=element_distance[base_elem]

            for t,r2,lc,d in zip(T, R2, LC, D):
                TRCD_pairs.append((t, r2, lc,d))

            if TRCD_pairs: 
                filterd_pairs = [pair for pair in TRCD_pairs if pair[0] > 5000 and pair[0] < 20000 ]
                if filterd_pairs:
                    best=max(filterd_pairs, key=lambda x: x[1])
                else:
                    best=(0,0,0,0)
                final_T[base_elem]= best[0]
                final_R2[base_elem]= best[1]
                final_lc[base_elem]= best[2]
                final_distance[base_elem]= best[3] 

    
    for elem, distances in final_distance.items():
        if distances<10000 and final_R2[elem]>0:
            elements_confidence[elem]=np.exp(-1.5*distances/final_R2[elem]) #指数映射
        else:
            elements_confidence[elem]=0

def compute_element_confidence_shape(elements, peak_wl, peak_int,global_wl,global_intensity,scope=0.2,plot=False,target='KI'):
    """
    方案二：用理论和实验谱形的欧几里得距离作为相似度
    elements: 元素数据库 { "ElemI": {"data": [wl, intensity]} }
    peak_wl: 实验寻峰得到的峰位
    peak_int: 实验寻峰得到的峰强度
    scope: 容许匹配窗口 (nm),默认1nm
    """

    match_results = {}
    element_distance = defaultdict(list)
    element_T = defaultdict(list)
    element_R2 = defaultdict(list)
    element_linecounts = defaultdict(list)
    final_results = {} #元素层面显示
    final_T={}
    final_R2={}
    final_lc={}
    final_distance={}
    elements_confidence={}
    Boltzmann_T={}
    Boltzmann_R2={}
    Boltzmann_linecounts={}


    #遍历每一个粒子
    for element_name, element_data in elements.items():
        element_matrix = element_data["data"]
        element_wl = element_matrix[:, 0]
        element_intensity = element_matrix[:, 1]
        element_A=element_matrix[:,2]
        element_E=element_matrix[:,3]
        element_g=element_matrix[:,4]

        #reset
        matched_I=[]
        matched_wl=[]
        wl_iterative=[]
        I_iterative=[]


        theo_vec, exp_vec, matched_theo, matched_exp = match_spectral_lines_weighted(element_wl, element_intensity, peak_wl, peak_int, scope)
        theo_vec = np.array(theo_vec)
        exp_vec = np.array(exp_vec)
        N_total = len(element_wl)
        N_matched = len(matched_exp)
        match_ratio = N_matched / N_total if N_total > 0 else 0 # 匹配率
        

        # 归一化
        if np.sum(theo_vec) > 0:
            theo_vec = theo_vec / np.sum(theo_vec)
        if np.sum(exp_vec) > 0:
            exp_vec = exp_vec / np.sum(exp_vec)

        O_distance =(np.sqrt(np.sum((theo_vec - exp_vec) ** 2)))/(0.03 + match_ratio)  # 考虑匹配率的影响 0.03防止除0   
        if O_distance ==0: #完全没谱线或者只有一条谱线的时候
            O_distance=1e+4

        #BoltzmannT，R2计算
        if len(matched_theo) >= 2:
            # 提取匹配到的谱线参数（与 matched_exp 对应的理论参数）
            matched_wl = np.array([t[0] for t in matched_theo])
            matched_I = np.array([t[1] for t in matched_exp])  # 实验强度

            # 从理论库中取对应的 A、E、g
            matched_idx = [np.argmin(np.abs(element_wl - wl)) for wl in matched_wl]
            slope, intercept, T_fit, R2, y  = Boltzmann_fit(matched_I, matched_wl, element_A[matched_idx], element_g[matched_idx], element_E[matched_idx])
            slope,intecept,T_fit_iterative,R2_itertative,y,E_iterative,wl_iterative,I_iterative,A_iterative,g_iterative=Boltzmann_fit_iterative(matched_I, matched_wl, element_A[matched_idx], element_g[matched_idx], element_E[matched_idx],R2_start_threshold=0.97, max_iter=3, verbose=False)
            
            Boltzmann_T[element_name] = T_fit
           
            Boltzmann_R2[element_name] = R2
            Boltzmann_linecounts[element_name]= len(matched_theo)

        else:
            Boltzmann_T[element_name] = 0
            Boltzmann_R2[element_name] = 0
            Boltzmann_linecounts[element_name]= 0

        if element_name == target and plot:
            plt.figure(figsize=(8,4))

           # print(len(E_iterative))
        # 全部理论谱线（浅蓝）
            all_theo_intensity = element_intensity / np.sum(element_intensity)
            for wl, inten_norm in zip(element_wl, all_theo_intensity):
                plt.vlines(wl, 0, inten_norm,
                        color='lightblue', alpha=0.5,
                        label='All Theoretical' if wl==element_wl[0] else "")
                plt.legend(loc='upper right')


            # 理论匹配谱线（蓝）      
            if matched_theo:
                matched_theo_intensity = np.array([inten for _, inten in matched_theo])
                matched_theo_norm = matched_theo_intensity / np.sum(matched_theo_intensity)
                for (wl, _), inten_norm_theo in zip(matched_theo, matched_theo_norm):
                    plt.vlines(wl, 0, inten_norm_theo,
                            color='b', alpha=0.7,
                            label='Matched Theoretical' if wl==matched_theo[0][0] else "")
                    plt.legend(loc='upper right')

            # --- 匹配成功的实验谱线（红色） ---
            if matched_exp:
                matched_exp_intensity = np.array([inten for _, inten in matched_exp])
                matched_exp_norm = matched_exp_intensity / np.sum(matched_exp_intensity)
                matched_exp_normalized = [(wl, inten_norm_exp)
                          for (wl, _), inten_norm_exp in zip(matched_exp, matched_exp_norm)]
                
                for (wl, _), inten_norm_exp in zip(matched_exp, matched_exp_norm):
                    plt.vlines(wl, 0, inten_norm_exp,
                            color='r', alpha=0.7,
                            label='Matched Experimental' if wl==matched_exp[0][0] else "")
                    plt.legend(loc='upper right')
            # plt.minorticks_on()
            plt.tick_params(axis='both', which='both', direction='in',top=True, right=True)
                    
            ### --- 新增波形标注逻辑 --- ###
            plt.figure(figsize=(10,4))
            plt.plot(global_wl, global_intensity, color='black', lw=1, label='Original Spectrum')

            # 标出所有理论谱线位置（浅蓝色线）
            for wl in element_wl:
                plt.axvline(wl, color='cyan', alpha=0.3)

            # 标出匹配到的实验峰（红色点）
            for wl, inten in matched_exp:
                plt.scatter(wl, inten, color='red', s=25)

            plt.title('Original Spectrum with CrII Peaks Marked')
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Intensity')

            plt.legend(loc='upper right')


            plt.title(f'Matched Stick Spectrum for {element_name}')
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Normalized Intensity')
            # plt.minorticks_on()
            plt.tick_params(axis='both', which='both', direction='in',top=True, right=True)

            Boltzmann_plot(matched_exp, matched_theo, element_A, element_E, element_g, element_wl,element_name,mode='normal')
            iterative_combined = np.column_stack((wl_iterative, I_iterative))
            Boltzmann_plot(iterative_combined, iterative_combined, A_iterative, E_iterative, g_iterative, wl_iterative,element_name+"_iterative", mode='iterative')
            plt.show()

        match_results[element_name] = O_distance
        base_elem = ''.join([c for c in element_name if not c.isdigit() and c not in ["I","V"]])
        element_T[base_elem].append(Boltzmann_T[element_name])
        element_R2[base_elem].append(Boltzmann_R2[element_name])
        element_linecounts[base_elem].append(Boltzmann_linecounts[element_name])
        element_distance[base_elem].append(O_distance)
    # print(element_T)


#筛选
#元素距离筛选
    for base_elem, distances in element_distance.items():
        min_distance = min(distances)
        if min_distance < 47.13333:  
            final_results[base_elem] = min_distance
        else:
            final_results[base_elem] = np.mean(distances)
            
#元素T和R2输出
    for base_elem, Ts in element_T.items():
        # 过滤出大于0的温度
        valid_T = [t for t in Ts if t > 0]

        if valid_T:
            # 找出所有符合条件的 (T, R2) 对
            TRC_pairs = []
            if base_elem in element_R2:
                R2s = element_R2[base_elem]
                LCs=element_linecounts[base_elem]
                # 遍历 Ts 列表，筛选出有效温度对应的 R²
                for t, r2 ,lc in zip(Ts, R2s, LCs):
                    # if t > 0 and r2!=1:  # 初筛T>0,R2!=1
                    # if t>0 and lc>2:
                    if t>0 :
                        TRC_pairs.append((t, r2, lc))
            
            if TRC_pairs: 
                # 选出 R² 最大的那一组
                selected_T, selected_R2, selected_LC = max(TRC_pairs, key=lambda x: x[1])
            else:
                # 如果没有 R² 对应信息，就退化为取最小温度
                selected_T = min(valid_T)
                selected_R2 = 0
                selected_LC=0

            # 保存结果
            final_T[base_elem] = selected_T
            final_R2[base_elem] = selected_R2

        else: #如果element_T同时为空
            # if base_elem=='Si': #特殊元素判据
            #     print(f"{base_elem}没有有效温度，无法计算置信度。")
            final_T[base_elem] = 0
            final_R2[base_elem] = 0
          
#反归一化置信度输出
    for elem, distances in final_results.items():
        if elem=='Ca': #特殊元素判据
            print(f"{elem}的距离为{distances}，R2为{final_R2[elem]}，T为{final_T[elem]}")
        if distances<10000 and final_R2[elem]>0:
            #elements_confidence[elem]=1/(1+distances) #倒数映射
            elements_confidence[elem]=np.exp(-1.5*distances/final_R2[elem]) #指数映射
            if final_T[elem]<5000 or final_T[elem]>20000: #电子温度判据
                elements_confidence[elem]=0
        else:
            elements_confidence[elem]=0
            # if elem=='Ca': #特殊元素判据
            #    print('1')
    return match_results,final_results,final_T,final_R2,elements_confidence

 

 #-----数据导入-----

#数据库导入
folder_path = r'D:\LIBS\RREdetectation\Elements_database' #元素库路径
folder_path2 =r'D:\LIBS\RREdetectation\Rareearth_pt3' #稀土元素光谱路径

#attention:elements_database_pt2 header=1 

#-----主程序-----
elements_main,elements_main_list=elements_database_pt2(folder_path,T) #通过调节path/path2可以达成基础元素还是稀土元素mode


#print(elements_main)
signal_path1= r'D:\LIBS\RREdetectation\SpecSimuDatabase' #普通元素光谱数据库   a.t%
signal_path2= r'D:\LIBS\RREdetectation\Rareearth\Spectrum' #稀土元素光谱100%   a.t%
signal_path3= r'D:\LIBS\RREdetectation\RREs' #岩石基体95%+稀土元素光谱5%   a.t%
signal_path4= r'D:\LIBS\RREdetectation\RREs\last3' #Sm、Tb、Gd最后三种的高接纳度测试
signal_path5= r'D:\LIBS\RREdetectation\Rockbasespectral' #八大岩石基体元素检测

target_path=signal_path5

I_file_list = glob.glob(os.path.join(target_path, "*.csv"))
I_elements_list = [os.path.splitext(os.path.basename(f))[0] for f in I_file_list]

target_files=['070171_95'] #待测光谱文件名列表（不带扩展名）
target_element='Pr'
specifybotton = False  # True: 遍历全部文件，仅输出目标元素；False: 只跑 target_files，输出全部元素 （全文件，单元素）
checkallbutton=False#是否检测文件内的全部光谱 （全文件）
plotbotton=False#是否绘图展示Boltzmann图
plottarget='TiII' #Boltzmann图绘制目标元素



# 绘图模式开启时强制关闭 specify、checkall，仅输出目标图像但全量跑文件
if plotbotton:
    specifybotton = False
    checkallbutton = False

# 根据模式选择要处理的文件
if specifybotton:
    files_to_process = I_elements_list
elif checkallbutton:
    files_to_process = I_elements_list
else:
    files_to_process = [name for name in I_elements_list if name in target_files]

for I_element_name in files_to_process:
 
    data=pd.read_csv(os.path.join(target_path, I_element_name + ".csv"),header=0,skipinitialspace=True)#待测光谱路径
    data = data.fillna(0).to_numpy()
    data = np.nan_to_num(data, nan=0.0)
    x = data[:, 0]
    intensity_sum=data[:,1]
    signal=data[:,1]
    intensity_ionized=data[:,3]
    true_peak_idx, peak_wl, peak_int = wavelet_peak_detection(signal,x,wavelet='mexh', scales=np.arange(1, 11), 
                               neighbor=4, min_length=3, coeffi_threshold=700, window=5)#峰值校正
    
    #基体元素检测
    particle_main,elements_main,elements_T_main,elements_R2_main,elements_confidence_main=compute_element_confidence_shape(elements_main, peak_wl, peak_int,x,intensity_sum,
                                                                                          scope=0.2,plot=plotbotton,target=plottarget)
    
    print(elements_confidence_main)
    elements_rockmain = []
    for elem, conf in elements_confidence_main.items():
        if conf>0.7: #置信度阈值
            elements_rockmain.append(elem)
    print(elements_rockmain)
 

    #elements_database_line_switch header=1
    elements_rareearth,elements_rareearth_list=elements_database_lineswitch(folder_path2,T,elements_rockmain,LineSwitchMode=False) 
    particle_result,elements_result,elements_T,elements_R2,elements_confidence=compute_element_confidence_shape(elements_rareearth, peak_wl, peak_int,x,intensity_sum,
                                                                                                scope=0.2,plot=plotbotton,target=plottarget)
    
    
    # print(elements_result,type(elements_result))
    # print(elements_confidence,type(elements_confidence))



    print("\n---" ,I_element_name, "---") 
    # 元素+置信度
    print("--- 元素层面（距离 + 置信度） ---")
    sorted_elems = sorted(elements_result.keys(), key=lambda x: elements_result[x])
    #输出显示部分
    if specifybotton:
        for elem in sorted_elems:
            if elem != target_element:
                continue
            dist = elements_result.get(elem, np.nan)
            conf = elements_confidence.get(elem, 0)
            T = elements_T.get(elem, 0)
            R2 = elements_R2.get(elem, 0)
            temp_text = color_text(f"温度={T:<8.4f}", BLUE)
            r2_text = color_text(f"R2 = {R2:<8.4f}", YELLOW)
            conf_text = color_text(f"置信度 = {conf:<8.4f}", GREEN)
            print(f"{elem:<6s} 平均距离 = {dist:<8.4f} | {temp_text} | {r2_text} | {conf_text}")
            break
    else:
        for elem in sorted_elems:
            dist = elements_result.get(elem, np.nan)
            conf = elements_confidence.get(elem, 0)
            T = elements_T.get(elem, 0)
            R2 = elements_R2.get(elem, 0)
            temp_text = color_text(f"温度={T:<8.4f}", BLUE)
            r2_text = color_text(f"R2 = {R2:<8.4f}", YELLOW)
            conf_text = color_text(f"置信度 = {conf:<8.4f}", GREEN)
            print(f"{elem:<6s} 平均距离 = {dist:<8.4f} | {temp_text} | {r2_text} | {conf_text}")






