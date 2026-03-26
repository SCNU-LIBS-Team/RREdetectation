import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from Elements_detectation import compute_element_confidence_shape
from Elements_Combfact import elements_database_pt2
from Wavelet_peakfinding import wavelet_peak_detection

kB=8.617330350e-5 #eV/K

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

#本部分为T_iteration，用于在检测基体元素时顺便迭代电子温度


folder_path = r'D:\LIBS\RREdetectation\Elements_database' 
data_folder = r'D:\LIBS\RREdetectation\Rockbasespectral_11_0.75eV'

run_mode = 'single'  # 'single' or 'traverse'
target_files = ['07840_95']  # 待测光谱文件名列表（不带扩展名）
T0=8700
candidate_mode = 'alterable'  # 'fix' or 'alterable'


 
def T_iteration(signal, x, intensity_sum, T_initial, max_iterations=10, tolerance=1e-3, candidate_mode='fix'):
    T = float(T_initial)
    top_candidate_element = None
    top_candidate_element_T = 0.0
    top_candidate_confidence = 0.0
    top_candidate_R2 = 0.0
    fixed_element = None

    if candidate_mode not in ('fix', 'alterable'):
        raise ValueError("candidate_mode 只能是 'fix' 或 'alterable'")

    for iteration in range(max_iterations):
        elements_main, elements_main_list = elements_database_pt2(folder_path, T)
        true_peak_idx, peak_wl, peak_int = wavelet_peak_detection(
            signal,
            x,
            wavelet='mexh',
            scales=np.arange(1, 11),
            neighbor=4,
            min_length=3,
            coeffi_threshold=700,
            window=5,
        )
        particle_main, elements_main, elements_T_main, elements_R2_main, elements_confidence_main = compute_element_confidence_shape(
            elements_main,
            peak_wl,
            peak_int,
            x,
            intensity_sum,
            scope=0.3,
        )

        if not elements_confidence_main:
            print("没有最概然元素")
            break

        valid_candidates = {
            elem: conf
            for elem, conf in elements_confidence_main.items()
            if float(elements_R2_main.get(elem, 0.0)) != 1.0
        }
        candidate_pool = valid_candidates if valid_candidates else elements_confidence_main

        if candidate_mode == 'fix':
            # 首轮锁定最概然元素，后续迭代不再改变元素身份
            if fixed_element is None:
                fixed_element = max(candidate_pool, key=candidate_pool.get)
                print(
                    f"第 {iteration + 1} 轮迭代，锁定最概然元素: {fixed_element}，"
                    f"此时温度为：{float(elements_T_main.get(fixed_element, 0.0))}"
                )

            if fixed_element not in elements_T_main:
                print(f"固定元素 {fixed_element} 在当前迭代结果中不存在，停止迭代")
                break

            top_candidate_element = fixed_element
        else:
            # 每轮允许最概然元素变化
            top_candidate_element = max(candidate_pool, key=candidate_pool.get)

        top_candidate_element_T = float(elements_T_main.get(top_candidate_element, 0.0))
        top_candidate_confidence = float(elements_confidence_main.get(top_candidate_element, 0.0))
        top_candidate_R2 = float(elements_R2_main.get(top_candidate_element, 0.0))
        print(
            f"最概然元素: {top_candidate_element}，对应温度: {top_candidate_element_T:.4f} K，"
            f"置信度: {top_candidate_confidence:.4f}，R2: {top_candidate_R2:.4f}"
        )

        # 避免除零，使用相对变化率判断收敛
        denom = max(abs(T), 1e-12)
        rel_change = abs(top_candidate_element_T - T) / denom
        if rel_change < tolerance:
            T = top_candidate_element_T
            print(f"迭代收敛于 T={T:.4f} K，迭代次数={iteration + 1}")
            break
        
        T = top_candidate_element_T+rel_change * (top_candidate_element_T-T)  

    return T, top_candidate_element, top_candidate_element_T, top_candidate_confidence


all_csv_files = sorted(glob.glob(os.path.join(data_folder, "*.csv")))
if not all_csv_files:
    print(f"未找到CSV文件: {data_folder}")

if run_mode == 'single':
    csv_files = [os.path.join(data_folder, f"{name}.csv") for name in target_files]
elif run_mode == 'traverse':
    csv_files = all_csv_files
else:
    raise ValueError("run_mode 只能是 'single' 或 'traverse'")

for csv_file in csv_files:
    file_name = os.path.basename(csv_file)
    try:
        if not os.path.exists(csv_file):
            print(f"\n=== 文件: {file_name} ===")
            print("文件不存在，跳过")
            continue

        data = pd.read_csv(csv_file, header=0, skipinitialspace=True)
        data = data.fillna(0).to_numpy()

        if data.shape[1] < 2:
            print(f"\n=== 文件: {file_name} ===")
            print("列数不足，至少需要两列（波长、强度），跳过")
            continue

        x = data[:, 0]
        intensity_sum = data[:, 1]
        signal = data[:, 1]

        print(f"\n=== 文件: {file_name} ===")
        T_plasma = T_iteration(signal, x, intensity_sum, T0, 10, 1e-5, candidate_mode)
        print(
            f"最终迭代得到的电子温度 T_plasma={T_plasma[0]:.4f} K, "
            f"最概然元素={T_plasma[1]}, 该元素对应的温度={T_plasma[2]:.4f} K, 置信度={T_plasma[3]:.4f}"
        )
    except Exception as e:
        print(f"\n=== 文件: {file_name} ===")
        print(f"处理失败: {e}")