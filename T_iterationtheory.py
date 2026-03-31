import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import time
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
data_folder = r'D:\LIBS\RREdetectation\Rockbasespectral_11'

run_mode = 'traverse'  # 'single' or 'traverse'
target_files = ['03134_95']  # 待测光谱文件名列表（不带扩展名）
T0 = 12500
candidate_mode = 'alterable'  # 'fix' or 'alterable'

# 迭代稳健性参数
T_MIN = 7000.0
T_MAX = 20000.0
MULTISTART_COUNT = 9
DAMPING_ALPHA = 0.35
TOP_K = 3
temperature_solver = 'iteration'  #brute_force
BRUTE_FORCE_STEP = 250.0
COMPARE_CSV_PATH = os.path.join(data_folder, "temperature_compare_results.csv")




 
def _candidate_score(confidence, r2):
    # 用置信度和R2偏离惩罚构建评分，减弱局部噪声峰的影响
    return float(confidence) - 0.35 * abs(float(r2) - 1.0)


#top-3算法 算出每轮迭代的候选元素及其对应温度和R2，综合评分选出加权目标温度(target_temperature)
def _pick_target_temperature(candidate_pool, elements_T_main, elements_R2_main, top_k=3):
    ranked = sorted(candidate_pool.items(), key=lambda kv: kv[1], reverse=True)
    top_items = ranked[:max(1, min(top_k, len(ranked)))]

    scores = []
    temperatures = []
    for elem, conf in top_items:
        T_elem = float(elements_T_main.get(elem, 0.0))
        r2_elem = float(elements_R2_main.get(elem, 0.0))
        scores.append(_candidate_score(conf, r2_elem))
        temperatures.append(T_elem)

    scores = np.asarray(scores, dtype=float)
    temperatures = np.asarray(temperatures, dtype=float)
    # softmax加权求目标温度
    shifted = scores - np.max(scores)
    weights = np.exp(shifted)
    weights = weights / max(np.sum(weights), 1e-12)
    return float(np.sum(weights * temperatures)), [e for e, _ in top_items]


def T_iteration_single(signal, x,T_initial, max_iterations=10, tolerance=1e-3, candidate_mode='fix',
                       t_min=7000.0, t_max=20000.0, alpha=0.35, top_k=3):
    T = float(np.clip(T_initial, t_min, t_max))
    top_candidate_element = None
    top_candidate_element_T = 0.0
    top_candidate_confidence = 0.0
    top_candidate_R2 = 0.0
    fixed_element = None
    best_score = -np.inf
    stable_rounds = 0

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
            signal,
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

        #获得目标温度
        target_temperature, ranked_elements = _pick_target_temperature(
            candidate_pool,
            elements_T_main,
            elements_R2_main,
            top_k=top_k,
        )

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

        current_score = _candidate_score(top_candidate_confidence, top_candidate_R2)
        if current_score > best_score:
            best_score = current_score

        # 阻尼更新目标温度
        previous_T = T
        T = (1.0 - alpha) * T + alpha * target_temperature
        T = float(np.clip(T, t_min, t_max))

        # 双条件收敛：温度变化足够小，且连续两轮稳定
        denom = max(abs(T), 1e-12)
        rel_change = abs(T - previous_T) / denom
        if rel_change < tolerance:
            stable_rounds += 1
        else:
            stable_rounds = 0

        if stable_rounds >= 2:
            print(f"迭代收敛于 T={T:.4f} K，迭代次数={iteration + 1}")
            break

        print(
            f"候选集(top-{min(top_k, len(ranked_elements))}): {ranked_elements}，"
            f"加权目标温度: {target_temperature:.4f} K，更新后温度: {T:.4f} K"
        )

    return T, top_candidate_element, top_candidate_element_T, top_candidate_confidence, best_score


def T_iteration(signal, x, T_initial, max_iterations=10, tolerance=1e-3, candidate_mode='fix',
                t_min=7000.0, t_max=25000.0, multistart_count=9, alpha=0.35, top_k=3):
    
    # 支持单初值和多初值；默认会在温度区间内自动多起点
    if isinstance(T_initial, (list, tuple, np.ndarray)):
        initial_points = [float(t) for t in T_initial]
    else:
        if multistart_count <= 1:
            initial_points = [float(T_initial)]
        else:
            initial_points = np.linspace(float(t_min), float(t_max), int(multistart_count)).tolist()

    best_result = None
    best_score = -np.inf

    for idx, t0 in enumerate(initial_points):
        print(color_text(f"\n[多起点] 第 {idx + 1}/{len(initial_points)} 个初值: T0={float(t0):.2f} K", BLUE))
        result = T_iteration_single(
            signal,
            x,
            t0,
            max_iterations=max_iterations,
            tolerance=tolerance,
            candidate_mode=candidate_mode,
            t_min=t_min,
            t_max=t_max,
            alpha=alpha,
            top_k=top_k,
        )
        current_score = float(result[4])
        if current_score > best_score:
            best_score = current_score
            best_result = result

    if best_result is None:
        return float(T_initial), None, 0.0, 0.0

    print(color_text(f"[多起点] 选择全局最优结果，评分={best_score:.4f}", GREEN))
    return best_result[0], best_result[1], best_result[2], best_result[3]


def Brute_Force_T_iteration(signal, x, t_min=7000.0, t_max=25000.0, t_step=250.0):
    if t_step <= 0:
        raise ValueError("t_step 必须大于 0")
    if t_max < t_min:
        raise ValueError("t_max 必须大于等于 t_min")

    # 峰位仅依赖原始光谱，与温度无关，放到循环外减少重复计算
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

    temperature_grid = np.arange(float(t_min), float(t_max) + 0.5 * float(t_step), float(t_step))

    best_scan_T = None
    best_element = None
    best_element_T = 0.0
    best_confidence = -np.inf
    best_R2 = 0.0

    for scan_idx, scan_T in enumerate(temperature_grid, start=1):
        elements_main, elements_main_list = elements_database_pt2(folder_path, float(scan_T))
        particle_main, elements_main, elements_T_main, elements_R2_main, elements_confidence_main = compute_element_confidence_shape(
            elements_main,
            peak_wl,
            peak_int,
            x,
            signal,
            scope=0.3,
        )

        if not elements_confidence_main:
            continue

        candidate_element = max(elements_confidence_main, key=elements_confidence_main.get)
        candidate_confidence = float(elements_confidence_main.get(candidate_element, 0.0))
        candidate_element_T = float(elements_T_main.get(candidate_element, 0.0))
        candidate_R2 = float(elements_R2_main.get(candidate_element, 0.0))

        print(
            f"[穷举 {scan_idx}/{len(temperature_grid)}] 扫描温度={float(scan_T):.2f} K，"
            f"最概然元素={candidate_element}，置信度={candidate_confidence:.4f}，R2={candidate_R2:.4f}"
        )

        if candidate_confidence > best_confidence:
            best_confidence = candidate_confidence
            best_scan_T = float(scan_T)
            best_element = candidate_element
            best_element_T = candidate_element_T
            best_R2 = candidate_R2

    if best_element is None:
        print("穷举结束：未找到有效候选元素")
        return float(t_min), None, 0.0, 0.0

    print(
        color_text(
            f"[穷举最优] 扫描温度={best_scan_T:.2f} K，最概然元素={best_element}，"
            f"元素温度={best_element_T:.4f} K，置信度={best_confidence:.4f}，R2={best_R2:.4f}",
            GREEN,
        )
    )
    return best_scan_T, best_element, best_element_T, best_confidence



# 实际运行：同一文件比对 iteration 与 brute_force
start = time.perf_counter()

all_csv_files = sorted(glob.glob(os.path.join(data_folder, "*.csv")))
if not all_csv_files:
    print(f"未找到CSV文件: {data_folder}")

if run_mode == 'single':
    csv_files = [os.path.join(data_folder, f"{name}.csv") for name in target_files]
elif run_mode == 'traverse':
    csv_files = all_csv_files
else:
    raise ValueError("run_mode 只能是 'single' 或 'traverse'")

compare_rows = []
iteration_total_time = 0.0
bruteforce_total_time = 0.0

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
        signal = data[:, 1]

        print(color_text(f"\n=== 文件: {file_name} | iteration ===", BLUE))
        t_iter_begin = time.perf_counter()
        iteration_result = T_iteration(
            signal,
            x,
            T0,
            max_iterations=12,
            tolerance=1e-5,
            candidate_mode=candidate_mode,
            t_min=T_MIN,
            t_max=T_MAX,
            multistart_count=MULTISTART_COUNT,
            alpha=DAMPING_ALPHA,
            top_k=TOP_K,
        )
        iteration_elapsed = time.perf_counter() - t_iter_begin
        iteration_total_time += iteration_elapsed

        print(color_text(f"\n=== 文件: {file_name} | brute_force ===", BLUE))
        t_bf_begin = time.perf_counter()
        brute_force_result = Brute_Force_T_iteration(
            signal,
            x,
            t_min=T_MIN,
            t_max=T_MAX,
            t_step=BRUTE_FORCE_STEP,
        )
        brute_force_elapsed = time.perf_counter() - t_bf_begin
        bruteforce_total_time += brute_force_elapsed

        print(
            f"对比结果: iteration_T={iteration_result[0]:.4f} K, "
            f"brute_force_T={brute_force_result[0]:.4f} K, "
            f"iteration耗时={iteration_elapsed:.4f} 秒, "
            f"brute_force耗时={brute_force_elapsed:.4f} 秒"
        )

        compare_rows.append(
            {
                "文件名": file_name,
                "iteration": float(iteration_result[0]),
                "brute_force": float(brute_force_result[0]),
            }
        )

    except Exception as e:
        print(f"\n=== 文件: {file_name} ===")
        print(f"处理失败: {e}")

compare_rows.append(
    {
        "文件名": "算法运算时间(秒)",
        "iteration": round(iteration_total_time, 6),
        "brute_force": round(bruteforce_total_time, 6),
    }
)

compare_df = pd.DataFrame(compare_rows, columns=["文件名", "iteration", "brute_force"])
compare_df.to_csv(COMPARE_CSV_PATH, index=False, encoding="utf-8-sig")
print(color_text(f"\n已输出对比结果: {COMPARE_CSV_PATH}", GREEN))

end = time.perf_counter()
print(f"总耗时: {end - start:.4f} 秒")