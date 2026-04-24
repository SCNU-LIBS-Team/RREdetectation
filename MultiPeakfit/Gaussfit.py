import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy.signal as signal
import pywt


#数据导入部分
signal_path=r'D:\LIBS\RREdetectation\MultiPeakfit\4_27data.csv'
data=pd.read_csv(signal_path,header=0,encoding="gbk")

wl=data.iloc[:,0]
rel_int=data.iloc[:,1]
wl=pd.to_numeric(wl, errors='coerce')
rel_int=pd.to_numeric(rel_int, errors='coerce')

valid_mask=wl.notna() & rel_int.notna()
wl=wl[valid_mask]
rel_int=rel_int[valid_mask]

# 基线平移: 将强度整体减去最小值
rel_int = rel_int - rel_int.min()


# 自动找局部极大值
extrema_idx = []
for i in range(1, len(rel_int) - 1):
    is_local_max = rel_int.iloc[i] > rel_int.iloc[i - 1] and rel_int.iloc[i] > rel_int.iloc[i + 1]
    if is_local_max:
        extrema_idx.append(i)

# 可选: 手动设置目标峰位（单位: nm）。为空则使用自动找峰结果。
manual_peak_wl = [
    # 275.43, 275.57
# 305.85,306.20
]

if len(manual_peak_wl) > 0:
    wl_np_for_peak = wl.to_numpy(dtype=float)
    # 将手动峰位映射到最接近的采样点索引
    extrema_idx = sorted({int(np.argmin(np.abs(wl_np_for_peak - target_mu))) for target_mu in manual_peak_wl})

if len(extrema_idx) == 0:
    raise ValueError('未找到可用峰位，请检查数据或 manual_peak_wl 设置。')

peak_wl = wl.iloc[extrema_idx]
peak_int = rel_int.iloc[extrema_idx]


#连续小波（文章复现）
class CWTPeakFWHMEstimator:
    def __init__(self, wl, intensity, scale=10.0, threshold=0.01):

        self.wl = np.asarray(wl, dtype=float)
        self.intensity = np.asarray(intensity, dtype=float)
        self.scale = scale
        self.threshold = threshold
    
    #小波母函数
    def mexican_hat_wavelet(self,points,a):

        A = 2 / (np.sqrt(3 * a) * (np.pi**0.25))
        wsq = a**2
        x = np.linspace(-points//2, points//2, points)
        return A * (1 - x**2/wsq) * np.exp(-x**2/(2*wsq))

    #小波近似二阶导
    def cwt_second_derivative(self):

        scale_val = self.scale

        n = len(self.intensity)
        points = min(200, n)
        if points < 3:
            points = 3
        if points % 2 == 0:
            points -= 1

        wavelet = self.mexican_hat_wavelet(points=points, a=scale_val)
        cwt = np.convolve(self.intensity, wavelet, mode='same')
        return cwt

    def find_peaks_from_second_derivative(self,cwt_data, threshold=0.01):

        minima = signal.argrelextrema(cwt_data, np.less)[0]

        min_val = np.min(cwt_data)
        selected = [i for i in minima if abs(cwt_data[i]) > threshold * abs(min_val)]
        print (f"找到 {len(selected)} 个满足阈值条件的极小值点，原始极小值点数量: {len(minima)}")
        return np.array(selected, dtype=int)

    def remove_edge_artifacts(self,cwt_data, minima_indices):

        maxima_indices = signal.argrelextrema(cwt_data, np.greater)[0]
        valid_minima = []

        for i, m in enumerate(minima_indices):
            left_max = maxima_indices[maxima_indices < m]
            # 找右侧最近极大值
            right_max = maxima_indices[maxima_indices > m]

            has_left = len(left_max) > 0
            has_right = len(right_max) > 0

            # 判断是否为边界伪峰
            if i == 0:
                # 第一个极小值：必须有左极大值
                if not has_left:
                    continue

            if i == len(minima_indices) - 1:
                # 最后一个极小值：必须有右极大值
                if not has_right:
                    continue

            # 中间的极小值默认保留（论文假设artifact只在边界）
            valid_minima.append(m)
        return np.array(valid_minima, dtype=int)

    def estimate_fwhm(self, cwt_data, peak_indices, wavelength):
        fwhm_list = []
        wavelength = np.asarray(wavelength, dtype=float)
        for idx in peak_indices:
            # 左侧最大值
            left = idx
            while left > 1 and cwt_data[left-1] > cwt_data[left]:
                left -= 1
            # 右侧最大值
            right = idx
            while right < len(cwt_data)-2 and cwt_data[right+1] > cwt_data[right]:
                right += 1
            delta_x = abs(wavelength[right] - wavelength[left])
            fwhm = 0.7 * delta_x   # 论文FWHM经验公式
            fwhm_list.append(fwhm)
        return np.array(fwhm_list)
    
    
    def cwt_peak_detection(self):
        cwt_data = self.cwt_second_derivative()
        peaks = self.find_peaks_from_second_derivative(cwt_data, self.threshold)

        #去除伪影峰
        peaks = self.remove_edge_artifacts(cwt_data, peaks)
        peaks = np.asarray(peaks, dtype=int)
        fwhm = self.estimate_fwhm(cwt_data, peaks, self.wl)
        return peaks, fwhm, cwt_data


#估计两个峰的FWHM
estimator = CWTPeakFWHMEstimator(wl, rel_int, scale=0.48, threshold=0.01)
cwt_peaks, cwt_fwhm, cwt_data = estimator.cwt_peak_detection()

wl_np = np.asarray(wl, dtype=float)
intensity_np = np.asarray(rel_int, dtype=float)
peak_indices = np.asarray(extrema_idx, dtype=int)
peak_indices = peak_indices[(peak_indices >= 0) & (peak_indices < len(wl_np))]

top2_local = np.argsort(intensity_np[peak_indices])[-2:]
selected_idx = np.sort(peak_indices[top2_local])
fwhm_two = estimator.estimate_fwhm(np.asarray(cwt_data, dtype=float), selected_idx,wl_np)



# 高斯函数
def gaussian(x, a, mu, sigma):
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))



# 对每个局部极大点做单峰高斯拟合
fitted_params = []
total_fit = np.zeros_like(wl.to_numpy(dtype=float))
x_full = wl.to_numpy(dtype=float)
y_full = rel_int.to_numpy(dtype=float)
component_fits = []

peak_mu = wl.iloc[extrema_idx].to_numpy(dtype=float)
peak_height_upper = rel_int.iloc[extrema_idx].to_numpy(dtype=float)

if peak_mu.size > 0:
    x_span = float(x_full.max() - x_full.min())
    sigma_min = max(x_span / (len(x_full) * 10.0), 1e-4)
    sigma_max = max(x_span / 2.0, sigma_min * 10.0)

    def gaussian_sum_fixed_mu(x, amps, sigmas, mus):
        y_sum = np.zeros_like(x, dtype=float)
        for a_i, mu_i, sigma_i in zip(amps, mus, sigmas):
            y_sum += gaussian(x, a_i, mu_i, sigma_i)
        return y_sum

    amp_upper = np.maximum(peak_height_upper, 1e-8)
    amp_init = amp_upper * 0.5
    sigma_init = np.full(peak_mu.size, max(x_span / (8.0 * max(peak_mu.size, 1)), sigma_min), dtype=float)

    x0 = np.concatenate([amp_init, sigma_init])
    bounds = [(0.0, float(u)) for u in amp_upper] + [(sigma_min, sigma_max)] * peak_mu.size

    ratio_candidates = np.arange(0.0, 1.0001, 0.05)
    best_ratio = None
    best_full_rms = np.inf
    best_solution = None

    for ratio in ratio_candidates:
        # 拟合窗口: 左边界向左扩展 ratio*FWHM1，右边界向右扩展 ratio*FWHM2
        if peak_mu.size >= 2 and np.size(fwhm_two) >= 2:
            left_mu = float(wl_np[selected_idx[0]])
            right_mu = float(wl_np[selected_idx[1]])
            fit_mask = (x_full >= left_mu - ratio * float(fwhm_two[0])) & (x_full <= right_mu + ratio * float(fwhm_two[1]))
            if np.count_nonzero(fit_mask) < 3:
                fit_mask = np.ones_like(x_full, dtype=bool)
        else:
            fit_mask = np.ones_like(x_full, dtype=bool)

        x_fit = x_full[fit_mask]
        y_fit = y_full[fit_mask]

        def window_gaussian_rms(params):
            n_peak = peak_mu.size
            amps = params[:n_peak]
            sigmas = params[n_peak:]
            y_fit_all = gaussian_sum_fixed_mu(x_fit, amps, sigmas, peak_mu)
            return float(np.sqrt(np.mean((y_fit - y_fit_all) ** 2)))

        ratio_result = minimize(
            window_gaussian_rms,
            x0=x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 20000, 'ftol': 1e-12},
        )

        if not ratio_result.success:
            continue

        n_peak = peak_mu.size
        amps_full = ratio_result.x[:n_peak]
        sigmas_full = ratio_result.x[n_peak:]
        y_pred_full = gaussian_sum_fixed_mu(x_full, amps_full, sigmas_full, peak_mu)
        full_rms = float(np.sqrt(np.mean((y_full - y_pred_full) ** 2)))

        if full_rms < best_full_rms:
            best_full_rms = full_rms
            best_ratio = float(ratio)
            best_solution = ratio_result.x.copy()

    if best_solution is not None:
        n_peak = peak_mu.size
        best_amps = best_solution[:n_peak]
        best_sigmas = best_solution[n_peak:]

        for a_i, mu_i, sigma_i in zip(best_amps, peak_mu, best_sigmas):
            fitted_params.append((float(a_i), float(mu_i), float(sigma_i)))
            y_comp = gaussian(x_full, float(a_i), float(mu_i), float(sigma_i))
            component_fits.append(y_comp)
            total_fit += y_comp

        print(f'Best ratio: {best_ratio:.2f}, Full-spectrum RMS: {best_full_rms:.6f}')
    else:
        print('Global gaussian fitting failed for all ratio candidates.')



if fitted_params:
    fitted_params_arr = np.array(fitted_params, dtype=float)
    fitted_mu = fitted_params_arr[:, 1]
    fitted_amp = fitted_params_arr[:, 0]
    print('Fitted peaks (A, mu, sigma):')
    print(pd.DataFrame(fitted_params_arr, columns=['A', 'mu', 'sigma']))
else:
    fitted_mu = np.array([])
    fitted_amp = np.array([])



plt.figure(figsize=(7,5))
plt.plot(wl, rel_int, color='tab:blue', linewidth=1.8, label='wl-int')

if component_fits:
    for idx, y_comp in enumerate(component_fits):
        if idx == 0:
            plt.plot(wl, y_comp, color='tab:green', linewidth=1.2, alpha=0.85, label='Gaussian Components')
        else:
            plt.plot(wl, y_comp, color='tab:green', linewidth=1.2, alpha=0.85)
            
if fitted_params:
    plt.plot(wl, total_fit, color='tab:orange', linewidth=1.8, linestyle='--', label='Gaussian Sum Fit')
    plt.scatter(fitted_mu, fitted_amp, color='tab:green', s=28, label='Fitted Peaks', zorder=6)
    
    
# if lorentz_params:
#     lorentz_params_arr = np.array(lorentz_params, dtype=float)
#     print('Lorentz fitted peaks (A, x0, gamma):')
#     print(pd.DataFrame(lorentz_params_arr, columns=['A', 'x0', 'gamma']))

# if lorentz_component_fits:
#     for idx, y_l_comp in enumerate(lorentz_component_fits):
#         if idx == 0:
#             plt.plot(wl, y_l_comp, color='tab:purple', linewidth=1.1, alpha=0.75, label='Lorentz Components')
#         else:
#             plt.plot(wl, y_l_comp, color='tab:purple', linewidth=1.1, alpha=0.75)
# if lorentz_params:
#     plt.plot(wl, total_lorentz_fit, color='black', linewidth=1.6, linestyle='-.', label='Lorentz Sum Fit')


plt.xlabel('Wavelength (nm)', fontsize=15, fontweight="semibold")
plt.ylabel('Relative Intensity', fontsize=15, fontweight="semibold")
plt.title('Wavelength-Intensity Spectrum', fontsize=15, fontweight="semibold")
for spine in plt.gca().spines.values():
    spine.set_linewidth(1.8)
for label in plt.gca().get_xticklabels():
    label.set_fontweight("semibold")
for label in plt.gca().get_yticklabels():
    label.set_fontweight("semibold")
plt.scatter(peak_wl, peak_int, color='tab:red', s=36, label='Local Extrema', zorder=5)

plt.grid(alpha=0.3)
plt.legend(loc='upper right', prop={"weight": "semibold", "size": 12}, frameon=False)
plt.tight_layout()
plt.show()









