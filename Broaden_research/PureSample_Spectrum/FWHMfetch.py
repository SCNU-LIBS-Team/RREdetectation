import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os
import sys
from scipy.optimize import brentq, curve_fit
from scipy.special import wofz

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from Wavelet_peakfinding import find_peaks_ridge,peak_correction,wavelet_peak_detection #寻峰

file_path = Path(r'D:\LIBS\RREdetectation\Broaden_research\PureSample_Spectrum')
df = pd.read_csv(os.path.join(file_path, 'Y1.asc'), header=None)

wavelengths = pd.to_numeric(df.iloc[:, 0], errors='coerce').to_numpy()
intensities = df.iloc[:, 1:4].apply(pd.to_numeric, errors='coerce').mean(axis=1).to_numpy()

valid_mask = np.isfinite(wavelengths) & np.isfinite(intensities)
wavelengths = wavelengths[valid_mask]
intensities = intensities[valid_mask]


def voigt_profile(x, sigma, gamma):
    z = (x + 1j * gamma) / (sigma * np.sqrt(2.0))
    return np.real(wofz(z)) / (sigma * np.sqrt(2.0 * np.pi))


def normalized_voigt(x, center, sigma, gamma):
    v0 = voigt_profile(0.0, sigma, gamma)
    return voigt_profile(x - center, sigma, gamma) / v0


def voigt_model(x, height, center, sigma, gamma, baseline, slope):
    return baseline + slope * (x - center) + height * normalized_voigt(x, center, sigma, gamma)


def voigt_fwhm(sigma, gamma):
    def half_height_error(dx):
        return voigt_profile(dx, sigma, gamma) / voigt_profile(0.0, sigma, gamma) - 0.5

    right = max(float(sigma), float(gamma), 1e-8)
    for _ in range(80):
        if half_height_error(right) < 0:
            return 2.0 * brentq(half_height_error, 0.0, right)
        right *= 2.0

    gaussian_fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma
    lorentz_fwhm = 2.0 * gamma
    return 0.5346 * lorentz_fwhm + np.sqrt(0.2166 * lorentz_fwhm**2 + gaussian_fwhm**2)


def estimate_raw_fwhm(x, y, peak_idx, baseline):
    peak_height = y[peak_idx] - baseline
    if peak_height <= 0:
        return np.nan

    half_height = baseline + 0.5 * peak_height

    left = peak_idx
    while left > 0 and y[left] > half_height:
        left -= 1

    right = peak_idx
    while right < len(y) - 1 and y[right] > half_height:
        right += 1

    if left == 0 or right == len(y) - 1:
        return np.nan

    def interpolate_crossing(i1, i2):
        if y[i2] == y[i1]:
            return x[i1]
        return x[i1] + (half_height - y[i1]) * (x[i2] - x[i1]) / (y[i2] - y[i1])

    left_x = interpolate_crossing(left, left + 1)
    right_x = interpolate_crossing(right - 1, right)
    return right_x - left_x


def fit_one_voigt_peak(wavelengths, intensities, center_guess, peak_height_guess, rank, fit_half_width=None):
    x = np.asarray(wavelengths, dtype=float)
    y = np.asarray(intensities, dtype=float)
    wl_step = float(np.nanmedian(np.abs(np.diff(x))))
    if not np.isfinite(wl_step) or wl_step <= 0:
        wl_step = 1e-3

    if fit_half_width is None:
        fit_half_width = max(0.25, 30.0 * wl_step)

    mask = (x >= center_guess - fit_half_width) & (x <= center_guess + fit_half_width)
    if np.count_nonzero(mask) < 9:
        nearest_idx = int(np.argmin(np.abs(x - center_guess)))
        left = max(0, nearest_idx - 20)
        right = min(len(x), nearest_idx + 21)
        mask = np.zeros_like(x, dtype=bool)
        mask[left:right] = True

    x_fit = x[mask]
    y_fit = y[mask]
    if x_fit.size < 6:
        return {
            'rank': rank,
            'peak_wl': float(center_guess),
            'peak_int': float(peak_height_guess),
            'success': False,
            'message': 'not enough points in fitting window',
            'fwhm': np.nan,
        }

    peak_idx_local = int(np.argmin(np.abs(x_fit - center_guess)))
    edge_n = max(3, x_fit.size // 8)
    left_edge = np.median(y_fit[:edge_n])
    right_edge = np.median(y_fit[-edge_n:])
    baseline0 = float(min(left_edge, right_edge, np.percentile(y_fit, 10)))
    x_span = float(x_fit[-1] - x_fit[0])
    y_span = float(np.max(y_fit) - np.min(y_fit))
    slope0 = float((right_edge - left_edge) / max(x_span, wl_step))
    height0 = float(max(y_fit[peak_idx_local] - baseline0, y_span, 1e-8))

    raw_fwhm = estimate_raw_fwhm(x_fit, y_fit, peak_idx_local, baseline0)
    if not np.isfinite(raw_fwhm) or raw_fwhm <= 0:
        raw_fwhm = max(8.0 * wl_step, fit_half_width / 4.0)

    sigma_min = max(wl_step / 20.0, 1e-8)
    sigma_max = max(fit_half_width, sigma_min * 10.0)
    sigma0 = float(np.clip(raw_fwhm / 2.35482, sigma_min * 2.0, sigma_max * 0.8))
    gamma0 = float(np.clip(raw_fwhm / 2.0, sigma_min * 2.0, sigma_max * 0.8))

    center_tol = min(fit_half_width * 0.35, max(8.0 * wl_step, 1e-6))
    height_upper = max(height0 * 5.0, y_span * 5.0, 1e-8)
    baseline_lower = float(np.min(y_fit) - max(height0, y_span))
    baseline_upper = float(np.max(y_fit))
    slope_limit = max(abs(y_span / max(x_span, wl_step)) * 10.0, 1e-8)

    p0 = np.array([height0, center_guess, sigma0, gamma0, baseline0, slope0], dtype=float)
    lower_bounds = np.array(
        [0.0, center_guess - center_tol, sigma_min, sigma_min, baseline_lower, -slope_limit],
        dtype=float,
    )
    upper_bounds = np.array(
        [height_upper, center_guess + center_tol, sigma_max, sigma_max, baseline_upper, slope_limit],
        dtype=float,
    )
    p0 = np.clip(p0, lower_bounds + 1e-12, upper_bounds - 1e-12)

    try:
        popt, _ = curve_fit(
            voigt_model,
            x_fit,
            y_fit,
            p0=p0,
            bounds=(lower_bounds, upper_bounds),
            maxfev=50000,
        )
        height, center, sigma, gamma, baseline, slope = popt
        fwhm = float(voigt_fwhm(sigma, gamma))
        dense_x = np.linspace(x_fit[0], x_fit[-1], 300)
        dense_y = voigt_model(dense_x, *popt)

        return {
            'rank': rank,
            'peak_wl': float(center_guess),
            'peak_int': float(peak_height_guess),
            'height': float(height),
            'center': float(center),
            'sigma': float(sigma),
            'gamma': float(gamma),
            'baseline': float(baseline),
            'slope': float(slope),
            'fwhm': fwhm,
            'x_fit': dense_x,
            'y_fit': dense_y,
            'half_y': float(baseline + 0.5 * height),
            'success': True,
            'message': 'ok',
        }
    except Exception as exc:
        return {
            'rank': rank,
            'peak_wl': float(center_guess),
            'peak_int': float(peak_height_guess),
            'success': False,
            'message': str(exc),
            'fwhm': np.nan,
        }


def fit_top_voigt_peaks(wavelengths, intensities, peak_wl, peak_int, top_n=10):
    peak_wl_np = np.asarray(peak_wl, dtype=float)
    peak_int_np = np.asarray(peak_int, dtype=float)
    valid_peak_mask = np.isfinite(peak_wl_np) & np.isfinite(peak_int_np)
    peak_wl_np = peak_wl_np[valid_peak_mask]
    peak_int_np = peak_int_np[valid_peak_mask]

    if peak_wl_np.size == 0:
        raise ValueError('No valid peaks found in peak_wl / peak_int.')

    top_count = min(top_n, peak_int_np.size)
    top_order = np.argsort(peak_int_np)[-top_count:][::-1]
    results = []

    for rank, peak_array_idx in enumerate(top_order, start=1):
        results.append(
            fit_one_voigt_peak(
                wavelengths,
                intensities,
                peak_wl_np[peak_array_idx],
                peak_int_np[peak_array_idx],
                rank,
            )
        )

    return results


true_peak_idx, peak_wl, peak_int =wavelet_peak_detection(            
            intensities,
            wavelengths,
            wavelet='mexh',
            scales=np.arange(1, 11),
            neighbor=4,
            min_length=3,
            coeffi_threshold=700,
            window=5,
)

voigt_results = fit_top_voigt_peaks(wavelengths, intensities, peak_wl, peak_int, top_n=10)

fwhm_table = pd.DataFrame(
    [
        {
            'rank_by_intensity': item['rank'],
            'detected_wavelength_nm': item['peak_wl'],
            'detected_intensity': item['peak_int'],
            'fit_center_nm': item.get('center', np.nan),
            'fwhm_nm': item['fwhm'],
            'sigma': item.get('sigma', np.nan),
            'gamma': item.get('gamma', np.nan),
            'success': item['success'],
            'message': item['message'],
        }
        for item in voigt_results
    ]
)
print('Top intensity peaks Voigt FWHM:')
print(fwhm_table.to_string(index=False))


plt.figure(figsize=(7, 5))
plt.plot(wavelengths, intensities, color='tab:blue', linewidth=1.5, label='Mean spectrum')

top_peak_wl = np.array([item['peak_wl'] for item in voigt_results], dtype=float)
top_peak_int = np.array([item['peak_int'] for item in voigt_results], dtype=float)
plt.scatter(
    top_peak_wl,
    top_peak_int,
    facecolors='none',
    edgecolors='tab:red',
    s=58,
    linewidths=1.7,
    label='Top 5 peaks',
    zorder=6,
)

voigt_label_used = False
fwhm_label_used = False


#拟合峰绘制
for item in voigt_results:
    plt.annotate(
        str(item['rank']),
        (item['peak_wl'], item['peak_int']),
        textcoords='offset points',
        xytext=(5, 6),
        fontsize=10,
        fontweight='semibold',
        color='tab:red',
    )

    if not item['success']:
        continue

    plt.plot(
        item['x_fit'],
        item['y_fit'],
        color='tab:orange',
        linewidth=1.5,
        linestyle='--',
        label='Voigt fit' if not voigt_label_used else None,
    )
    voigt_label_used = True

    left_fwhm = item['center'] - item['fwhm'] / 2.0
    right_fwhm = item['center'] + item['fwhm'] / 2.0
    plt.hlines(
        item['half_y'],
        left_fwhm,
        right_fwhm,
        colors='tab:green',
        linestyles=':',
        linewidth=1.7,
        label='FWHM' if not fwhm_label_used else None,
        zorder=7,
    )
    fwhm_label_used = True

for spine in plt.gca().spines.values():
    spine.set_linewidth(1.8)
for label in plt.gca().get_xticklabels():
    label.set_fontweight("semibold")
for label in plt.gca().get_yticklabels():
    label.set_fontweight("semibold")

plt.xlabel('Wavelength (nm)', fontsize=15, fontweight="semibold")
plt.ylabel('Intensity', fontsize=15, fontweight="semibold")
plt.title('Pure Sample Spectrum', fontsize=15, fontweight="semibold")

plt.minorticks_on()
plt.tick_params(axis='both', which='major', direction='in', top=True, right=True, width=2.0, length=6, labelsize=12)
plt.tick_params(axis='both', which='minor', direction='in', top=True, right=True, width=2.0, length=6, labelsize=12)
plt.grid(alpha=0.3)

plt.legend(loc="upper right", prop={"weight": "semibold", "size": 12}, frameon=False)
plt.tight_layout()
plt.show()
