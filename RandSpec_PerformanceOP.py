#该文件用来生成随机光谱数据，用于测试光谱分析算法的正确性和鲁棒性
import numpy as np
import pandas as pd
import os

def _read_csv_with_fallback(csv_path):
    for enc in ('utf-8', 'utf-8-sig', 'gbk', 'gb18030', 'latin1'):
        try:
            return pd.read_csv(csv_path, encoding=enc, header=0)
        except UnicodeDecodeError:
            continue
    raise ValueError(f'无法读取 CSV 文件: {csv_path}')


def _build_performance_result(df, df_contents, confidence_path):
    confidence = df.to_numpy()
    contents = df_contents.to_numpy()

    if confidence.shape[0] != contents.shape[0]:
        raise ValueError(f'confidence 和 contents 行数不一致，无法逐行计算 detection_rate: {confidence_path}')
    if confidence.shape[1] < 16 or contents.shape[1] < 16:
        raise ValueError(f'confidence 或 contents 列数不足，至少需要到第16列: {confidence_path}')

    # 输出CSV文件要求
    # 内容包含：光谱名称，检测率，误检率，误检元素，检出限
    element_names = df.columns[1:16].tolist()
    results = []
    for i in range(confidence.shape[0]):
        spectrum_name = confidence[i, 0]

        # 第2到第16列（15个稀土元素列，Python 切片为 1:16）逐列统计命中与误检
        confidence_row = pd.to_numeric(pd.Series(confidence[i, 1:16]), errors='coerce').fillna(0).to_numpy(dtype=float)
        contents_row = pd.to_numeric(pd.Series(contents[i, 1:16]), errors='coerce').fillna(0).to_numpy(dtype=float)
        confidence_nonzero = confidence_row != 0
        contents_nonzero = contents_row != 0

        # 命中：confidence 非零且 contents 对应列也非零
        num = int(np.sum(confidence_nonzero & contents_nonzero))
        # 误检：confidence 非零但 contents 对应列为零
        false_mask = confidence_nonzero & (~contents_nonzero)
        false_num = int(np.sum(false_mask))

        # 如果出现分母为零的情况，设置一个小的默认值（如0.001）以避免除零错误
        detection_rate = num / np.sum(contents_nonzero) if np.sum(contents_nonzero) > 0 else 0.001
        false_detection_rate = false_num / np.sum(confidence_nonzero) if np.sum(confidence_nonzero) > 0 else 0.001
        false_detection_elements = [element_names[j] for j, is_false in enumerate(false_mask) if is_false]
        false_detection_elements_str = ';'.join(false_detection_elements)

        # 检出限：在命中列中取 contents 的最小值
        hit_mask = confidence_nonzero & contents_nonzero
        detection_limit = float(np.min(contents_row[hit_mask])) if np.any(hit_mask) else np.nan
        results.append([spectrum_name, detection_rate, false_detection_rate, false_detection_elements_str, detection_limit])

    return pd.DataFrame(
        results,
        columns=['Spectrum Name', 'Detection Rate', 'False Detection Rate', 'False Detection Elements', 'Detection Limit'],
    )


def RandSepc_PerforOP(signal_path):
    contents_path = os.path.join(signal_path, "Randomrareearth_contents.csv")
    df_contents = _read_csv_with_fallback(contents_path)

    performance_targets = [
        (
            os.path.join(signal_path, "rareearth_confidence_results.csv"),
            os.path.join(signal_path, "performance_result.csv"),
        ),
        (
            os.path.join(signal_path, "rareearth_confidence_results_with_fit.csv"),
            os.path.join(signal_path, "performance_result_with_fit.csv"),
        ),
    ]

    for confidence_path, output_path in performance_targets:
        if not os.path.exists(confidence_path):
            print('未找到置信度 CSV，跳过性能评估: ' + confidence_path)
            continue

        df = _read_csv_with_fallback(confidence_path)
        results_df = _build_performance_result(df, df_contents, confidence_path)
        results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print('结果已保存到 ' + output_path)

