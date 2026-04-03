import numpy as np
import pandas as pd

confidence_path = 'D:/LIBS/RREdetectation/RandomSpectrum/Pt1/rareearth_confidence_results.csv'
contents_path = 'D:/LIBS/RREdetectation/RandomSpectrum/Pt1/Randomrareearth_contents.csv'
df = None
df_contents = None
for enc in ('utf-8', 'utf-8-sig', 'gbk', 'gb18030', 'latin1'):
    try:
        df = pd.read_csv(confidence_path, encoding=enc, header=0)
        df_contents = pd.read_csv(contents_path, encoding=enc, header=0)
        break
    except UnicodeDecodeError:
        continue

if df is None or df_contents is None:
    raise ValueError(f'无法读取 CSV 文件: {confidence_path} or {contents_path}')

# 转换为 numpy 数组
confidence = df.to_numpy()
contents = df_contents.to_numpy()


if confidence.shape[0] != contents.shape[0]:
    raise ValueError('confidence 和 contents 行数不一致，无法逐行计算 detection_rate')
if confidence.shape[1] < 12 or contents.shape[1] < 12:
    raise ValueError('confidence 或 contents 列数不足，至少需要到第12列')



#输出CSV文件要求
#内容包含：光谱名称，检测率，误检率，检出限
results = []
for i in range(confidence.shape[0]):
    spectrum_name = confidence[i, 0]

    # 第2到第12列（Python 切片为 1:12）逐列统计命中与误检
    confidence_row = pd.to_numeric(pd.Series(confidence[i, 1:12]), errors='coerce').fillna(0).to_numpy(dtype=float)
    contents_row = pd.to_numeric(pd.Series(contents[i, 1:12]), errors='coerce').fillna(0).to_numpy(dtype=float)
    confidence_nonzero = confidence_row != 0
    contents_nonzero = contents_row != 0

    # 命中：confidence 非零且 contents 对应列也非零
    num = int(np.sum(confidence_nonzero & contents_nonzero))
    # 误检：confidence 非零但 contents 对应列为零
    false_num = int(np.sum(confidence_nonzero & (~contents_nonzero)))

    detection_rate = num / 11.0
    false_detection_rate = false_num / 11.0
    # 检出限：在命中列中取 contents 的最小值
    hit_mask = confidence_nonzero & contents_nonzero
    detection_limit = float(np.min(contents_row[hit_mask])) if np.any(hit_mask) else np.nan
    results.append([spectrum_name, detection_rate, false_detection_rate, detection_limit])
print(results)