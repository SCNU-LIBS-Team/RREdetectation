import pandas as pd

# 原子量表
atomic_weights = {
    "H": 1.008, "C": 12.011, "O": 15.999, "Na": 22.990, "Mg": 24.305,
    "Al": 26.982, "Si": 28.086, "P": 30.974, "S": 32.06, "Cl": 35.45,
    "K": 39.098, "Ca": 40.078, "Ti": 47.867, "V": 50.942, "Cr": 51.996,
    "Mn": 54.938, "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546,
    "Zn": 65.38, "Sr": 87.62, "Ba": 137.327,"Mo": 95.94, "Li": 6.941,
    "Pb": 207.2
}

# 1️⃣ 读取 Excel
df = pd.read_excel("真实浓度标签.xlsx")

# 2️⃣ 选取元素列
element_cols = [col for col in df.columns if col in atomic_weights]

# 3️⃣ wt% / atomic weight  转为 mol（并保留 2 位小数）
mol_df = df[element_cols].copy()
for elem in element_cols:
    mol_df[elem] = (df[elem] / atomic_weights[elem]).round(6)

# 4️⃣ 求总 mol
mol_sum = mol_df.sum(axis=1)

# 5️⃣ 计算 at%
at_df = (mol_df.div(mol_sum, axis=0) * 100).round(6)

# 6️⃣ 加后缀 & 输出
at_df = at_df.add_suffix("_at%")
df_out = pd.concat([df, at_df], axis=1)

df_out.to_excel("真实浓度_at%.xlsx", index=False)

print("转换完成！新文件已保存为：真实浓度_at%.xlsx")

