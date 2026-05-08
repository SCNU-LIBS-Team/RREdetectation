import numpy as np
from sif_parser import np_open


file_path=r'D:\LIBS\RREdetectation\Broaden_research\Sif\1.sif'
data, info = np_open(file_path)
print(info.keys())
for key, value in info.items():
    print(key, ":", value)
