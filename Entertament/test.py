
import numpy as np
import pandas as pd
import glob
import os
import pywt
import matplotlib.pyplot as plt
from collections import defaultdict
from Wavelet_peakfinding import find_peaks_ridge,peak_correction,wavelet_peak_detection #寻峰
from Elements_Combfact import elements_database #元素库制作
from scipy.optimize import linear_sum_assignment


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
                best=max(filterd_pairs, key=lambda x: x[1])
                final_T[base_elem]= best[0]
                final_R2[base_elem]= best[1]
                final_lc[base_elem]= best[2]
                final_distance[base_elem]= best[3] 
            else:
                final_T[base_elem]= 0
                final_R2[base_elem]= 0
                final_lc[base_elem]= 0
                final_distance[base_elem]= 0
    
    for elem, distances in final_distance.items():
        if distances<10000:
            elements_confidence[elem]=np.exp(-1.5*distances/final_R2[elem]) #指数映射
        else:
            elements_confidence[elem]=0