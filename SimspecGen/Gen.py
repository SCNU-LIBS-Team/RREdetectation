from simLIBS import SimulatedLIBS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


contents=pd.read_csv(r"D:\LIBS\RREdetectation\SimspecGen\saved\contents.csv")
#参数说明： mode有geneate和debug两种，generate模式会生成contents.csv中所有的光谱文件，debug模式会指定文件并且打开浏览器
def generate_spectra(mode,file_name):
    if mode=='generate':
        for i in range(0, len(contents)):
            elements_main=['Si', 'Al', 'Fe','Ca','K','Na','Mg','Cr','Cu','Mn','Zn','Ti','Ni','Mo','C','Li','Pb']
            elements_rareearth=['Lu','La','Y','Pr','Sm','Eu','Tb','Ho','Er','Tm','Yb']
            elements=elements_main + elements_rareearth
            file_name=contents.iloc[i, 0]
            file_name = str(contents.iloc[i, 0]).removeprefix("GBW")
            percentages_main=contents.iloc[i, 1:].tolist()
            percentages_rareearth=[0.4545,0.4545,0.4545,0.4545,0.4545,0.4545,0.4545,0.4545,0.4545,0.4545,0.4545] 
            percentages=percentages_main + percentages_rareearth  

                # Remove elements with zero percentage
            elements = [elem for elem, perc in zip(elements, percentages) if perc > 0]
            percentages = [perc for perc in percentages if perc > 0]

            if len(percentages) > 0:
                percentages[-1] = 100.0 - sum(percentages[:-1])

            libs = SimulatedLIBS(
                Te=0.86,
                Ne=10**16,
                elements=elements,
                percentages=percentages,
                resolution=3000,
                low_w=200,
                upper_w=900,
                max_ion_charge=2,
                webscraping='dynamic',
                headless=True,
                keep_browser_open=False,
                detach_browser=False
            )
            save_dir = r"D:\LIBS\RREdetectation\SimspecGen\saved"
            libs.save_to_csv(os.path.join(save_dir, file_name + "_95.csv"))
            print( f'{file_name}_95.csv',"saved successfully.")
            delay = 5  # 设置延迟时间，单位为秒

    elif mode=='debug':
        if file_name is None:
            print("请提供要调试的文件路径")
            return
        elements_main=['Si', 'Al', 'Fe','Ca','K','Na','Mg','Cr','Cu','Mn','Zn','Ti','Ni','Mo','C','Li','Pb']
        elements_rareearth=['Lu','La','Y','Pr','Sm','Eu','Tb','Ho','Er','Tm','Yb']
        elements=elements_main + elements_rareearth
        if not file_name.startswith("GBW"):
            file_name = "GBW" + file_name
        matched = contents[contents.iloc[:, 0] == file_name]
        percentages_main=matched.iloc[0, 1:].tolist()
        percentages_rareearth=[0.4545,0.4545,0.4545,0.4545,0.4545,0.4545,0.4545,0.4545,0.4545,0.4545,0.4545] 
        percentages=percentages_main + percentages_rareearth  

            # Remove elements with zero percentage
        elements = [elem for elem, perc in zip(elements, percentages) if perc > 0]
        percentages = [perc for perc in percentages if perc > 0]

        if len(percentages) > 0:
            percentages[-1] = 100.0 - sum(percentages[:-1])

        libs = SimulatedLIBS(
            Te=0.86,
            Ne=10**16,
            elements=elements,
            percentages=percentages,
            resolution=3000,
            low_w=200,
            upper_w=900,
            max_ion_charge=2,
            webscraping='dynamic',
            headless=False,
            keep_browser_open=True,
            detach_browser=True
        )
        delay = 5  # 设置延迟时间，单位为秒

generate_spectra(mode='debug',file_name="07107")