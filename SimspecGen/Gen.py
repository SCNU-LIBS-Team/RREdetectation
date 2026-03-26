from simLIBS import SimulatedLIBS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import traceback


contents=pd.read_csv(r"D:\LIBS\RREdetectation\SimspecGen\saved\contents.csv")
FLAG_COLUMN = "generated_flag"

if FLAG_COLUMN not in contents.columns:
    contents[FLAG_COLUMN] = ""


def get_pending_count():
    return int((contents[FLAG_COLUMN].astype(str).str.strip().str.upper() != 'G').sum())

#参数说明： mode有geneate和debug两种，generate模式会生成contents.csv中所有的光谱文件，debug模式会指定文件并且打开浏览器
def generate_spectra(mode,file_name,resolution=3000,Te=0.5,Ne=1e17,low_w=200,upper_w=900):
    if mode=='generate':
        pending_indices = [i for i in range(len(contents)) if str(contents.at[i, FLAG_COLUMN]).strip().upper() != 'G']
        total_pending = len(pending_indices)
        print(f"待生成光谱数量: {total_pending}")
        print(f"分辨率为：{resolution}, Te={Te} eV, Ne={Ne} cm^-3, 波长范围: {low_w}-{upper_w} nm")

        if total_pending == 0:
            print("所有光谱都已生成，无需处理。")
            return

        completed_now = 0
        for i in pending_indices:

            elements_main=['Si', 'Al', 'Fe','Ca','K','Na','Mg','Cr','Cu','Mn','Zn','Ti','Ni','Mo','C','Li','Pb']
            elements_rareearth=['Lu','La','Y','Pr','Sm','Eu','Tb','Ho','Er','Tm','Yb']
            elements=elements_main + elements_rareearth
            file_name=contents.iloc[i, 0]
            file_name = str(contents.iloc[i, 0]).removeprefix("GBW")
            main_columns = [f"{element}_at%" for element in elements_main]
            percentages_main = [contents.at[i, column] for column in main_columns]
            percentages_rareearth=[0.4545,0.4545,0.4545,0.4545,0.4545,0.4545,0.4545,0.4545,0.4545,0.4545,0.4545] 
            percentages=percentages_main + percentages_rareearth  

                # Remove elements with zero percentage
            elements = [elem for elem, perc in zip(elements, percentages) if perc > 0]
            percentages = [perc for perc in percentages if perc > 0]

            if len(percentages) > 0:
                percentages[-1] = 100.0 - sum(percentages[:-1])

            libs = SimulatedLIBS(
                Te=Te,
                Ne=Ne,
                elements=elements,
                percentages=percentages,
                resolution=resolution,
                low_w=low_w,
                upper_w=upper_w,
                max_ion_charge=2,
                webscraping='dynamic',
                headless=True,
                keep_browser_open=False,
                detach_browser=False
            )
            save_dir = r"D:\LIBS\RREdetectation\SimspecGen\saved"
            libs.save_to_csv(os.path.join(save_dir, file_name + "_95.csv"))
            contents.at[i, FLAG_COLUMN] = 'G'
            contents.to_csv(r"D:\LIBS\RREdetectation\SimspecGen\saved\contents.csv", index=False)
            completed_now += 1
            print( f'{file_name}_95.csv',"saved successfully.")
            print(f"当前进度: {completed_now}/{total_pending}")
            delay = 5  # 设置延迟时间，单位为秒

        print(f"本次生成完成，共处理: {completed_now}/{total_pending}")

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
        main_columns = [f"{element}_at%" for element in elements_main]
        percentages_main = matched.loc[matched.index[0], main_columns].tolist()
        percentages_rareearth=[0.4545,0.4545,0.4545,0.4545,0.4545,0.4545,0.4545,0.4545,0.4545,0.4545,0.4545] 
        percentages=percentages_main + percentages_rareearth  

            # Remove elements with zero percentage
        elements = [elem for elem, perc in zip(elements, percentages) if perc > 0]
        percentages = [perc for perc in percentages if perc > 0]

        if len(percentages) > 0:
            percentages[-1] = 100.0 - sum(percentages[:-1])

        libs = SimulatedLIBS(
            Te=1.2,
            Ne=10**17,
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

def run_generate_with_auto_restart(retry_wait_seconds=10):
    while True:
        pending_before = get_pending_count()
        if pending_before == 0:
            print("所有任务已完成，程序结束。")
            break

        print(f"检测到剩余任务 {pending_before} 条，开始运行生成任务。")
        try:
            generate_spectra(mode='generate',file_name="03116",resolution=3000,Te=1.5,Ne=1e17,low_w=200,upper_w=900)
            pending_after = get_pending_count()
            if pending_after == 0:
                print("本轮运行后所有任务已完成。")
                break
            print(f"本轮结束后仍有 {pending_after} 条未完成，将继续下一轮。")
        except KeyboardInterrupt:
            print("检测到手动中断，程序退出。")
            break
        except Exception as error:
            print(f"检测到程序中断: {error}")
            traceback.print_exc()
            print(f"{retry_wait_seconds} 秒后自动重试...")
            time.sleep(retry_wait_seconds)


run_generate_with_auto_restart(retry_wait_seconds=10)
