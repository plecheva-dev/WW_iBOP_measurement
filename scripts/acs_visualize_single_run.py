
from custom_tools.acs_data_reader import get_acs_IOP
from custom_tools.acs_outlier_detection_functions import run_spectra_cleaning_pipeline

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

dir_path = "data/raw/1_acs_runs/runs"

file_list = [ff for ff in os.listdir(dir_path) if ff.startswith('run_21_ACS.')]
f= file_list[0]

metadata = pd.read_csv("data/raw/1_acs_runs/metadata_acs.csv")

if __name__ == "__main__":
           
    df_arr_A, df_arr_C = get_acs_IOP(os.path.join(dir_path, f))
    wav_A = np.array(df_arr_A.columns[1:].astype(float))
    wav_C = np.array(df_arr_C.columns[1:].astype(float))

    samplename = metadata.loc[metadata["run_nr"]==int(f[-3:]), "sample_name"]
    
    # 1. Generate the figures (they are stored in Matplotlib's internal memory)
    _, fig_A = run_spectra_cleaning_pipeline(df_arr_A, wav_A, threshold_575=0.05)
    _, fig_C = run_spectra_cleaning_pipeline(df_arr_C, wav_C, threshold_575=0.05)
    
    # 2. Add a unique title to the window (optional but helpful)
    fig_A.canvas.manager.set_window_title(f"{samplename} - A")
    fig_C.canvas.manager.set_window_title(f"{samplename} - S")
    
    # 3. Open the interactive windows
    # This command is "blocking", meaning the script stays alive 
    # as long as the windows are open.
    plt.show()
