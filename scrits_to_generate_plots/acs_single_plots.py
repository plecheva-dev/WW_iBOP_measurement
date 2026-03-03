from custom_tools.acs_data_reader import get_acs_IOP
from custom_tools.acs_outlier_detection_functions import run_spectra_cleaning_pipeline
from pathlib import Path
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

dir_path = "data/raw/1_acs_runs/runs"

file_list = [ff for ff in os.listdir(dir_path) if ff.startswith('run_21_ACS.')]

metadata = pd.read_csv("data/raw/1_acs_runs/metadata_acs.csv")

def main():
    for f in file_list:
        try:
            df_arr_A, df_arr_C = get_acs_IOP(os.path.join(dir_path, f))
            wav_A = np.array(df_arr_A.columns[1:].astype(float))
            wav_C = np.array(df_arr_C.columns[1:].astype(float))

            samplename = metadata.loc[metadata["run_nr"]==int(f[-3:]), "sample_name"].values[0]
            
            
            
            # 1. Generate the figures (they are stored in Matplotlib's internal memory)
            _, fig_A = run_spectra_cleaning_pipeline(df_arr_A, wav_A, threshold_575=0.05)
            _, fig_C = run_spectra_cleaning_pipeline(df_arr_C, wav_C, threshold_575=0.05)

            if "diw" in samplename:    
            # Define your directory path
                output_dir = Path("data/plots/acs_single_plots/diw")
            else:
                output_dir = Path("data/plots/acs_single_plots/samples")

            # Create the directory (and any missing parent folders)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Now save your plot
            fig_A.savefig(output_dir / f"{f}_{samplename}_A.png")
            fig_C.savefig(output_dir / f"{f}_{samplename}_S.png")
            
            plt.close('all')
            
        except:
            print(f"Failed for: {f}")

if __name__ == "__main__":
    main()


