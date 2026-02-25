
from custom_tools.acs_data_reader import get_acs_IOP
from custom_tools.acs_outlier_detection_functions import run_spectra_cleaning_pipeline

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

def plot_median_and_std(df_arr, wav, label, ax):
    clean_values = df_arr.values[:, 1:]
    mean_arr, std_arr = np.mean(clean_values, axis=0), np.std(clean_values, axis=0)
    ax.plot(wav, mean_arr, lw=2, label=label)
    ax.fill_between(wav, mean_arr-std_arr, mean_arr+std_arr, alpha=0.2)

dir_path = "data/raw/1_acs_runs/runs"

file_list = [f for f in os.listdir(dir_path) if f.startswith('run_21_ACS.')]

metadata = pd.read_csv("data/raw/1_acs_runs/metadata_acs.csv")

def detect_sample_type(sample_name):
    return "diw" if "diw" in sample_name else "sample"

metadata["sample_type"] = metadata["sample_name"].apply(detect_sample_type)

def get_runs_from_samplename(samplename, metadata):
    # Filter rows where sample_name matches, then grab the run_nr column
    runs = metadata.loc[metadata["sample_name"] == samplename, "run_nr"].tolist()
    
    for (i, r) in enumerate(runs):
        if r<10:
            runs[i] = f"00{r}"
        elif r<100:
            runs[i] = f"0{r}"
        else:
            runs[i] = str(r)
    # Return as a list
    return runs

samplename_list = metadata["sample_name"].unique()

def main():
    for samplename in samplename_list:

        run_list = get_runs_from_samplename(samplename=samplename, metadata=metadata)

        fig, ax = plt.subplots(ncols=2)

        df_arr_A_cumm = pd.DataFrame() 
        df_arr_C_cumm = pd.DataFrame() 

        for run in run_list:
            f = f"run_21_ACS.{run}"
            df_arr_A, df_arr_C = get_acs_IOP(os.path.join(dir_path, f))
            wav_A = np.array(df_arr_A.columns[1:].astype(float))
            wav_C = np.array(df_arr_C.columns[1:].astype(float))
            df_arr_A_cumm = pd.concat([df_arr_A_cumm, df_arr_A])
            df_arr_C_cumm = pd.concat([df_arr_C_cumm, df_arr_C])
            
            df_arr_A_clean, _ = run_spectra_cleaning_pipeline(df_arr_A, wav_A, plot=False)
            n_valid_spectra = len(df_arr_A_clean)
            
            if n_valid_spectra > 0:
                plot_median_and_std(
                    df_arr=df_arr_A_clean, 
                    wav=wav_A,
                    label=f"{run} ({n_valid_spectra} spectra)",
                    ax=ax[0]
                    )
                
            df_arr_C_clean, _ = run_spectra_cleaning_pipeline(df_arr_C, wav_C, plot=False)
            n_valid_spectra = len(df_arr_C_clean)
            
            if n_valid_spectra > 0:
                plot_median_and_std(
                    df_arr=df_arr_C_clean, 
                    wav=wav_C,
                    label=f"{run} ({n_valid_spectra} spectra)",
                    ax=ax[1]
                    )


        df_arr_A_cumm_clean, _ = run_spectra_cleaning_pipeline(df_arr_A_cumm, wav_A, plot=False)
        n_valid_spectra = len(df_arr_A_cumm_clean)

        if n_valid_spectra > 0:
            plot_median_and_std(
                    df_arr=df_arr_A_cumm_clean, 
                    wav=wav_A,
                    label=f"cumm ({n_valid_spectra} spectra)",
                    ax=ax[0]
                    )
            
        df_arr_C_cumm_clean, _ = run_spectra_cleaning_pipeline(df_arr_C_cumm, wav_C, plot=False)
        n_valid_spectra = len(df_arr_C_cumm_clean)

        if n_valid_spectra > 0:
            plot_median_and_std(
                    df_arr=df_arr_C_cumm_clean, 
                    wav=wav_C,
                    label=f"cumm ({n_valid_spectra} spectra)",
                    ax=ax[1]
                    )

        if not "diw" in samplename:
            ax[0].legend()
            ax[1].legend()

        plt.title(samplename)
        # Define your directory path
        output_dir = Path("data/plots/acs_triplicate_plots_cleaned")

        # Create the directory (and any missing parent folders)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Now save your plot
        plt.savefig(output_dir / f"{samplename}.png")
        plt.close()

if __name__ == "__main__":
    main()