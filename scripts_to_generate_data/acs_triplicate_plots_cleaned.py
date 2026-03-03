
from custom_tools.acs_data_reader import get_acs_IOP
from custom_tools.acs_outlier_detection_functions_advanced import run_advanced_spectra_cleaning_pipeline_alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")


def subplot_helper(ax, wav, spectra_df, label, color, ls="-"):
   
    median_sp = spectra_df.median(axis=0)
    p25 = spectra_df.quantile(0.25, axis=0)
    p75 = spectra_df.quantile(0.75, axis=0)
    
    # Plot Median line
    ax.plot(wav, median_sp, color=color, lw=2, label=f"Median {label}(n={len(spectra_df)})", ls=ls)
    
    # Plot Shaded IQR area
    ax.fill_between(wav, p25, p75, color=color, alpha=0.2)
    
    ax.legend(loc='upper right', fontsize='small')

def concatenate_df(df_cumm, df):
    # Check if df_cumm is None or empty
    if df_cumm is None or df_cumm.empty:
        df_cumm = df.copy()
    else:
        df_cumm = pd.concat([df_cumm, df], ignore_index=True)
    
    return df_cumm

dir_path = "data/raw/1_acs_runs/runs"

file_list = [f for f in os.listdir(dir_path) if f.startswith('run_21_ACS.')]

metadata = pd.read_csv("data/raw/1_acs_runs/metadata_acs.csv")

def detect_sample_type(sample_name):
    return "diw" if "diw" in sample_name else "sample"

metadata["sample_type"] = metadata["sample_name"].apply(detect_sample_type)

def convert_run_list_to_string(run_list):
    for (i, r) in enumerate(run_list):
        if r<10:
            run_list[i] = f"00{r}"
        elif r<100:
            run_list[i] = f"0{r}"
        else:
            run_list[i] = str(r)
    # Return as a list
    return run_list

def get_runs_from_samplename(samplename, metadata):
    # Filter rows where sample_name matches, then grab the run_nr column
    runs = metadata.loc[metadata["sample_name"] == samplename, "run_nr"].tolist()
    runs = convert_run_list_to_string(runs)
    return runs

samplename_list = metadata["sample_name"].unique()

outlier_A = [
    102, 346, 406, 412, 436, 
    453, 455, 463, 481, 487
]

dubious_A = [
    413, 472, 7, 58, 360,
    361, 397, 418, 441, 442, 
    448, 465, 473, 474 
]

outlier_S = [
    7, 13, 38, 47, 50, 
    64, 70, 73, 82, 83, 
    84, 394, 397, 400, 406, 
    410, 411, 436, 437, 453, 
    455, 463, 466
]

dubious_S = [
    61, 80, 81, 26, 29, 
    43, 44, 67, 71, 72, 
    75, 79, 105, 401, 413, 
    414, 432, 441, 442, 443
]

outlier_A = convert_run_list_to_string(outlier_A)
outlier_S = convert_run_list_to_string(outlier_S)
dubious_A = convert_run_list_to_string(dubious_A)
dubious_S = convert_run_list_to_string(dubious_S)

color_list = ["blue", "red", "green"]

def main():
    for samplename in samplename_list:
        if "diw" not in samplename:
            run_list = get_runs_from_samplename(samplename=samplename, metadata=metadata)

            fig, ax = plt.subplots(ncols=2)

            df_arr_A_cumm = pd.DataFrame() 
            df_arr_S_cumm = pd.DataFrame() 

            for i, run in enumerate(run_list):
                if run in outlier_A or run in outlier_S: ### WRONG LOGIC HERE !!!
                    continue
                
                if run in dubious_A or run in dubious_S:
                    ls = "--"
                else:
                    ls="-"
                try:
                    f = f"run_21_ACS.{run}"
                    df_arr_A, df_arr_S = get_acs_IOP(os.path.join(dir_path, f))
                    wav_A = np.array(df_arr_A.columns.astype(float))
                    wav_S = np.array(df_arr_S.columns.astype(float))
                    
                    df_arr_A_cumm = concatenate_df(df_arr_A_cumm, df_arr_A)
                    df_arr_S_cumm = concatenate_df(df_arr_S_cumm, df_arr_S)
                    
                    df_arr_A_clean, _ = run_advanced_spectra_cleaning_pipeline_alt(df_arr_A, plot=False)
                    n_valid_spectra = df_arr_A_clean.shape[0]
                    
                    if n_valid_spectra > 0:
                        subplot_helper(
                            ax=ax[0], wav = wav_A, spectra_df=df_arr_A_clean, label=str(run), color=color_list[i], ls=ls)
                        
                    df_arr_S_clean, _ = run_advanced_spectra_cleaning_pipeline_alt(df_arr_S, plot=False)
                    n_valid_spectra = df_arr_S_clean.shape[0]
                    
                    if n_valid_spectra > 0:
                        subplot_helper(
                            ax=ax[1], wav = wav_S, spectra_df=df_arr_S_clean, label=str(run), color=color_list[i], ls=ls)
                except Exception as e:
                    print(f"FAILED FOR RUN: {run}")
                    print(f"ERROR TYPE: {type(e).__name__}")
                    print(f"ERROR MESSAGE: {e}")
            try: 
                df_arr_A_cumm_clean, _ = run_advanced_spectra_cleaning_pipeline_alt(df_arr_A_cumm, plot=False)
                n_valid_spectra = df_arr_A_cumm_clean.shape[0]

                if n_valid_spectra > 0:
                    subplot_helper(
                            ax=ax[0], wav = wav_A, spectra_df=df_arr_A_cumm_clean, label=str(run), color="black")
            except: 
                pass
            
            try:     
                df_arr_S_cumm_clean, _ = run_advanced_spectra_cleaning_pipeline_alt(df_arr_S_cumm, plot=False)
                n_valid_spectra = df_arr_S_cumm_clean.shape[0]

                if n_valid_spectra > 0:
                    subplot_helper(
                            ax=ax[1], wav = wav_S, spectra_df=df_arr_S_cumm_clean, label=str(run), color="black")

            except: 
                pass

            ax[0].legend()
            ax[1].legend()

            plt.title(samplename)
            # Define your directory path
            output_dir = Path("data/plots/2026-02-26_acs_triplicate_plots_alt_cleaned2")

            # Create the directory (and any missing parent folders)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Now save your plot
            plt.savefig(output_dir / f"{samplename}.png")
            plt.close()

if __name__ == "__main__":
    main()