from custom_tools.acs_data_reader import get_acs_IOP
from custom_tools.acs_outlier_detection_functions_advanced import run_advanced_spectra_cleaning_pipeline_alt
import numpy as np
import pandas as pd
from pathlib import Path
import os
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def analyze_clean_spectra(clean_sp, samplename):
    mean_sp, std_sp = np.mean(clean_sp, axis=0), np.std(clean_sp, axis=0)
    n_valid_sp = len(clean_sp)
    mean_mean = np.mean(mean_sp)
    mean_std = np.mean(std_sp)
    
    analysis_dict = {
            "sample": samplename,
            "n_valid_spectra": n_valid_sp,
            "mean_mean": mean_mean, 
            "mean_std": mean_std,
            "rel_std": 100*mean_std/mean_mean
        }
    
    visual_inspection_needed = _test_if_visual_inspection_is_needed(analysis_dict)
    
    analysis_dict["visual_inspection_needed"] = visual_inspection_needed
    
    return analysis_dict

def _test_if_visual_inspection_is_needed(analysis_dict):
    n = analysis_dict["n_valid_spectra"]
    rel_std = analysis_dict["rel_std"]
    mean_std = analysis_dict["mean_std"] # Ensure this matches your dict key

    # Condition 1: Sample size is too small
    if n < 20:
        return True
    
    # Condition 2: High relative noise AND significant absolute noise
    if n >= 20:
        if rel_std > 5 and mean_std > 1:
            return True
    
    # Otherwise, it's considered a valid/clean sample
    return False

def savefig_in_correct_folder(output_path, fig, samplename, f, analysis_dict, sampletype):
    visual_inspection_needed = analysis_dict["visual_inspection_needed"]
    
    if visual_inspection_needed:
        output_dir = Path(f"{output_path}/visual inspection needed")
    else: 
        output_dir = Path(f"{output_path}/valid")

    # Create the directory (and any missing parent folders)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Now save your plot
    fig.savefig(output_dir / f"{f}_{samplename}_{sampletype}_relSTD={analysis_dict['rel_std']:.1f}_STD={analysis_dict['mean_std']:.1f}.png")





dir_path = "data/raw/1_acs_runs/runs"
output_path = "data/plots/2026-02-26_advanced_outlier_detection"
file_list = [ff for ff in os.listdir(dir_path) if ff.startswith('run_21_ACS.')]

metadata = pd.read_csv("data/raw/1_acs_runs/metadata_acs.csv")

saved_data_a = []
saved_data_s = []

for f in file_list:
    try:
        
        samplename = metadata.loc[metadata["run_nr"]==int(f[-3:]), "sample_name"].values[0]
        
        if "diw" not in samplename:
        
            df_arr_A, df_arr_S = get_acs_IOP(os.path.join(dir_path, f))
            wav_A = np.array(df_arr_A.columns[1:].astype(float))
            wav_S = np.array(df_arr_S.columns[1:].astype(float))
           
            # 1. Generate the figures (they are stored in Matplotlib's internal memory)
            clean_A, fig_A = run_advanced_spectra_cleaning_pipeline_alt(df_arr_A, plot=True)
            analyzis_clean_A = analyze_clean_spectra(clean_A, samplename)
            saved_data_a.append(analyzis_clean_A)
            
            savefig_in_correct_folder(output_path, fig_A, samplename, f, analyzis_clean_A, sampletype="A")
                    
            clean_S, fig_S = run_advanced_spectra_cleaning_pipeline_alt(df_arr_S, plot=True)
            analyzis_clean_S = analyze_clean_spectra(clean_S, samplename)
            saved_data_s.append(analyzis_clean_S)

            savefig_in_correct_folder(output_path, fig_S, samplename, f, analyzis_clean_S, sampletype="S")
            
            plt.close('all')
        
    except Exception as e:
        print(f"FAILED FOR FILE: {f}")
        print(f"ERROR TYPE: {type(e).__name__}")
        print(f"ERROR MESSAGE: {e}")

df_outlier_report_A = pd.DataFrame(saved_data_a)
df_outlier_report_S = pd.DataFrame(saved_data_s)


# Set aesthetic style
sns.set_theme(style="whitegrid")

# Create the figure and axes
fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12, 10))

# --- Row 1: Absorption Data (df_outlier_report_A) ---

# Left: Mean Std vs N Valid Spectra
sns.scatterplot(data=df_outlier_report_A, x="n_valid_spectra", y="mean_std", 
                ax=ax[0, 0], color='tab:blue', s=60)
ax[0, 0].set_title(r"Absorption: $\bar{\sigma}$ vs. $N_{valid}$", fontsize=14)
ax[0, 0].set_xlabel(r"Number of Valid Spectra ($N_{valid}$)")
ax[0, 0].set_ylabel(r"Mean Standard Deviation ($m^{-1}$)")

# Right: Relative Std vs N Valid Spectra
sns.scatterplot(data=df_outlier_report_A, x="n_valid_spectra", y="rel_std", 
                ax=ax[0, 1], color='tab:blue', s=60)
ax[0, 1].set_title(r"Absorption: $\sigma_{rel}$ vs. $N_{valid}$", fontsize=14)
ax[0, 1].set_xlabel(r"Number of Valid Spectra ($N_{valid}$)")
ax[0, 1].set_ylabel(r"Relative Std (%)")


# --- Row 2: Scattering Data (df_outlier_report_S) ---

# Left: Mean Std vs N Valid Spectra
sns.scatterplot(data=df_outlier_report_S, x="n_valid_spectra", y="mean_std", 
                ax=ax[1, 0], color='tab:orange', s=60)
ax[1, 0].set_title(r"Scattering: $\bar{\sigma}$ vs. $N_{valid}$", fontsize=14)
ax[1, 0].set_xlabel(r"Number of Valid Spectra ($N_{valid}$)")
ax[1, 0].set_ylabel(r"Mean Standard Deviation ($m^{-1}$)")

# Right: Relative Std vs N Valid Spectra
sns.scatterplot(data=df_outlier_report_S, x="n_valid_spectra", y="rel_std", 
                ax=ax[1, 1], color='tab:orange', s=60)
ax[1, 1].set_title(r"Scattering: $\sigma_{rel}$ vs. $N_{valid}$", fontsize=14)
ax[1, 1].set_xlabel(r"Number of Valid Spectra ($N_{valid}$)")
ax[1, 1].set_ylabel(r"Relative Std (%)")

# Adjust layout to prevent overlap
plt.tight_layout()
fig.savefig(f"{output_path}/spectra_mapping.png")
df_outlier_report_A.to_csv(f"{output_path}/outlier_report_a.csv")
df_outlier_report_S.to_csv(f"{output_path}/outlier_report_s.csv")