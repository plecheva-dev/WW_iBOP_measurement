# %%
from custom_tools.acs_data_reader import get_acs_IOP
from custom_tools.acs_outlier_detection_functions import run_spectra_cleaning_pipeline
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

dir_path = "data/raw/1_acs_runs/runs"

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
            clean_A, _ = run_spectra_cleaning_pipeline(df_arr_A, wav_A, threshold_575=0.05, plot=False)
            mA, sA = np.mean(clean_A, axis=0), np.std(clean_A, axis=0)
            saved_data_a.append(
                {
                    "sample": samplename,
                    "n_valid_spectra": len(clean_A),
                    "mean_val": np.mean(mA), 
                    "mean_std": np.mean(sA)
                }
            )
            
            
            clean_S, _ = run_spectra_cleaning_pipeline(df_arr_S, wav_S, threshold_575=0.05, plot=False)
            mS, sS = np.mean(clean_S, axis=0), np.std(clean_S, axis=0)
            saved_data_s.append(
                {
                    "sample": samplename,
                    "n_valid_spectra": len(clean_S),
                    "mean_val": np.mean(mS), 
                    "mean_std": np.mean(sS),
                }
            )
        
    except:
        print(f"Failed for: {f}")

df_outlier_report_A = pd.DataFrame(saved_data_a)
df_outlier_report_A["relative_std"] = 100 * df_outlier_report_A["mean_std"] / df_outlier_report_A["mean_val"]
df_outlier_report_S = pd.DataFrame(saved_data_s)
df_outlier_report_S["relative_std"] = 100 * df_outlier_report_S["mean_std"] / df_outlier_report_S["mean_val"]




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
sns.scatterplot(data=df_outlier_report_A, x="n_valid_spectra", y="relative_std", 
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
sns.scatterplot(data=df_outlier_report_S, x="n_valid_spectra", y="relative_std", 
                ax=ax[1, 1], color='tab:orange', s=60)
ax[1, 1].set_title(r"Scattering: $\sigma_{rel}$ vs. $N_{valid}$", fontsize=14)
ax[1, 1].set_xlabel(r"Number of Valid Spectra ($N_{valid}$)")
ax[1, 1].set_ylabel(r"Relative Std (%)")

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the plot
plt.show()


### Next step: for runs with relative std >5% and/or n_valid_spectra<20 => make plots and visual inspect. Add relative std to plot subtitle