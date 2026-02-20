from custom_tools.acs_data_reader import get_acs_IOP
from custom_tools.acs_outlier_detection_functions import get_outlier_max_mask
from custom_tools.acs_outlier_detection_functions import get_outlier_deriv575nm_mask
from custom_tools.acs_outlier_detection_functions import get_outlier_iterative_mask

import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")



def run_spectra_cleaning_pipeline(df, wav, threshold_575=0.001, iterative_threshold=10.0):
    # Prepare the figure
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 15), sharex=True)
    
    # Extract just the numeric data (assuming column 0 is an index/label)
    spectra_values = df.values[:, 1:]
    
    # ---------------------------------------------------------
    # i) Plot All Spectra, Mean, and STD
    # ---------------------------------------------------------
    axes[0, 0].plot(wav, spectra_values.T, color="gray", alpha=0.1, lw=0.5)
    m0, s0 = np.mean(spectra_values, axis=0), np.std(spectra_values, axis=0)
    axes[0, 0].plot(wav, m0, color="red", lw=2, label="Mean")
    axes[0, 0].fill_between(wav, m0-s0, m0+s0, color="red", alpha=0.2)
    axes[0, 0].set_title(f"Step 1: All Original Spectra (n={len(df)})")

    # ---------------------------------------------------------
    # ii) Calculate MAX outliers and plot them
    # ---------------------------------------------------------
    max_outlier_mask = get_outlier_max_mask(df)
    if max_outlier_mask.any():
        axes[1, 0].plot(wav, df[max_outlier_mask].values[:, 1:].T, color="black", alpha=0.6)
    axes[1, 0].set_title(f"Step 2: MAX Outliers Detected (n={max_outlier_mask.sum()})")

    # ---------------------------------------------------------
    # iii) Deduct MAX, then calculate and plot deriv575nm outliers
    # ---------------------------------------------------------
    df_after_max = df[~max_outlier_mask]
    # We apply the 575 check ONLY to those that passed the MAX check
    deriv_outlier_mask = get_outlier_deriv575nm_mask(df_after_max, threshold=threshold_575)
    
    if deriv_outlier_mask.any():
        axes[2, 0].plot(wav, df_after_max[deriv_outlier_mask].values[:, 1:].T, color="purple", alpha=0.6)
    axes[2, 0].set_title(f"Step 3: 575nm Deriv Outliers (n={deriv_outlier_mask.sum()})")

    # ---------------------------------------------------------
    # iv) Plot Remaining Spectra (after MAX and 575)
    # ---------------------------------------------------------
    df_remaining = df_after_max[~deriv_outlier_mask]
    rem_values = df_remaining.values[:, 1:]
    
    if rem_values.any():
        axes[0, 1].plot(wav, rem_values.T, color="gray", alpha=0.2, lw=0.5)
        m3, s3 = np.mean(rem_values, axis=0), np.std(rem_values, axis=0)
        axes[0, 1].plot(wav, m3, color="blue", lw=2)
        axes[0, 1].fill_between(wav, m3-s3, m3+s3, color="blue", alpha=0.2)
        axes[0, 1].set_title(f"Step 4: Remaining Spectra after Hard Filters (n={len(df_remaining)})")

        # ---------------------------------------------------------
        # v) Calculate Iterative Outliers on remaining
        # ---------------------------------------------------------
        # Note: get_outlier_iterative_mask returns a 'valid' mask (ones for keep)
        iterative_valid_mask = get_outlier_iterative_mask(rem_values, threshold=iterative_threshold)
        
        # Plot the ones that got kicked out by iterative cleaning
        iter_outliers = rem_values[~iterative_valid_mask]
        if len(iter_outliers) > 0:
            axes[1, 1].plot(wav, iter_outliers.T, color="orange", alpha=0.6)
        axes[1, 1].set_title(f"Step 5: Iterative Outliers Removed (n={len(iter_outliers)})")

        # ---------------------------------------------------------
        # vi) Plot final Cleaned Spectra
        # ---------------------------------------------------------
        clean_values = rem_values[iterative_valid_mask]
        axes[2, 1].plot(wav, clean_values.T, color="black", alpha=0.1, lw=0.5)
        m5, s5 = np.mean(clean_values, axis=0), np.std(clean_values, axis=0)
        axes[2, 1].plot(wav, m5, color="green", lw=2, label="Final Mean")
        axes[2, 1].fill_between(wav, m5-s5, m5+s5, color="green", alpha=0.2)
        axes[2, 1].set_title(f"Step 6: Final Cleaned Spectra (n={len(clean_values)})")
        axes[2, 1].set_xlabel("Wavelength (nm)")

    return fig


dir_path = "data/raw/1_acs_runs/runs"

file_list = [ff for ff in os.listdir(dir_path) if ff.startswith('run_21_ACS.')]

f = "run_21_ACS.480"

df_arr_A, df_arr_C = get_acs_IOP(os.path.join(dir_path, f))
wav_A = np.array(df_arr_A.columns[1:].astype(float))
wav_C = np.array(df_arr_C.columns[1:].astype(float))

if __name__ == "__main__":
    # 1. Generate the figures (they are stored in Matplotlib's internal memory)
    fig_A = run_spectra_cleaning_pipeline(df_arr_A, wav_A)
    fig_C = run_spectra_cleaning_pipeline(df_arr_C, wav_C)
    
    # 2. Add a unique title to the window (optional but helpful)
    fig_A.canvas.manager.set_window_title('Dataset A - Cleaning Pipeline')
    fig_C.canvas.manager.set_window_title('Dataset C - Cleaning Pipeline')
    
    # 3. Open the interactive windows
    # This command is "blocking", meaning the script stays alive 
    # as long as the windows are open.
    plt.show()
