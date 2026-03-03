
import numpy as np
import matplotlib.pyplot as plt
from .acs_outlier_detection_functions.iterative import apply_iterative_outlier_detection
from .acs_outlier_detection_functions.iterative import apply_asymmetric_outlier_detection
from .acs_outlier_detection_functions.minmax import apply_minmax_outlier_detection_and_correction
from .acs_outlier_detection_functions.slope_576nm import apply_slope576_outlier_detection_and_correction



def subplot_helper(ax, wav, spectra_df, label, display_stats=False):
    # Plot individual spectra
    for idx, sp in spectra_df.iterrows():
        ax.plot(wav, sp, color="gray", alpha=0.3, lw=0.5)
    
    if display_stats:
        # Calculate Median and IQR (25th and 75th percentiles)
        # Using numeric_only=True if your DF has non-numeric columns
        median_sp = spectra_df.median(axis=0)
        p25 = spectra_df.quantile(0.25, axis=0)
        p75 = spectra_df.quantile(0.75, axis=0)
        
        # Plot Median line
        ax.plot(wav, median_sp, color="blue", lw=2, label="Median")
        
        # Plot Shaded IQR area
        ax.fill_between(wav, p25, p75, color="blue", alpha=0.2, label="IQR (25-75%)")
        
        ax.legend(loc='upper right', fontsize='small')
    
    n_sp = spectra_df.shape[0]
    ax.set_title(f"{label} (n= {n_sp})")


def run_advanced_spectra_cleaning_pipeline(spectra_df, plot=True):
    
    minmax_outlier_index, valid_index, updated_df = apply_minmax_outlier_detection_and_correction(spectra_df)
    
    remaining_spectra_after_minmax_df = updated_df.loc[valid_index]        
    slope576_outlier_index, valid_index, updated_df = apply_slope576_outlier_detection_and_correction(remaining_spectra_after_minmax_df)       
    
    remaining_spectra_after_slope576_df = updated_df.loc[valid_index]
    iterative_outlier_index, valid_index = apply_iterative_outlier_detection(remaining_spectra_after_slope576_df)
    
    valid_spectra_df = updated_df.loc[valid_index]
    
    fig = None
    
    if plot: 
        fig, ax = plt.subplots(nrows=3, ncols=2, figsize = (12, 9), sharex=True)
        wav = np.array(spectra_df.columns.astype("float"))

        subplot_helper(ax=ax[0, 0], wav=wav, spectra_df=spectra_df, label="All spectra", display_stats=True)
        subplot_helper(ax=ax[1, 0], wav=wav, spectra_df=spectra_df.loc[minmax_outlier_index], label="MinMAx outlier")
        subplot_helper(ax=ax[2, 0], wav=wav, spectra_df=remaining_spectra_after_minmax_df.loc[slope576_outlier_index], label="slope576 outlier")
        subplot_helper(ax=ax[0, 1], wav=wav, spectra_df=remaining_spectra_after_slope576_df, label="Remaining")
        subplot_helper(ax=ax[1, 1], wav=wav, spectra_df=remaining_spectra_after_slope576_df.loc[iterative_outlier_index], label="Iterative outliers")
        subplot_helper(ax=ax[2, 1], wav=wav, spectra_df=valid_spectra_df, label="Valid spectra", display_stats=True)

        plt.tight_layout()

    return minmax_outlier_index, slope576_outlier_index, iterative_outlier_index, valid_index, valid_spectra_df, fig


def run_advanced_spectra_cleaning_pipeline_alt(spectra_df, plot=True):
    
    spectra_df = spectra_df.reset_index(drop=True)

    minmax_outlier_index, minmax_valid_index, updated_df = apply_minmax_outlier_detection_and_correction(spectra_df)
    
    remaining_spectra_after_minmax_df = updated_df.loc[minmax_valid_index]
    
    wav = np.array(spectra_df.columns.astype("float"))
    mask = wav < 576
    
    iterative_outlier_index_1, iter1_valid_index = apply_asymmetric_outlier_detection(remaining_spectra_after_minmax_df.loc[:, mask], threshold_iter=5, mad_floor_perc=0.2)
    
    remaining_spectra_after_first_iterative_df = updated_df.loc[iter1_valid_index]
              
    slope576_outlier_index, valid_index, updated_df = apply_slope576_outlier_detection_and_correction(remaining_spectra_after_first_iterative_df)       
    
    remaining_spectra_after_slope576_df = updated_df.loc[valid_index]
    
    iterative_outlier_index_2, valid_index = apply_asymmetric_outlier_detection(remaining_spectra_after_slope576_df, threshold_iter=2)
    
    valid_spectra_df = updated_df.loc[valid_index]
    
    fig = None
    
    if plot: 
        fig, ax = plt.subplots(nrows=4, ncols=3, figsize = (15, 10), sharex=True)
        
        all_spectra = spectra_df
        minmax_outliers = spectra_df.loc[minmax_outlier_index]
        remaining_after_minmax = remaining_spectra_after_minmax_df
        iter1_outlier = remaining_spectra_after_minmax_df.loc[iterative_outlier_index_1]
        remaining_after_iter1 = remaining_spectra_after_first_iterative_df
        slope576_outlier = remaining_spectra_after_first_iterative_df.loc[slope576_outlier_index]
        remaining_after_slope = remaining_spectra_after_slope576_df
        iter2_outlier = remaining_spectra_after_slope576_df.loc[iterative_outlier_index_2]
        
        subplot_helper(
            ax=ax[0, 0], wav=wav, spectra_df=all_spectra, label="All spectra", display_stats=True)
        ax[0, 0].plot(wav, valid_spectra_df.median(axis=0), ls="--")
        subplot_helper(
            ax=ax[1, 0], wav=wav, spectra_df=minmax_outliers, label="MinMAx outlier")
        
        subplot_helper(
            ax=ax[2, 0], wav=wav, spectra_df=remaining_after_minmax, label="Remaining after Minmax", display_stats=True)
        subplot_helper(
            ax=ax[3, 0], wav=wav, spectra_df=iter1_outlier, label="Iter1 outlier")
        
        subplot_helper(
            ax=ax[0, 1], wav=wav, spectra_df=remaining_after_iter1, label="Remaining after iter1", display_stats=True)
        subplot_helper(
            ax=ax[1, 1], wav=wav, spectra_df=slope576_outlier, label="Slope outliers")
        
        subplot_helper(
            ax=ax[2, 1], wav=wav, spectra_df=remaining_after_slope, label="Remaining after slope", display_stats=True)
        subplot_helper(
            ax=ax[3, 1], wav=wav, spectra_df=iter2_outlier, label="Iter2 outlier")

        subplot_helper(
            ax=ax[3, 2], wav=wav, spectra_df=valid_spectra_df, label="Valid sp", display_stats=True)
        
        plt.tight_layout()

    return valid_spectra_df, fig

