
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def run_spectra_cleaning_pipeline(df, wav, threshold_575=0.001, iterative_threshold=10.0, plot: bool = True):
    # ---------------------------------------------------------
    # 1. CALCULATIONS & MASKING (Logic Phase)
    # ---------------------------------------------------------
    spectra_values = df.values[:, 1:]
    
    # i) MAX outliers (True = Outlier)
    max_outlier_mask = get_outlier_max_mask(df)
    df_after_max = df[~max_outlier_mask]
    
    # ii) 575nm Deriv outliers (True = Outlier)
    # Calculated only on those that passed the MAX filter
    deriv_outlier_mask = get_outlier_deriv575nm_mask(df_after_max, threshold=threshold_575)
    df_after_max_and_575nm = df_after_max[~deriv_outlier_mask]
    rem_values = df_after_max_and_575nm.values[:, 1:]
    
    # iii) Iterative Outliers (True = Outlier)
    iter_outlier_mask = np.array([])
    clean_df = pd.DataFrame()
    
    if len(rem_values) > 0:
        # Using the updated function where True = Outlier
        iter_outlier_mask = get_outlier_iterative_mask(rem_values, threshold=iterative_threshold)
        clean_df = df_after_max_and_575nm[~iter_outlier_mask]
        iter_outlier_values = rem_values[iter_outlier_mask]
        clean_values = rem_values[~iter_outlier_mask]
    else:
        iter_outlier_values = np.array([])
        clean_values = np.array([])

    # ---------------------------------------------------------
    # 2. PLOTTING (Presentation Phase)
    # ---------------------------------------------------------
    fig = None
    if plot:
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 8), sharex=True)
        
        # Step 1: Original Data
        axes[0, 0].plot(wav, spectra_values.T, color="gray", alpha=0.1, lw=0.5)
        m0, s0 = np.mean(spectra_values, axis=0), np.std(spectra_values, axis=0)
        axes[0, 0].plot(wav, m0, color="red", lw=2, label="Mean")
        axes[0, 0].fill_between(wav, m0-s0, m0+s0, color="red", alpha=0.2)
        axes[0, 0].set_title(f"1. Original Spectra (n={len(df)})")

        # Step 2: MAX Outliers
        if max_outlier_mask.any():
            axes[1, 0].plot(wav, df[max_outlier_mask].values[:, 1:].T, color="black", alpha=0.6)
        axes[1, 0].set_title(f"2. MAX Outliers (n={max_outlier_mask.sum()})")

        # Step 3: 575nm Deriv Outliers
        if deriv_outlier_mask.any():
            axes[2, 0].plot(wav, df_after_max[deriv_outlier_mask].values[:, 1:].T, color="purple", alpha=0.6)
        axes[2, 0].set_title(f"3. 575nm Deriv Outliers (n={deriv_outlier_mask.sum()})")

        # Step 4: Remaining after hard filters
        if len(rem_values) > 0:
            axes[0, 1].plot(wav, rem_values.T, color="gray", alpha=0.2, lw=0.5)
            m3, s3 = np.mean(rem_values, axis=0), np.std(rem_values, axis=0)
            axes[0, 1].plot(wav, m3, color="blue", lw=2)
            axes[0, 1].fill_between(wav, m3-s3, m3+s3, color="blue", alpha=0.2)
            axes[0, 1].set_title(f"4. Post-Hard Filters (n={len(df_after_max_and_575nm)})")

            # Step 5: Iterative Outliers Removed
            if len(iter_outlier_values) > 0:
                axes[1, 1].plot(wav, iter_outlier_values.T, color="orange", alpha=0.6)
            axes[1, 1].set_title(f"5. Iterative Outliers Removed (n={len(iter_outlier_values)})")

            # Step 6: Final Cleaned
            if len(clean_values) > 0:
                axes[2, 1].plot(wav, clean_values.T, color="black", alpha=0.1, lw=0.5)
                m5, s5 = np.mean(clean_values, axis=0), np.std(clean_values, axis=0)
                axes[2, 1].plot(wav, m5, color="green", lw=2)
                axes[2, 1].fill_between(wav, m5-s5, m5+s5, color="green", alpha=0.2)
            axes[2, 1].set_title(f"6. Final Cleaned (n={len(clean_values)})")
            axes[2, 1].set_xlabel("Wavelength (nm)")

        plt.tight_layout()

    # Return the clean dataframe and the figure handle
    return clean_df, fig

def get_outlier_iterative_mask(spectra_arr, threshold=10.0, max_iter=50):
    """
    Iteratively flags outliers based on MAD distance to current mean.
    Returns an outlier_mask where True = Outlier, False = Valid Data.
    """
    # Start with NO outliers (all False)
    outlier_mask = np.zeros(spectra_arr.shape[0], dtype=bool)

    for _ in range(max_iter):
        # Identify currently 'valid' indices to compute the mean
        valid_indices = np.where(~outlier_mask)[0]
        
        if len(valid_indices) == 0:
            break
            
        current = spectra_arr[valid_indices]

        # 1. Mean of currently valid spectra
        mean_current = np.mean(current, axis=0)

        # 2. Distances of valid spectra to that mean
        distances = np.linalg.norm(current - mean_current, axis=1)

        # 3. Robust threshold (median + threshold * MAD)
        med = np.median(distances)
        mad = np.median(np.abs(distances - med)) + 1e-12
        limit = med + threshold * mad

        # 4. Find the 'worst' spectrum among the current valid set
        worst_local_idx = np.argmax(distances)
        worst_distance = distances[worst_local_idx]

        # STOP if even the worst remaining spectrum is within the limit
        if worst_distance <= limit:
            break

        # 5. Map the local 'worst' index back to the global array index
        worst_global_idx = valid_indices[worst_local_idx]
        
        # Flag it as an outlier (True)
        outlier_mask[worst_global_idx] = True

    return outlier_mask



### Outlier detection based on 575mn derivative

def get_outlier_deriv575nm_mask(spectra_arr, threshold=0.001): 
    outlier_mask = spectra_arr.apply(_detect_outlier_spectrum_deriv575nm, axis=1, args=(threshold,))
    return outlier_mask

def _calc_absolute_derivative_spectrum(spectrum):
    deriv_spectrum = np.abs(np.gradient(spectrum))
    return deriv_spectrum

def _detect_outlier_spectrum_deriv575nm(spectrum, threshold=0.001):
    # I need to change the hardcoded 575nm value
    deriv_spectrum = _calc_absolute_derivative_spectrum(spectrum)
    value_575nm = deriv_spectrum[43]

    Local_median_deriv = np.median(deriv_spectrum[40:46])

    if value_575nm > Local_median_deriv+threshold:
        return True
    else:
        return False


### Outlier detection based on max values

def get_outlier_max_mask(spectra_arr): 
    outlier_mask = spectra_arr.apply(_detect_outlier_spectrum_max_value, axis=1)
    return outlier_mask

def _detect_outlier_spectrum_max_value(spectrum):

    max = spectrum.max()
    if max > 99:
        return True
    else:
        return False