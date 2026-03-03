
import numpy as np
from scipy.ndimage import label

def apply_minmax_outlier_detection_and_correction(spectra_df, threshold_low=0.01, threshold_high=99, n_max_features=5):
    
    valid_index = []
    minmax_outlier_index = []
    wav = np.array(spectra_df.columns.astype(float))
    df_cleaned = spectra_df.copy()
    
    for idx, sp in spectra_df.iterrows():
        sp_minmax_mask = _get_minmax_mask(sp, threshold_low, threshold_high)
        is_recoverable = _determine_if_mask_is_recoverable(sp_minmax_mask, max_gap=2, n_max_features=n_max_features)
        if is_recoverable:
            updated_sp, _ = _interpolate_values_in_recoverable_spectrum(
                mask=sp_minmax_mask, wavelengths=wav, spectrum=sp)
            df_cleaned.loc[idx, :] = updated_sp
            valid_index.append(idx)
        else:
            minmax_outlier_index.append(idx)
            
    return minmax_outlier_index, valid_index, df_cleaned

def _get_minmax_mask(sp_arr, threshold_low, threshold_high):
    return (sp_arr < threshold_low) | (sp_arr > threshold_high)

def _determine_if_mask_is_recoverable(mask, max_gap: int = 2, n_max_features: int = 5):
    """
    Checks if a spectrum can be repaired based on gap size and total gap count.
    """
    # labels: an array where each outlier cluster has a unique ID
    # num_features: the total number of distinct outlier clusters found
    labels, num_features = label(mask)
    
    # 1. Check if the total number of outlier events is too high
    if num_features > n_max_features:
        return False
    
    # 2. Check if any individual outlier cluster is too wide
    if num_features > 0:
        for feat in range(1, num_features + 1):
            cluster_size = np.sum(labels == feat)
            if cluster_size > max_gap:
                return False
                
    return True

def _interpolate_values_in_recoverable_spectrum(mask, wavelengths, spectrum):
    ok_indices = ~mask
    # Convert to values to ensure numpy handles the arrays cleanly
    interp_values = np.interp(
        wavelengths[mask.values], 
        wavelengths[ok_indices.values], 
        spectrum.values[ok_indices.values]
    )
    cleaned_spectrum = spectrum.copy()
    cleaned_spectrum[mask] = interp_values
    return cleaned_spectrum, interp_values