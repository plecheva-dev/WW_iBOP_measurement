import numpy as np

def apply_slope576_outlier_detection_and_correction(spectra_df, n_slope576=5, threshold_slope576=0.5):
    
    valid_index = []
    slope576_outlier_index = []
    wav = np.array(spectra_df.columns.astype(float))
    df_cleaned = spectra_df.copy()
    target_wav_pre, target_wav_post = _detect_stitch_wavelengths(wav)
    
    for idx, sp in spectra_df.iterrows():
        wav_pre_seg, sp_pre_seg = _get_pre_seg(sp, wav, target_wav_pre, n_slope576)
        wav_post_seg, sp_post_seg = _get_post_seg(sp, wav, target_wav_post, n_slope576)
        
        if not _is_stitch_needed(
            sp_pre=sp[target_wav_pre], 
            sp_post=sp[target_wav_post], 
            rel_threshold=0.01
            ):
            
            valid_index.append(idx)
            continue
        
        # 2. Linear fits: y = mx + c
        a_pre, b_pre = np.polyfit(wav_pre_seg, sp_pre_seg, 1)
        a_post, b_post = np.polyfit(wav_post_seg, sp_post_seg, 1)
        
        stitch_validity, _ = _check_stitch_validity(
            slope_pre=a_pre, slope_post=a_post, threshold=threshold_slope576
            ) 
        
        if stitch_validity:
        
            x_mid = (target_wav_pre + target_wav_post) / 2.0
            y_pre_mid = a_pre * x_mid + b_pre
            y_post_mid = a_post * x_mid + b_post
            
            # 4. Calculate the required shift
            offset = y_pre_mid - y_post_mid
            
            # 5. Apply the offset to all values above the gap
            sp_stitched = sp.copy()
            sp_stitched[wav >= target_wav_post] += offset
            
            df_cleaned.loc[idx, :] = sp_stitched
            
            valid_index.append(idx)
        else:
            slope576_outlier_index.append(idx)
        
    return slope576_outlier_index, valid_index, df_cleaned

def _is_stitch_needed(sp_pre, sp_post, rel_threshold=0.01):
    """
    Checks if the gap between two segments is large enough to justify stitching.
    Returns True if the gap is > rel_threshold, False if the data is continuous.
    """
    
    # Calculate absolute difference
    abs_gap = np.abs(sp_pre - sp_post)
    meean_val = (np.abs(sp_pre) + np.abs(sp_post)) / 2.0 + 1e-9
    rel_gap = abs_gap / meean_val
    
    # If the gap is greater than the threshold (e.g., 10%), it needs attention
    return rel_gap > rel_threshold

def _detect_stitch_wavelengths(wav, gap_center=576.0):
    """
    Finds the wavelengths in the array immediately before and after the gap_center.
    """
    # Find the index where gap_center would be inserted to maintain order
    idx_post = np.searchsorted(wav, gap_center)
    
    # The 'pre' index is the one immediately before it
    idx_pre = idx_post - 1
    
    # Boundary checks to ensure the center isn't outside the spectral range
    if idx_post >= len(wav) or idx_pre < 0:
        raise ValueError(f"Gap center {gap_center} is outside the range of the wavelength array.")
        
    target_wav_pre = wav[idx_pre]
    target_wav_post = wav[idx_post]
    
    return target_wav_pre, target_wav_post


def _get_pre_seg(sp, wav, target_wav_pre, n):
    idx_pre = np.where(wav <= target_wav_pre)[0][-1]
    wav_pre_seg = wav[idx_pre - n + 1 : idx_pre + 1]
    sp_pre_seg = sp.iloc[idx_pre - n + 1 : idx_pre + 1]
    return wav_pre_seg, sp_pre_seg

def _get_post_seg(sp, wav, target_wav_post, n):
    idx_post = np.where(wav >= target_wav_post)[0][0]
    wav_post_seg = wav[idx_post : idx_post + n]
    sp_post_seg = sp.iloc[idx_post : idx_post + n]
    return wav_post_seg, sp_post_seg
    
def _check_stitch_validity(slope_pre, slope_post, threshold=0.5):
  
    # Calculate the absolute difference
    abs_diff = np.abs(slope_pre - slope_post)
    
    # Calculate the average magnitude of the slopes
    # Adding 1e-9 (epsilon) to prevent division by zero
    avg_mag = (np.abs(slope_pre) + np.abs(slope_post)) / 2.0 + 1e-9
    
    # Calculate relative difference
    rel_diff = abs_diff / avg_mag
    
    # Valid if the change is below the threshold
    is_valid = rel_diff <= threshold
    
    return is_valid, rel_diff
