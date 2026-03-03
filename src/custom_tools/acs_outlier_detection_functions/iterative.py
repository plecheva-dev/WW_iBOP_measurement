import numpy as np

def apply_iterative_outlier_detection(spectra_df, threshold_iter=10.0, max_iter=50):
    # Convert to numpy immediately for heavy math: (Samples, Wavelengths)  
    data = spectra_df.values 
    n_samples = data.shape[0]
    outlier_mask = np.zeros(n_samples, dtype=bool)

    for _ in range(max_iter):
        valid_indices = np.where(~outlier_mask)[0]
        
        if len(valid_indices) == 0:
            break
            
        # Select rows using numpy indexing
        current = data[valid_indices]

        # 1. Mean spectrum of the current valid set
        med_current = np.median(current, axis=0)

        # 2. Euclidean distances of each valid spectrum to the mean
        # current - mean_current broadcasts correctly in NumPy
        diffs = current - med_current
        distances = np.linalg.norm(diffs, axis=1)

        # 3. Robust MAD-based threshold
        med = np.median(distances)
        mad = np.median(np.abs(distances - med)) + 1e-12
        limit = med + (threshold_iter * mad)

        # 4. Identify the 'worst' spectrum in the current set
        worst_local_idx = np.argmax(distances)
        worst_distance = distances[worst_local_idx]

        # STOP if the most extreme spectrum is within the robust limit
        if worst_distance <= limit:
            break

        # 5. Flag the global index
        worst_global_idx = valid_indices[worst_local_idx]
        outlier_mask[worst_global_idx] = True

    valid_index = np.array(spectra_df[~outlier_mask].index)
    iterative_outlier_index = np.array(spectra_df[outlier_mask].index)
    
    return iterative_outlier_index, valid_index

def apply_asymmetric_outlier_detection(spectra_df, threshold_iter=5.0, max_iter=50, low_percentile=10, mad_floor_perc=0.1):
    data = spectra_df.values 
    n_samples = data.shape[0]
    outlier_mask = np.zeros(n_samples, dtype=bool)

    for _ in range(max_iter):
        valid_indices = np.where(~outlier_mask)[0]
        if len(valid_indices) <= 2: # Changed to 2: need enough for statistics
            break
            
        current = data[valid_indices]
        ref_spectrum = np.percentile(current, q=low_percentile, axis=0)
        mad_floor = max(0.25, mad_floor_perc * ref_spectrum.mean(axis=0))
        
        # Calculate positive-only residuals
        positive_diffs = np.maximum(current - ref_spectrum, 0)
        scores = np.linalg.norm(positive_diffs, axis=1)

        # Robust MAD-based threshold
        med_score = np.median(scores)
        mad_score = np.median(np.abs(scores - med_score))

        # --- THE REFINED LOGIC ---
        # Instead of breaking, we enforce a minimum "spread" (the noise floor).
        # This prevents the threshold from becoming 0, but still allows us 
        # to catch a few stray outliers that are way above the floor.
        effective_mad = max(mad_score, mad_floor)
        
        limit = med_score + (threshold_iter * effective_mad)
        # -------------------------

        worst_local_idx = np.argmax(scores)
        worst_score = scores[worst_local_idx]

        # If the worst spectrum is still within our "Noise Floor Gate", we stop.
        # Otherwise, we keep removing until the "Monsters" are gone.
        if worst_score <= limit:
            break

        outlier_mask[valid_indices[worst_local_idx]] = True

    return spectra_df.index[outlier_mask].values, spectra_df.index[~outlier_mask].values