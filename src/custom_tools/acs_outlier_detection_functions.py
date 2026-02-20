
import numpy as np


def get_outlier_iterative_mask(spectra_arr, threshold=10.0, max_iter=50):
    """
    Iteratively remove outliers based on MAD distance to current mean.
    Removes 1 spectrum at a time (the worst), recomputes mean each iteration.
    """

    outlier_mask = np.ones(spectra_arr.shape[0], dtype=bool)

    for _ in range(max_iter):
        current = spectra_arr[outlier_mask]

        # Mean of remaining spectra
        mean_current = np.mean(current, axis=0)

        # Distances to mean
        distances = np.linalg.norm(current - mean_current, axis=1)

        # Robust threshold (median + threshold*MAD)
        med = np.median(distances)
        mad = np.median(np.abs(distances - med)) + 1e-12
        limit = med + threshold * mad

        # Worst spectrum
        worst_local = np.argmax(distances)
        worst_distance = distances[worst_local]

        # STOP if all spectra are within limit
        if worst_distance <= limit:
            break

        # Map to global index
        worst_global = np.where(outlier_mask)[0][worst_local]
        outlier_mask[worst_global] = False  # remove only this one

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