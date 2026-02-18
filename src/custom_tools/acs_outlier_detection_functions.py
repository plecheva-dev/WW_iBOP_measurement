
import numpy as np


def iterative_clean_mean_spectra(spectra, threshold=10.0, max_iter=50):
    """
    Iteratively remove outliers based on MAD distance to current mean.
    Removes 1 spectrum at a time (the worst), recomputes mean each iteration.
    """

    mask = np.ones(spectra.shape[0], dtype=bool)

    for _ in range(max_iter):
        current = spectra[mask]

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
        worst_global = np.where(mask)[0][worst_local]
        mask[worst_global] = False  # remove only this one

    # Final outputs
    mean_clean = np.mean(spectra[mask], axis=0)
    std_clean = np.std(spectra[mask], axis=0)
    return mean_clean, mask, std_clean

def calc_absolute_derivative_spectrum(spectrum):
    deriv_spectrum = np.abs(np.gradient(spectrum))
    return deriv_spectrum

def detect_outlier_spectrum_575nm(spectrum):
    deriv_spectrum = calc_absolute_derivative_spectrum(spectrum)
    value_575nm = deriv_spectrum[43]

    Local_median_deriv = np.median(deriv_spectrum[40:46])

    if value_575nm > Local_median_deriv+0.001:
        return True
    else:
        return False

def detect_outlier_spectrum_max_value(spectrum):

    max = spectrum.max()
    if max > 99:
        return True
    else:
        return False