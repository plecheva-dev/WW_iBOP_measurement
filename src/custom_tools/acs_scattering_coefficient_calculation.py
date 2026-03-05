import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def calculate_scattering_for_sample(sample_name, df_a, df_c):
    """
    Calculates Scattering (S = C - A) for a single sample by interpolating 
    C wavelengths to match A wavelengths.
    """
    # 1. Extract rows for the specific sample
    row_a = df_a[df_a["full_sample_name"] == sample_name]
    row_c = df_c[df_c["full_sample_name"] == sample_name]

    # 2. Extract wavelengths (assuming columns are numeric wavelength strings)
    # We exclude metadata columns to get just the spectral data
    meta_cols = ["full_sample_name", "index"] # Add any other non-numeric columns here
    
    waves_a = np.array([float(col) for col in df_a.columns if col not in meta_cols])
    waves_c = np.array([float(col) for col in df_c.columns if col not in meta_cols])
    
    vals_a = row_a.drop(columns=[c for c in row_a.columns if c in meta_cols]).values.flatten()
    vals_c = row_c.drop(columns=[c for c in row_c.columns if c in meta_cols]).values.flatten()

    # 3. Linear Interpolation
    # We interpolate C-values to the wavelengths of A
    f_interp_c = interp1d(waves_c, vals_c, kind='linear', fill_value="extrapolate")
    vals_c_interp = f_interp_c(waves_a)

    # 4. Calculate Scattering: S = C - A
    vals_s = vals_c_interp - vals_a
    
    return waves_a, vals_s

def generate_scattering_df(df_a, df_c):
    """
    Loops through samples and builds the final scattering dataframe.
    """
    scattering_dict = {}
    
    # Get common samples
    samples_in_a = set(df_a["full_sample_name"].unique())
    samples_in_c = set(df_c["full_sample_name"].unique())
    common_samples = samples_in_a.intersection(samples_in_c)
    
    waves_out = None

    for sample in common_samples:
        try:
            waves, s_values = calculate_scattering_for_sample(sample, df_a, df_c)
            scattering_dict[sample] = s_values
            if waves_out is None:
                waves_out = waves
        except Exception as e:
            print(f"Error processing scattering for {sample}: {e}")

    # Create DataFrame
    df_s = pd.DataFrame.from_dict(scattering_dict, orient='index', columns=waves_out)
    df_s.reset_index(inplace=True)
    df_s = df_s.rename(columns={"index": "full_sample_name"})
    
    return df_s