from custom_tools.acs_data_reader import get_acs_IOP
from custom_tools.acs_outlier_detection_functions_advanced import run_advanced_spectra_cleaning_pipeline_alt
from custom_tools.acs_utils import get_runs_list_from_samplename_and_metadata, concatenate_df 
from custom_tools.acs_utils import handle_exception_when_run_not_working, get_sp_df_stats, convert_run_list_to_string
import pandas as pd
from scipy.interpolate import interp1d
import numpy as np
import json
import os
import warnings
warnings.filterwarnings("ignore")

##### UTILS



def calculate_scattering(med_sp_a, waves_a, med_sp_c, waves_c):
    """
    Interpolates Beam Attenuation (C) to Absorption (A) wavelengths 
    and subtracts them to find Scattering (B).
    """
    # Create the interpolation function for C
    # kind='linear' is standard; fill_value="extrapolate" handles slight range mismatches
    f_interp_c = interp1d(waves_c.astype(float), med_sp_c, kind='linear', fill_value="extrapolate")
    
    # Get C values at A wavelengths
    c_interpolated = f_interp_c(waves_a.astype(float))
    
    # b = c - a
    scattering = c_interpolated - med_sp_a
    return scattering

def preprocess_final_sp_df(df):
    # 1. Split the 'full_sample_name' column
    # This creates a temporary DataFrame with two columns
    metadata = df['full_sample_name'].str.split('_', expand=True)

    # 2. Assign the new columns to your original DataFrame
    df['sample'] = metadata[0]
    df['preprocessing'] = metadata[1]

    # 3. Reorder columns to put the new ones at the front
    # We grab the new names, then every name that isn't those two
    cols = ['sample', 'preprocessing'] + [c for c in df.columns if c not in ['sample', 'preprocessing']]
    df = df[cols]
    return df



###### preparing data

dir_path = "data/raw/1_acs_runs/runs"

file_list = [f for f in os.listdir(dir_path) if f.startswith('run_21_ACS.')]

metadata = pd.read_csv("data/raw/1_acs_runs/metadata_acs.csv")

samplename_list = metadata["sample_name"].unique()

with open("data/processed/outlier.json", 'r') as f:
    data = json.load(f)

outlier_A = data['outlier_A']
outlier_C = data['outlier_C']

outlier_A = convert_run_list_to_string(outlier_A)
outlier_C = convert_run_list_to_string(outlier_C)

clean_spectra_A_dict = {}
clean_spectra_C_dict = {}
clean_spectra_B_dict = {}

##### Main function

def main():
    for samplename in samplename_list:
        if "diw" not in samplename:
            run_list = get_runs_list_from_samplename_and_metadata(samplename=samplename, metadata=metadata)

            df_arr_A_cumm = pd.DataFrame() 
            df_arr_C_cumm = pd.DataFrame() 

            for i, run in enumerate(run_list):
                                
                try:
                    f = f"run_21_ACS.{run}"
                    df_arr_A, df_arr_C = get_acs_IOP(os.path.join(dir_path, f))
                    
                    if run not in outlier_A:
                        df_arr_A_cumm = concatenate_df(df_arr_A_cumm, df_arr_A)
                    
                    if run not in outlier_C:
                        df_arr_C_cumm = concatenate_df(df_arr_C_cumm, df_arr_C)
                    
                except Exception as e:
                    handle_exception_when_run_not_working(run=run, e=e)

            n_cumm_sp_A = df_arr_A_cumm.shape[0]
            if n_cumm_sp_A>0:
                df_arr_A_cumm_clean, _ = run_advanced_spectra_cleaning_pipeline_alt(df_arr_A_cumm, plot=False) 
                stats_A_dict = get_sp_df_stats(df_arr_A_cumm_clean)
                clean_spectra_A_dict[samplename] = stats_A_dict["median_spectrum"]
                
            n_cumm_sp_C = df_arr_C_cumm.shape[0]
            if n_cumm_sp_C>0:
                df_arr_C_cumm_clean, _ = run_advanced_spectra_cleaning_pipeline_alt(df_arr_C_cumm, plot=False)
                stats_C_dict = get_sp_df_stats(df_arr_C_cumm_clean)
                clean_spectra_C_dict[samplename] = stats_C_dict["median_spectrum"]
                
            if n_cumm_sp_A > 0 and n_cumm_sp_C > 0:
                # Extract median spectra and wavelength arrays
                med_a = stats_A_dict["median_spectrum"]
                waves_a = df_arr_A_cumm_clean.columns.astype("float")
                
                med_c = stats_C_dict["median_spectrum"]
                waves_c = df_arr_C_cumm_clean.columns.astype("float")

                # Calculate scattering
                scattering_spectrum = calculate_scattering(med_a, waves_a, med_c, waves_c)
                
                # Store in dictionary (using A wavelengths as the reference grid)
                clean_spectra_B_dict[samplename] = scattering_spectrum
                
                
    col_A = df_arr_A.columns
    clean_spectra_A_df = pd.DataFrame.from_dict(clean_spectra_A_dict, orient='index', columns=col_A)
    clean_spectra_A_df.reset_index(inplace=True)
    clean_spectra_A_df = clean_spectra_A_df.rename(columns={"index": "full_sample_name"})
    clean_spectra_A_df = preprocess_final_sp_df(clean_spectra_A_df)
    clean_spectra_A_df.to_csv("data/processed/2026-03-04_acs_a_clean_df.csv", index_label="index")
    
    col_C = df_arr_C.columns
    clean_spectra_C_df = pd.DataFrame.from_dict(clean_spectra_C_dict, orient='index', columns=col_C)
    clean_spectra_C_df.reset_index(inplace=True)
    clean_spectra_C_df = clean_spectra_C_df.rename(columns={"index": "full_sample_name"})
    clean_spectra_C_df = preprocess_final_sp_df(clean_spectra_C_df)
    clean_spectra_C_df.to_csv("data/processed/2026-03-04_acs_c_clean_df.csv", index_label="index")
    
    col_B = col_A
    clean_spectra_B_df = pd.DataFrame.from_dict(clean_spectra_B_dict, orient='index', columns=col_B)
    clean_spectra_B_df.reset_index(inplace=True)
    clean_spectra_B_df = clean_spectra_B_df.rename(columns={"index": "full_sample_name"})
    clean_spectra_B_df = preprocess_final_sp_df(clean_spectra_B_df)
    clean_spectra_B_df.to_csv("data/processed/2026-03-04_acs_b_clean_df.csv", index_label="index")

        
if __name__ == "__main__":
    main()