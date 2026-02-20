from custom_tools.acs_data_reader import get_acs_IOP
from custom_tools.acs_outlier_detection_functions import detect_outlier_spectrum_575nm
from custom_tools.acs_outlier_detection_functions import _detect_outlier_spectrum_max_value
from custom_tools.acs_outlier_detection_functions import iterative_clean_mean_spectra
import pandas as pd
import numpy as np
import os

import warnings
warnings.filterwarnings("ignore")

def main():

    dir_path = "data/raw/1_acs_runs/runs"
    file_list = [f for f in os.listdir(dir_path) if f.startswith('run_21_ACS.')]

    metadata = pd.read_csv("data/raw/1_acs_runs/metadata_acs.csv")

    def detect_sample_type(sample_name):
        return "diw" if "diw" in sample_name else "sample"

    metadata["sample_type"] = metadata["sample_name"].apply(detect_sample_type)

    a_df = pd.DataFrame()
    c_df = pd.DataFrame()

    for f in file_list:
        df_arr_A, df_arr_C = get_acs_IOP(os.path.join(dir_path, f))

        df_arr_A = df_arr_A[~df_arr_A.apply(detect_outlier_spectrum_575nm, axis=1)]
        df_arr_C = df_arr_C[~df_arr_C.apply(detect_outlier_spectrum_575nm, axis=1)]

        df_arr_A = df_arr_A[~df_arr_A.apply(_detect_outlier_spectrum_max_value, axis=1)]
        df_arr_C = df_arr_C[~df_arr_C.apply(_detect_outlier_spectrum_max_value, axis=1)]

        spectra_A = df_arr_A.iloc[:, 1:].to_numpy()
        spectra_C = df_arr_C.iloc[:, 1:].to_numpy()
        mean_A, mask_A, std_A = iterative_clean_mean_spectra(spectra_A, threshold=10.0, max_iter=spectra_A.shape[0]-1)
        mean_C, mask_C, std_C = iterative_clean_mean_spectra(spectra_C, threshold=10.0, max_iter=spectra_C.shape[0]-1)
        n_valid_spectra_A = mask_A.sum()
        n_valid_spectra_C = mask_C.sum()

        # putting n_valid_spectra_a, mean_A and std_A into an array
        data_to_save_A = np.concatenate(([n_valid_spectra_A], mean_A, std_A))
        data_to_save_C = np.concatenate(([n_valid_spectra_C], mean_C, std_C))

        if len(data_to_save_A) == 169 and len(data_to_save_C) == 169:
            run_nr = int(f[-3:])

            a_df[run_nr] = data_to_save_A
            c_df[run_nr] = data_to_save_C
        
        else:
            print(f"Skipping file {f} due to unexpected data length.")
            continue
        
    a_df = a_df.T
    a_df.rename_axis('run_nr', inplace=True)
    a_df.reset_index(inplace=True)
    a_df.columns = ['run_nr', 'n_valid_spectra'] + [wl for wl in df_arr_A.columns[1:].astype(float)] + [f'std_{wl}' for wl in df_arr_A.columns[1:].astype(float)]

    a_df = metadata.merge(a_df, on='run_nr')

    c_df = c_df.T
    c_df.rename_axis('run_nr', inplace=True)
    c_df.reset_index(inplace=True)
    c_df.columns = ['run_nr', 'n_valid_spectra'] + [wl for wl in df_arr_C.columns[1:].astype(float)] + [f'std_{wl}' for wl in df_arr_C.columns[1:].astype(float)]
    c_df = metadata.merge(c_df, on='run_nr')
    
    save_path = "data/processed/"
    
    a_df.to_csv(os.path.join(save_path, "a_df.csv"))
    c_df.to_csv(os.path.join(save_path, "c_df.csv"))
    
if __name__ == "__main__":
    main()