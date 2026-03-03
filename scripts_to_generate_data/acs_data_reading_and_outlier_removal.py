from custom_tools.acs_data_reader import get_acs_IOP
from custom_tools.acs_outlier_detection_functions_advanced import run_advanced_spectra_cleaning_pipeline_alt
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

##### variables and functions

outlier_A = [
    102, 346, 406, 412, 436, 
    453, 455, 463, 481, 487
]

outlier_S = [
    7, 13, 38, 47, 50, 
    64, 70, 73, 82, 83, 
    84, 394, 397, 400, 406, 
    410, 411, 436, 437, 453, 
    455, 463, 466
]

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

def concatenate_df(df_cumm, df):
    # Check if df_cumm is None or empty
    if df_cumm is None or df_cumm.empty:
        df_cumm = df.copy()
    else:
        df_cumm = pd.concat([df_cumm, df], ignore_index=True)
    
    return df_cumm

def detect_sample_type(sample_name):
    return "diw" if "diw" in sample_name else "sample"

def convert_run_list_to_string(run_list):
    for (i, r) in enumerate(run_list):
        if r<10:
            run_list[i] = f"00{r}"
        elif r<100:
            run_list[i] = f"0{r}"
        else:
            run_list[i] = str(r)
    # Return as a list
    return run_list

def get_runs_from_samplename(samplename, metadata):
    # Filter rows where sample_name matches, then grab the run_nr column
    runs = metadata.loc[metadata["sample_name"] == samplename, "run_nr"].tolist()
    runs = convert_run_list_to_string(runs)
    return runs

def handle_exception(run, e):
    print(f"FAILED FOR RUN: {run}")
    print(f"ERROR TYPE: {type(e).__name__}")
    print(f"ERROR MESSAGE: {e}")

def analyse_clean_sp_df(clean_sp_df):
    n_sp = clean_sp_df.shape[0]
    median_sp = clean_sp_df.median(axis=0)
    p25 = clean_sp_df.quantile(0.25, axis=0)
    p75 = clean_sp_df.quantile(0.75, axis=0)
    return n_sp, median_sp, p25, p75
###### preparing data

dir_path = "data/raw/1_acs_runs/runs"

file_list = [f for f in os.listdir(dir_path) if f.startswith('run_21_ACS.')]

metadata = pd.read_csv("data/raw/1_acs_runs/metadata_acs.csv")

samplename_list = metadata["sample_name"].unique()

outlier_A = convert_run_list_to_string(outlier_A)
outlier_S = convert_run_list_to_string(outlier_S)

clean_spectra_A_dict = {}
clean_spectra_S_dict = {}

##### Main function

def main():
    for samplename in samplename_list:
        if "diw" not in samplename:
            run_list = get_runs_from_samplename(samplename=samplename, metadata=metadata)

            df_arr_A_cumm = pd.DataFrame() 
            df_arr_S_cumm = pd.DataFrame() 

            for i, run in enumerate(run_list):
                                
                try:
                    f = f"run_21_ACS.{run}"
                    df_arr_A, df_arr_S = get_acs_IOP(os.path.join(dir_path, f))
                    
                    if run not in outlier_A:
                        df_arr_A_cumm = concatenate_df(df_arr_A_cumm, df_arr_A)
                    
                    if run not in outlier_S:
                        df_arr_S_cumm = concatenate_df(df_arr_S_cumm, df_arr_S)
                    
                except Exception as e:
                    handle_exception(run=run, e=e)

            n_cumm_sp_A = df_arr_A_cumm.shape[0]
            if n_cumm_sp_A>0:
                df_arr_A_cumm_clean, _ = run_advanced_spectra_cleaning_pipeline_alt(df_arr_A_cumm, plot=False) 
                _, med_sp_A, _, _ = analyse_clean_sp_df(df_arr_A_cumm_clean)
                clean_spectra_A_dict[samplename] = med_sp_A
                
            n_cumm_sp_S = df_arr_S_cumm.shape[0]
            if n_cumm_sp_S>0:
                df_arr_S_cumm_clean, _ = run_advanced_spectra_cleaning_pipeline_alt(df_arr_S_cumm, plot=False)
                _, med_sp_S, _, _ = analyse_clean_sp_df(df_arr_S_cumm_clean)
                clean_spectra_S_dict[samplename] = med_sp_S
                
                
    col_A = df_arr_A.columns
    clean_spectra_A_df = pd.DataFrame.from_dict(clean_spectra_A_dict, orient='index', columns=col_A)
    clean_spectra_A_df.reset_index(inplace=True)
    clean_spectra_A_df = clean_spectra_A_df.rename(columns={"index": "full_sample_name"})
    clean_spectra_A_df = preprocess_final_sp_df(clean_spectra_A_df)
    clean_spectra_A_df.to_csv("data/processed/acs_a_clean_df.csv", index_label="index")
    
    col_S = df_arr_S.columns
    clean_spectra_S_df = pd.DataFrame.from_dict(clean_spectra_S_dict, orient='index', columns=col_S)
    clean_spectra_S_df.reset_index(inplace=True)
    clean_spectra_S_df = clean_spectra_S_df.rename(columns={"index": "full_sample_name"})
    clean_spectra_S_df = preprocess_final_sp_df(clean_spectra_S_df)
    clean_spectra_S_df.to_csv("data/processed/acs_s_clean_df.csv", index_label="index")
        
if __name__ == "__main__":
    main()