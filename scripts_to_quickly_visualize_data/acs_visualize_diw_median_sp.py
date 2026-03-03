###
# This script gathers all the diw data from the ACs measurement and plot the median and IQRs.# 
# ###

from custom_tools.acs_data_reader import get_acs_IOP
from custom_tools.acs_utils import get_runs_list_from_samplename_and_metadata, concatenate_df 
from custom_tools.acs_utils import handle_exception_when_run_not_working, get_sp_df_stats
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

###### preparing data

dir_path = "data/raw/1_acs_runs/runs"

file_list = [f for f in os.listdir(dir_path) if f.startswith('run_21_ACS.')]

metadata = pd.read_csv("data/raw/1_acs_runs/metadata_acs.csv")

samplename_list = metadata["sample_name"].unique()

clean_spectra_A_dict = {}
clean_spectra_S_dict = {}

##### Main function

def main():
    for samplename in samplename_list:
        if "diw" in samplename:
            run_list = get_runs_list_from_samplename_and_metadata(samplename=samplename, metadata=metadata)

            df_arr_A_cumm = pd.DataFrame() 
            df_arr_S_cumm = pd.DataFrame() 

            for i, run in enumerate(run_list):
                                
                try:
                    f = f"run_21_ACS.{run}"
                    df_arr_A, df_arr_S = get_acs_IOP(os.path.join(dir_path, f))
                    
                    df_arr_A_cumm = concatenate_df(df_arr_A_cumm, df_arr_A)
                    df_arr_S_cumm = concatenate_df(df_arr_S_cumm, df_arr_S)
                    
                except Exception as e:
                    handle_exception_when_run_not_working(run=run, e=e)

    n_cumm_sp_A = df_arr_A_cumm.shape[0]
    wav_A = np.array(df_arr_A_cumm.columns.astype("float"))
    if n_cumm_sp_A>0:
        sp_A_dict = get_sp_df_stats(df_arr_A_cumm)
        
    n_cumm_sp_S = df_arr_S_cumm.shape[0]
    wav_S = np.array(df_arr_S_cumm.columns.astype("float"))
    if n_cumm_sp_S>0:
        sp_S_dict = get_sp_df_stats(df_arr_S_cumm)

        
    fig, ax = plt.subplots(ncols=2, sharey=True)
    
    ax[0].plot(wav_A, sp_A_dict["median_spectrum"], label="A")
    ax[0].fill_between(wav_A, sp_A_dict["iqr25_spectrum"], sp_A_dict["iqr75_spectrum"], alpha=0.2, label="IQR A")
    ax[0].set_title('ACS-A DIW Spectra')
    ax[0].set_xlabel('Wavelength [nm]')
    ax[0].set_ylabel('Absorption Coefficient [1/m]')    
    ax[0].legend()
    
    ax[1].plot(wav_S, sp_S_dict["median_spectrum"], label="S")
    ax[1].fill_between(wav_S, sp_S_dict["iqr25_spectrum"], sp_S_dict["iqr75_spectrum"], alpha=0.2, label="IQR S")
    ax[1].set_title('ACS-S DIW Spectra')
    ax[1].set_xlabel('Wavelength [nm]')
    ax[1].set_ylabel('Scattering Coefficient [1/m]')    
    ax[1].legend()
    
    
    plt.tight_layout()
    plt.show()
        
if __name__ == "__main__":
    main()