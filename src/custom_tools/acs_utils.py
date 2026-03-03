import pandas as pd


##### variables and functions

def concatenate_df(df_cumm, df):
    # Check if df_cumm is None or empty
    if df_cumm is None or df_cumm.empty:
        df_cumm = df.copy()
    else:
        df_cumm = pd.concat([df_cumm, df], ignore_index=True)
    
    return df_cumm

def _convert_run_list_to_string(run_list):
    for (i, r) in enumerate(run_list):
        if r<10:
            run_list[i] = f"00{r}"
        elif r<100:
            run_list[i] = f"0{r}"
        else:
            run_list[i] = str(r)
    # Return as a list
    return run_list

def get_runs_list_from_samplename_and_metadata(samplename, metadata):
    # Filter rows where sample_name matches, then grab the run_nr column
    runs = metadata.loc[metadata["sample_name"] == samplename, "run_nr"].tolist()
    runs = _convert_run_list_to_string(runs)
    return runs


def handle_exception_when_run_not_working(run, e):
    print(f"FAILED FOR RUN: {run}")
    print(f"ERROR TYPE: {type(e).__name__}")
    print(f"ERROR MESSAGE: {e}")
    
def get_sp_df_stats(clean_sp_df):
    n_sp = clean_sp_df.shape[0]
    median_sp = clean_sp_df.median(axis=0)
    p25 = clean_sp_df.quantile(0.25, axis=0)
    p75 = clean_sp_df.quantile(0.75, axis=0)
    return {
        "n_spectra":n_sp, 
        "median_spectrum": median_sp, 
        "iqr25_spectrum": p25, 
        "iqr75_spectrum": p75
        }