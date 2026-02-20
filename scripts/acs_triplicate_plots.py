### loading data

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


a_df = pd.read_csv("data/processed/a_df.csv", index_col=0)
c_df = pd.read_csv("data/processed/c_df.csv", index_col=0)

sp_cols_a = a_df.columns[5:89]
std_cols_a = a_df.columns[89:]

wv_a = sp_cols_a.astype(float)

sp_cols_c = c_df.columns[5:89]
std_cols_c = c_df.columns[89:]

wv_c = sp_cols_c.astype(float)

def main():
    for s in a_df["sample_name"].unique(): 
        # get and plot data, saving the plot to "acs_triplicates_plots/"
        
        fig, ax = plt.subplots(ncols=2, figsize=(12,5))
        a_df_s = a_df[a_df["sample_name"]==s]
        c_df_s = c_df[c_df["sample_name"]==s]
        
        for i, row in a_df_s.iterrows():
            ax[0].plot(wv_a, row[sp_cols_a], label=f'Run {row["run_nr"]}, {row["n_valid_spectra"]} samples')
            lower_bound = row[sp_cols_a].values - row[std_cols_a].values
            upper_bound = row[sp_cols_a].values + row[std_cols_a].values
            ax[0].fill_between(
                wv_a, 
                lower_bound.astype("float"), 
                upper_bound.astype("float"), 
                alpha=0.2)
        for i, row in c_df_s.iterrows():
            ax[1].plot(wv_c, row[sp_cols_c], label=f'Run {row["run_nr"]}, {row["n_valid_spectra"]} samples')
            lower_bound = row[sp_cols_c].values - row[std_cols_c].values
            upper_bound = row[sp_cols_c].values + row[std_cols_c].values    
            ax[1].fill_between(
                wv_c, 
                lower_bound.astype("float"), 
                upper_bound.astype("float"), 
                alpha=0.2)
            
        ax[0].set_title(f'Absorption Coefficient - Sample {s}')
        ax[0].set_xlabel('Wavelength (nm)')
        ax[0].set_ylabel('a (m$^{-1}$)')
        ax[0].legend()
        ax[1].set_title(f'Attenuation Coefficient - Sample {s}')
        ax[1].set_xlabel('Wavelength (nm)')
        ax[1].set_ylabel('c (m$^{-1}$)')
        ax[1].legend()
        plt.tight_layout()
            
        # Define your directory path
        output_dir = Path("data/plots/acs_triplicate_plots")

        # Create the directory (and any missing parent folders)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Now save your plot
        plt.savefig(output_dir / f"{s}.png")
        plt.close()
        
if __name__ == "__main__":
    main()