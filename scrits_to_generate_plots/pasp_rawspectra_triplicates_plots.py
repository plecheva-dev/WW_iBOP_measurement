###
# This script generates, for each filter measured with the PA absorption spectro, a plot with the triplicate measured.
# Spectra are already preprocessed, i.e. blank substracted and divided by the filtered volume. 
# ###


import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path



def main():
    
    path_pasp = "data/processed/pasp_df.csv" 
    pasp_df = pd.read_csv(path_pasp, index_col=0)

    spectral_cols = pasp_df.columns[7:]
    wav = spectral_cols.astype(float)
    short_sample_names = pasp_df["sample_name_short"].unique()

    for sample in short_sample_names:
        sample_replicates = pasp_df[pasp_df["sample_name_short"]==sample]
        
        plt.figure(figsize=(10, 6))
        for idx, row in sample_replicates.iterrows():
            plt.plot(wav, row[7:], label=row["sample_name"])
        
        plt.title(f"Spectra for sample: {sample} (triplicates)")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Absorbance")
        
        # Define your directory path
        output_dir = Path("data/plots/pasp_rawspectra_triplicate_plots")

        # Create the directory (and any missing parent folders)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Now save your plot
        plt.savefig(output_dir / f"{sample}.png")
        
        plt.close()
        
if __name__ == "__main__":
    main()