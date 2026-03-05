import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

## Utils
def get_spectrum(df, sample_name, preprocess_type, wavelength_cols):
    """Extracts a single spectrum array from a dataframe based on sample and preprocessing."""
    mask = (df["sample"] == sample_name) & (df["preprocessing"] == preprocess_type)
    row = df[mask]
    
    if not row.empty:
        return row[wavelength_cols].values[0]
    return None


def main(output_path):

    acsdata_path = "data/processed/"

    spectra_A_df = pd.read_csv(os.path.join(acsdata_path,"2026-03-04_acs_a_clean_df.csv"), index_col="index")
    sp_cols_A = spectra_A_df.columns[23:]
    wav_A = np.array(sp_cols_A.astype("float"))

    spectra_C_df = pd.read_csv(os.path.join(acsdata_path,"2026-03-04_acs_c_clean_df.csv"), index_col="index")
    sp_cols_C = spectra_C_df.columns[23:]
    wav_C = np.array(sp_cols_C.astype("float"))

    spectra_B_df = pd.read_csv(os.path.join(acsdata_path,"2026-03-04_acs_b_clean_df.csv"), index_col="index")
    sp_cols_B = spectra_B_df.columns[23:]
    wav_B = np.array(sp_cols_B.astype("float"))

    # Helper list to iterate through your sensors easily
    sensors = [
        {"df": spectra_A_df, "wav": wav_A, "cols": sp_cols_A, "label": "A"},
        {"df": spectra_B_df, "wav": wav_B, "cols": sp_cols_B, "label": "B"},
        {"df": spectra_C_df, "wav": wav_C, "cols": sp_cols_C, "label": "C"}
    ]

    # Create the directory (and any missing parent folders)
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)



    # Using your sandbox range
    for sampl in spectra_A_df["sample"].unique():
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharey=True)
        
        for i, preprocess in enumerate(["raw", "14u", "04u"]):
            ax = axes[i]
            ax.set_xlabel("Wavelength [nm]")
            for s in sensors:
                sp = get_spectrum(s["df"], sampl, preprocess, s["cols"])
                if sp is not None:
                    ax.plot(s["wav"], sp, label=s["label"], alpha=0.7)
            
            ax.set_title(f"Processing: {preprocess}")
            if i == 0:
                ax.set_ylabel("Intensity [1/m]")
            if i == 2:
                ax.legend()

        fig.suptitle(f"Sample Comparison: {sampl}", fontsize=14)
        fig.savefig(output_dir / f"{sampl}.png")

        plt.close(fig)
        

if __name__ == "__main__":
    output_path = "data/plots/2026-03-04_acs_sampleplots_A_B_C"
    main(output_path=output_path)
