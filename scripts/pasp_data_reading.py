import pandas as pd
import os
from custom_tools.pasp_data_reader import get_metadata_name_from_filename
from custom_tools.pasp_data_reader import spectro_data_reader

def main():
    
    folder_path = "data/raw/2_pa_spectra/pa_spectra"
    
    metadata_vol_df = pd.read_csv("data/raw/2_pa_spectra/metadata_pa_spectra.csv")
    
    pa_spectra_data = {}
    PaSp_medatada_df = pd.DataFrame(columns=[
            "sample_name",
            "sample_name_short",
            "datestr",
            "sampletype", 
            "filtertype",
            "spnr", 
            "filtered_vol_ml"
    ])
    
    i_meta = 0

    for file in os.listdir(folder_path):

        metadata_dict = get_metadata_name_from_filename(file)
        
        sample_name_short = metadata_dict["sample_name_short"]
            
        if sample_name_short in metadata_vol_df["sample_name"].values:
            row = metadata_vol_df[metadata_vol_df["sample_name"]==sample_name_short]
            metadata_dict["filtered_vol_ml"] = row["filtered_volume_ml"].values[0]
        else:
            metadata_dict["filtered_vol_ml"] = 0
        
        PaSp_medatada_df.loc[i_meta] = metadata_dict.values()
        i_meta += 1
        
        sample_name = metadata_dict["sample_name"]
        sp_df = spectro_data_reader(folder_path, file)
        pa_spectra_data[sample_name] = sp_df["absorbance"].values

        
    col = sp_df["wavelength"].values
    PaSp_df = pd.DataFrame.from_dict(pa_spectra_data, orient='index', columns=col)

    # re-ordering columns by wavelength
    PaSp_df = PaSp_df.reindex(sorted(PaSp_df.columns), axis=1)
    PaSp_df.reset_index(inplace=True)
    PaSp_df = PaSp_df.rename(columns={"index": "sample_name"})
    PaSp_df = PaSp_medatada_df.merge(PaSp_df)
    
    PaSp_df.to_csv("data/processed/pasp_df.csv")

if __name__ == "__main__":
    main()