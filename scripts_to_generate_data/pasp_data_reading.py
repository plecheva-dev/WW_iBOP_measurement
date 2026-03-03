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
        pasp_df = pd.DataFrame.from_dict(pa_spectra_data, orient='index', columns=col)

        # re-ordering columns by wavelength
        pasp_df = pasp_df.reindex(sorted(pasp_df.columns), axis=1)
        pasp_df.reset_index(inplace=True)
        pasp_df = pasp_df.rename(columns={"index": "sample_name"})
        pasp_df = PaSp_medatada_df.merge(pasp_df)
    
    pasp_df.to_csv("data/processed/pasp_raw_df.csv")

    ### normalizing spectra with blanks and filtered volume

    blk_04u_df = pasp_df[pasp_df["sample_name_short"] == "blk_04u"]
    blk_14u_df = pasp_df[pasp_df["sample_name_short"] == "blk_14u"]

    blk_04u_sp = blk_04u_df.iloc[:, 7:].median()
    blk_14u_sp = blk_14u_df.iloc[:, 7:].median()

    pasp_processed_df = pd.DataFrame(columns=pasp_df.columns[7:])
    sample_name_list = pasp_df[pasp_df["sampletype"] == "sample"]["sample_name_short"].unique()

    for sample in sample_name_list: 
        sps = pasp_df[pasp_df["sample_name_short"] == sample]
        median_sp = sps.iloc[:, 7:].median()
        volume =  sps["filtered_vol_ml"].values[0]
        
        if sps["filtertype"].values[0] == "04u":
            preprocessed_sp = (median_sp - blk_04u_sp) / volume
        
        elif sps["filtertype"].values[0] == "14u":
            preprocessed_sp = (median_sp - blk_14u_sp) / volume
            
        elif sps["filtertype"].values[0] == "raw":
            preprocessed_sp = (median_sp - blk_04u_sp) / volume # raw means sample filtered at 04u
        
        pasp_processed_df.loc[sample] = preprocessed_sp
        
    pasp_processed_df.to_csv("data/processed/pasp_normalized_df.csv", index_label="sample")

if __name__ == "__main__":
    main()