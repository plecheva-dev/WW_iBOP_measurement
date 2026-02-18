
import pandas as pd
import os

def spectro_data_reader(folder_path, sample_name):
    full_path = os.path.join(folder_path, sample_name)
    d = pd.read_csv(full_path, sep = ";", skiprows=1)
    d = d.rename(columns={'nm': "wavelength", 'A': "absorbance"})
    return d

def get_metadata_name_from_filename(filename):
    sample_name = filename[:-11]

    if "autozero" in sample_name:
        date = sample_name[:6]
        sampletype = "autozero"
        filtertype = "none"
        spnr = 1
        sample_name_short = "autozer"
    elif "blk" in sample_name:
        date, _, _, filtertype, spnr = sample_name.split("_")
        spnr = int(spnr[-1])
        sample_name_short = f"blk_{filtertype}"
        sampletype = "blank"
    else: 
        date, name, filtertype, spnr = sample_name.split("_")
        spnr = int(spnr[-1])
        sample_name_short = f"{name}_{filtertype}"
        sampletype = "sample"
        
    metadata_dict = {
            "sample_name": sample_name,
            "sample_name_short": sample_name_short,
            "datestr": date,
            "sampletype": sampletype, 
            "filtertype": filtertype,
            "spnr": spnr
        }
    
    return metadata_dict

def clean_sample_name(s):
    if "blk" in s and "04u" in s: 
        name = "blk_04u"
    elif "blk" in s and "14u" in s: 
        name = "blk_14u"
    elif "zero" in s: 
        name = "zero"
    else:
        name = "sample"
    return name







