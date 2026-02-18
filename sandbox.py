# %%
import pandas as pd

pasp_df = pd.read_csv("data/processed/pasp_df.csv", index_col=0)

pasp_df.head()
# %%
# step 1: get two median clearwater spectra
blk_04u_df = pasp_df[pasp_df["sample_name_short"] == "blk_04u"]
blk_14u_df = pasp_df[pasp_df["sample_name_short"] == "blk_14u"]

blk_04u_sp = blk_04u_df.iloc[:, 7:].median()
blk_14u_sp = blk_14u_df.iloc[:, 7:].median()

# %%
# step 2: calculate processed spectra

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
    
    pasp_processed_df.loc[sample] = preprocessed_sp
# %%
import seaborn as sns
for sp in pasp_processed_df.index:
    sns.lineplot(pasp_processed_df.loc[sp, :])