# %%

from custom_tools.acs_data_reader import get_acs_IOP
from custom_tools.acs_outlier_detection_functions import run_spectra_cleaning_pipeline

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

dir_path = "data/raw/1_acs_runs/runs"

# f = "run_21_ACS.028"

file_list = [ff for ff in os.listdir(dir_path) if ff.startswith('run_21_ACS.')]
fig, ax = plt.subplots()

for f in file_list[:100]:
    
    try:

        df_arr_A, df_arr_C = get_acs_IOP(os.path.join(dir_path, f))

        wav_C = np.array(df_arr_C.columns[1:].astype(float))
        sp_c, fig_C = run_spectra_cleaning_pipeline(df_arr_C, wav_C, threshold_575=0.05, plot=False)

        mean_sp = sp_c[sp_c.columns[1:]].mean()

        sns.regplot(x=np.log(wav_C), y=np.log(mean_sp), ax=ax)

    except:
        pass

plt.show()

# %%
from custom_tools.acs_data_reader import get_acs_IOP
from custom_tools.acs_outlier_detection_functions import run_spectra_cleaning_pipeline

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

dir_path = "data/raw/1_acs_runs/runs"

# f = "run_21_ACS.028"

file_list = [ff for ff in os.listdir(dir_path) if ff.startswith('run_21_ACS.')]
fig, ax = plt.subplots()

for f in file_list:
    
    try:

        df_arr_A, df_arr_C = get_acs_IOP(os.path.join(dir_path, f))

        wav_C = np.array(df_arr_C.columns[1:].astype(float))
        sp_c, fig_C = run_spectra_cleaning_pipeline(df_arr_C, wav_C, threshold_575=0.05, plot=False)

        mean_sp = sp_c[sp_c.columns[1:]].mean()

        slope, intercept = np.polyfit(np.log(wav_C), np.log(mean_sp), 1)
        ax.scatter(slope, intercept, color="black") # This is your [a, b] point

    except:
        pass

ax.set_xlabel('Slope (Exponent $k$)')
ax.set_ylabel('Intercept ($\log a$)')
ax.set_title('Parameter Space Mapping')
ax.grid(True)

plt.show()
# %%
