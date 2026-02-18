import pandas as pd
import numpy as np

def get_acs_IOP(fname_acs):
       
    with open(fname_acs,'r') as file:
        text_raw = [x for x in file.read().split("\n")]
        file.close()


    # read and save the wavelenghts 
    dum_arr = np.array(text_raw[98].split())
    dum_arr_float = []
    for ii in range(1,len(dum_arr)-5):
        dum_arr_float.append(float(dum_arr[ii][1:]))

    C_wl = dum_arr_float[0:85]
    A_wl = dum_arr_float[85:]

    # read the timesteps and the measured calibrated data
    dum_df_arr_C = []
    dum_df_arr_A = []
    dum_time = []
    for ii in np.arange(99,len(text_raw)-1):
        dum_arr = []
        dum_arr = np.array(text_raw[ii].split()).astype('float64')
        #dum_arr = dum_arr[:-8]
        dum_df_arr_C.append(dum_arr[1:86])
        dum_df_arr_A.append(dum_arr[86:-5])
        dum_time.append(dum_arr[0])

    df_arr_A = pd.DataFrame(dum_df_arr_A,columns = A_wl,index = dum_time)
    df_arr_C = pd.DataFrame(dum_df_arr_C,columns = C_wl,index = dum_time)

    df_arr_A = df_arr_A.rename_axis("time(ms)")
    df_arr_C = df_arr_C.rename_axis("time(ms)")
    
    return df_arr_A,df_arr_C