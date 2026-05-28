# FUNCTION TO CREATE NEW FEATURES 

import pandas as pd

def feat_eng(df):

    #-------------------------------------------
    # Create a column for Young's modulus

    df['YoungsModulus_end'] = (df['CladInnerStress_r_end']/df['CladInnerStrain_r_end']).fillna(0)
    #-------------------------------------------

    #-------------------------------------------
    # Create a column for time since gap closure

    # Identify where the gap closes
    first_closure_idx = df.index[df["GapWidth_end"] == 0].min()

    df["TimeSinceClosure"] = 0.0

    if pd.notna(first_closure_idx):
        t_start = df.loc[first_closure_idx, "Timesteps"]

        df.loc[first_closure_idx:, "TimeSinceClosure"] = df.loc[first_closure_idx:, "Timesteps"] - t_start
    #-------------------------------------------
    
    return df