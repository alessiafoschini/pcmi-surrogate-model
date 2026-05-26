import pandas as pd

def data_balancing(df):
    
    # Start from 11,5 Gwd/t of Burnup --> to focus on the PCMI phase 
    df = df[df['AverageBurnup'] > 11500].reset_index(drop=True) 

    # Balance the dataset
    def trim_timesteps(x):
        section = x.name[1] 

        target_col = "RidgeHeight"
        time_col = "Timesteps"
    
        # remove first and last timesteps of each section to avoid unstable points from simulations
        if section == 'B':
            mask_time = x[time_col] <= 103527200
            std_indices = x[mask_time].iloc[::10].index # skip every 10 points in BI to reduce single events dependency
        elif section == 'R':
            mask_time = x[time_col] >= 103529000
            std_indices = x[mask_time].iloc[:-5].index
        else:
            std_indices = x.index

        # keep the whole indices of fast ridging growth phase 
        mask_rare = (x[target_col] > 5e-8) & (x[target_col] < 2e-6)
        indices_to_keep = x[mask_rare].index

        final_indices = std_indices.union(indices_to_keep)

        return x.loc[final_indices].sort_index()

    group_cols = ['RodID', 'SectionID']

    df = (
        df.groupby(group_cols)
          .apply(trim_timesteps, include_groups=False)
          .reset_index()
    )

    return df