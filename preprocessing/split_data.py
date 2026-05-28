# FUNCTION TO SPLIT DATA INTO TRAINING AND TEST SETS
# AND TO SELECT FEATURES AND TARGET

import pandas as pd


def data_splitting(df):

    #-------------------------------------------
    # CREATE A TRAIN AND TEST SPLIT
    
    # Each rod is in one set only (to avoid data leakage)
    # Choose the test rods: they must be well-contained within the training data domain
    test_rods = [1]

    all_rods = df['RodID'].unique().tolist()

    train_rods = [rod for rod in all_rods if rod not in test_rods]

    print(f'\nTrain rods: {train_rods}')
    print(f'Test rods: {test_rods}')

    rod_name_map = { # To update if other rods are added in the future 
        1: 'AN1',
        2: 'AN2',
        3: 'AN3',
        4: 'AN4',
        5: 'AN8', 
        6: 'AN10',
        7: 'AN11'
    }   

    # Divide the dataset into training and test sets based on rodID
    train_df = (df[df["RodID"].isin(train_rods)].reset_index(drop=True))
    test_df = df[df["RodID"].isin(test_rods)].reset_index(drop=True) # no sampling --> for plotting
    #-------------------------------------------

    #-------------------------------------------
    # SPLIT FEATURES AND TARGET

    features = [
        #"Timesteps",
        "AverageBurnup",
        "AveragePower",
        #"AveragePowerRate",
        #"HoldTime",
        #"IntegratedPower",
        #"AverageRodPressure",
        "FGR",
        #"InterfaceP_mid", 
        "InterfaceP_end",
        #"GapWidth_mid",
        #"GapWidth_end",
        #"TimeSinceClosure", 
        #"CladInnerTemp_mid",
        #"CladInnerTemp_end",
        #"CladRadDisp_mid",
        "CladRadDisp_end", 
        #"CladInnerStress_r_mid",
        "CladInnerStress_r_end",
        #"CladInnerStrain_r_mid",
        #"CladInnerStrain_r_end",
        #"YoungsModulus_end",
        #"CladInnerCreep_r_mid",
        #"CladInnerCreep_r_end"
    ]

    print('\n=' * 20)
    print(f"Selected features: {features}")
    print('=' * 20)

    target = "RidgeHeight"
    #-------------------------------------------

    #-------------------------------------------
    # INDIVIDUATE TRAINING AND TEST DATA RANGES 

    comparison_data = []

    for i, feat in enumerate(features+[target]):
        train_min = train_df[feat].min()
        train_max = train_df[feat].max()
        test_min = test_df[feat].min()
        test_max = test_df[feat].max()  

        if test_min > train_min or test_max < train_max:
            in_range = 'YES'
        else:
            in_range = 'NO'

        comparison_data.append({
            'Feature': feat,
            'Train Min': train_min,
            'Train Max': train_max,
            'Test Min': test_min,
            'Test Max': test_max,
            'In_Range': in_range,
          })

    df_range_compare = pd.DataFrame(comparison_data)
    print("\n### Range Comparison Training vs Test ###")
    print(df_range_compare.to_string(index=False))
    #-------------------------------------------

    return train_df, test_df, features, target, rod_name_map