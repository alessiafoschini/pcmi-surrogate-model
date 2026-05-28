# SCRIPT TO PLOT DATA DISTRIBUTIONS 

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys

# Import the customized scripts
from balance_data import data_balancing
from split_data import data_splitting
from create_feat import feat_eng

# Individuate parent directory 
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add parent dir into Python memory
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Path to reach the dataset directory 
data_path = os.path.join(parent_dir, "data", "pcmi_dataset.csv")



# LOAD DATASET 
df = pd.read_csv(data_path, sep=';')


## FEATURE ENGINEERING 
#df = feat_eng(df)

#------------------------------------
# BALANCE THE SIMULATION DATA 

# flag to decide whether to balance the dataset or not 
# (to compare the distributions before and after balancing)
balance = False

if balance:
    df = data_balancing(df)
#------------------------------------


# SPLIT DATA INTO TRAINING AND TEST SETS
# AND GET FEATURES AND TARGET LISTS
train_df, test_df, features, target, _ = data_splitting(df)


# CREATE SUBFOLDER TO SAVE PLOTS
output_folder = "balanced_distr_plots" if balance else "raw_distr_plots"
os.makedirs(output_folder, exist_ok=True)



for v in features+[target]:
    
    sets = {
    "train": {"data": train_df[v].values, "color": "blue", "label": "Train"},
    "test": {"data": test_df[v].values, "color": "red", "label": "Test"}
    }

    plt.figure(figsize=(8, 5))

    # determine common number of bins 
    N = min(len(sets["train"]["data"]), len(sets["test"]["data"]))
    bins = int(np.sqrt(N))

    # Find the absolute global min and max
    global_min = min(sets["train"]["data"].min(), sets["test"]["data"].min())
    global_max = max(sets["train"]["data"].max(), sets["test"]["data"].max())

    for set_name, set_info in sets.items():

        plt.hist(set_info["data"], bins=bins, range=(global_min, global_max), color=set_info["color"], alpha=0.5, label=set_info["label"], density=True)

    plt.title(f"Distribution of {v}")
    plt.xlabel(v)
    plt.ylabel("Frequency")
    plt.legend()

    plt.tight_layout()


    plt.savefig(os.path.join(output_folder, f"hist_{v}.png"))
    plt.close()