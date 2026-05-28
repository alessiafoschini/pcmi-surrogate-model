# SCRIPT TO PLOT DATA DISTRIBUTIONS 

import pandas as pd
import numpy as np
import os
import sys

# Individuate parent directory 
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add parent dir into Python memory
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Path to reach the dataset directory 
data_path = os.path.join(parent_dir, "data", "pcmi_dataset.csv")

# Load dataset 
df = pd.read_csv(data_path, sep=';')

# Split data into training and test sets
# and get features and target lists
train_df, test_df, features, target, rod_name_map = data_splitting(df)

