# SCRIPT TO PLOT INTERACTIVE TIME PLOTS OF EACH VARIABLE

import plotly.express as px
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


# Plot raw data to visualize the time evolution
#and identify potential issues to be fixed in the preprocessing phase (e.g., data imbalance, outliers, etc.) 

variables = [
    #"Timesteps",
    "AverageBurnup",
    "AveragePower",
    "AveragePowerRate",
    #"HoldTime",
    #"IntegratedPower",
    #"AverageRodPressure",
    #"FGR",
    #"InterfaceP_mid", 
    "InterfaceP_end",
    #"GapWidth_mid",
    "GapWidth_end",
    #"CladInnerTemp_mid",
    "CladInnerTemp_end",
    #"CladRadDisp_mid",
    "CladRadDisp_end", 
    #"CladInnerStress_r_mid",
    "CladInnerStress_r_end",
    #"CladInnerStrain_r_mid",
    "CladInnerStrain_r_end",
    #"CladInnerCreep_r_mid",
    #"CladInnerCreep_r_end",
    "RidgeHeight"
]

section_id = input("Choose the section to plot (B for BASE, R for RAMP):\n >> ")
df = df[df["SectionID"] == section_id].reset_index(drop=True)


output_folder = "time_plots"
  

for v in variables:
    fig = px.line(
        df,
        x="AverageBurnup", # change with "Timesteps" if want to plot temporal evolution 
        y=v,
        color="RodID",
        title=f"{v} Evolution (section {section_id})",
        markers=False 
    )
    fig.update_layout(
        hovermode="x unified",
        template="plotly_white",
        legend_title_text="Rod ID"
    )

    # save the html file in the shared filesystem
    output_file = f"{v}_section{section_id}.html"
    os.makedirs(output_folder, exist_ok=True)
    
    full_path = os.path.join(output_folder, output_file)
    
    fig.write_html(full_path)
    print(f"File saved in: {full_path}")


