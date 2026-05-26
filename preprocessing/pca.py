import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

from balance_data import data_balancing

# Load dataset 
df = pd.read_csv(f'/data/foschini/train_dataset.csv', sep=';')

# Feature engineering 
# Create a column for Young's modulus
df['YoungsModulus_end'] = (df['CladInnerStress_r_end']/df['CladInnerStrain_r_end']).fillna(0)

# Create a column for time since gap closure
# Identify where the gap closes
first_closure_idx = df.index[df["GapWidth_end"] == 0].min()

df["TimeSinceClosure"] = 0.0

if pd.notna(first_closure_idx):
    t_start = df.loc[first_closure_idx, "Timesteps"]

    df.loc[first_closure_idx:, "TimeSinceClosure"] = df.loc[first_closure_idx:, "Timesteps"] - t_start


# Balance the simulation data
df = data_balancing(df)


# Create a dataframe for variables of interest only
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
    "TimeSinceClosure", 
    #"CladInnerTemp_mid",
    "CladInnerTemp_end",
    #"CladRadDisp_mid",
    "CladRadDisp_end", 
    #"CladInnerStress_r_mid",
    "CladInnerStress_r_end",
    #"CladInnerStrain_r_mid",
    "CladInnerStrain_r_end",
    #"YoungsModulus_end",
    #"CladInnerCreep_r_mid",
    #"CladInnerCreep_r_end",
    "RidgeHeight"
]

data = df[variables]


# Split per section
base_irr = data[df["SectionID"] == "B"].reset_index(drop=True)
ramp = data[df["SectionID"] == "R"].reset_index(drop=True)


# PCA helper function
def pca_analysis(X):
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=0.96)
    X_pca = pca.fit_transform(X_scaled)
    return pca, X_scaled, X_pca

pca, X_scaled, X_pca = pca_analysis(data)

# --- PCA for both datasets ---
pca_base, X_base_scaled, X_base_pca = pca_analysis(base_irr)
pca_ramp, X_ramp_scaled, X_ramp_pca = pca_analysis(ramp) 


# -------------------------------------------------------
# Create PCA loadings dataframe
# -------------------------------------------------------
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i+1}" for i in range(pca.n_components_)],
    index=variables
)

print("\n===== PCA Loadings (Feature Weights) =====")
print(loadings)


# -----------------------------------------------------------
#  FIGURE 1: PCA Loadings 
# -----------------------------------------------------------

def load_plot(pca, data, section_name):
    
    loadings = pca.components_.T

    plt.figure(figsize=(8,6))
    for i, feature in enumerate(data.columns):
        plt.arrow(0, 0, loadings[i, 0], loadings[i, 1],
                color='r', alpha=0.5, head_width=0.05)
        plt.text(loadings[i, 0]*1.15, loadings[i, 1]*1.15, feature,
                color='g', ha='center', va='center', fontsize=8)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'PCA Loading Plot - {section_name}')
    plt.grid(True)
    plt.axhline(0, color='grey', lw=1)
    plt.axvline(0, color='grey', lw=1)
    plt.savefig(f'{section_name}_pca.png', bbox_inches='tight', dpi=150)
    plt.show()


# --- Loading plot for full dataset ---
#plot = load_plot(pca, data, "FULL DATASET")

# --- Loading plots for both datasets ---
plot_base = load_plot(pca_base, base_irr, "BASE_IRRADIATION")
plot_ramp = load_plot(pca_ramp, ramp, "POWER_RAMP")


# -----------------------------------------------------------
#  FIGURE 2: Correlation Matrices
# -----------------------------------------------------------

#sns.heatmap(data.corr(method='spearman'), cmap='coolwarm', center=0)
#plt.title("Correlation Matrix - FULL DATASET")
#plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.heatmap(base_irr.corr(method='spearman').abs(), cmap='Blues', annot=True, fmt=".2f", vmin=0, vmax=1, center=0, ax=axes[0])
axes[0].set_title("Correlation Matrix - BASE IRRADIATION")

sns.heatmap(ramp.corr(method='spearman').abs(), cmap='Reds', annot=True, fmt=".2f", vmin=0, vmax=1, center=0, ax=axes[1])
axes[1].set_title("Correlation Matrix - POWER RAMP")

plt.tight_layout()
plt.savefig('corr_matrix.png', bbox_inches='tight', dpi=150)
plt.show()



