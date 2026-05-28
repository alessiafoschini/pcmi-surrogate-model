# SCRIPT TO TEST THE MODEL AND ANALYZE PREDICTIONS

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import joblib
import os
import sys
import xgboost as xgb

from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.colors import to_rgba
from scipy.stats import norm, chisquare

#-------------------------------------------
# Individuate parent directory 
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add parent dir into Python memory
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
#-------------------------------------------

# Import the customized scripts 
from preprocessing.balance_data import data_balancing
from preprocessing.split_data import data_splitting 
from preprocessing.create_feat import feat_eng




# Paths to reach the files 
data_path = os.path.join(parent_dir, "data", "pcmi_dataset.csv")
model_path = os.path.join(parent_dir, "xgb_model.json")
scaler_path = os.path.join(parent_dir, "target_scaler.pkl")

#-----------------------------------------
# LOAD DATASET 
df = pd.read_csv(data_path, sep=';')


## FEATURE ENGINEERING 
#df = feat_eng(df)


# BALANCE THE SIMULATION DATA 
df = data_balancing(df)


# SPLIT DATA INTO TRAINING AND TEST SETS
# AND GET FEATURES AND TARGET LISTS
train_df, test_df, features, target, rod_name_map = data_splitting(df)
#-----------------------------------------



# GET TRAIN AND TEST RODS
train_rods = train_df['RodID'].unique().tolist()
test_rods = test_df['RodID'].unique().tolist()

# CREATE LISTS
X_test = test_df[features].values
y_test = test_df[target].values



#-----------------------------------------
# TEST THE OPTIMAL MODEL

print("\nLoading model and target scaler...")

model = xgb.XGBRegressor()
model.load_model(model_path)

scaler = joblib.load(scaler_path)


print("\nMaking predictions on test set...")

preds_scaled = model.predict(X_test).flatten()
preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()


# PRINT EVALUATION METRICS
rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f"\nFinal RMSE on the test set: {rmse}")

r2 = r2_score(y_test, preds)
print(f"R² Score on the test set: {r2}")


# COMPUTE STANDARDIZED RESIDUALS 
std_residuals = (preds - y_test) / rmse 


# EXPAND TEST DATAFRAME
preds_df = pd.Series(preds, name = "PredictedRidgeHeight").reset_index(drop=True)
res_df = pd.Series(std_residuals, name='StandardizedResidual').reset_index(drop=True)

complete_test_df = pd.concat([test_df, preds_df, res_df], axis=1)
#----------------------------------------






##--------------------------------------------------------
# PREDICTIONS PLOTS
##--------------------------------------------------------


#-------------------------------------------
# FUNCTION TO CREATE SUBSETS FOR PLOTTING

def create_subset(train_set, test_set, target):
    train_subset = train_set[['RodID', 'AverageBurnup', target]].copy()
    test_subset = test_set[['RodID', 'AverageBurnup', target]].copy()
    
    preds_subset = test_set[['RodID', 'AverageBurnup', 'PredictedRidgeHeight']].copy()
    preds_subset['RodID'] = preds_subset['RodID'].astype(str) + "_predicted"
    preds_subset.columns = ['RodID', 'AverageBurnup', target]   
    
    return train_subset, test_subset, preds_subset
#-------------------------------------------


#-------------------------------------------
# CREATE INTERACTIVE PLOTS

# Choose section to plot 
section_id = input("\nChoose the section to plot (B for BASE, R for RAMP):\n >> ")

sec_train_df = train_df[train_df["SectionID"] == section_id].reset_index(drop=True)
sec_test_df = complete_test_df[complete_test_df["SectionID"] == section_id].reset_index(drop=True)

output_folder = "plots_predictions"

sec_train_subset, sec_test_subset, sec_preds_subset = create_subset(sec_train_df, sec_test_df, target)


for test_rod in test_rods:

    pred_label = f"{test_rod}_predicted"

    rod_output_folder = os.path.join(output_folder, f"rod{test_rod}")

    current_test = sec_test_subset[sec_test_subset['RodID'] == test_rod]

    current_pred = sec_preds_subset[sec_preds_subset['RodID'] == pred_label]
  
    df_ridges_rod = pd.concat([sec_train_subset, current_test, current_pred], axis=0, ignore_index=True)
  
    fig = px.line(
            df_ridges_rod,
            x="AverageBurnup",
            y=target,
            color="RodID",
            title=f"Ridge Height: evolution over Burnup (section {section_id})",
            labels={
            "AverageBurnup": "Burnup [kWd/kgU]",  
            target: "Ridge Height [m]"
            },
            markers=True
    )
    fig.update_layout(
            hovermode="x unified",
            template="plotly_white",
            legend_title_text="Rod ID"
    )

    for trace in fig.data:
        rod_id = trace.name
    
        if str(rod_id) in [str(r) for r in train_rods]:
            trace.line.dash = 'dash'

        elif str(rod_id) == str(test_rod):
            trace.line.color = 'blue'
            trace.line.width = 3
            trace.line.dash = 'solid'

        elif "predicted" in str(rod_id):
            trace.line.color = 'dodgerblue'
            trace.line.dash = 'solid'
            trace.line.width = 3

    # save the html file in the shared filesystem
    output_file = f"RidgeHeight_section{section_id}_{file_index}.html"
    os.makedirs(rod_output_folder, exist_ok=True)
    full_path = os.path.join(rod_output_folder, output_file)
    fig.write_html(full_path)
    print(f"\nInteractive plot saved in: {full_path}")
#-------------------------------------------



#-------------------------------------------
# CREATE STATIC PLOTS 

plot_train = True # change to False if plotting all training rods is not feasible

train_subset, test_subset, preds_subset = create_subset(train_df, complete_test_df, target)

ramp_train_df = train_df[train_df["SectionID"] == 'R'].reset_index(drop=True)
ramp_test_df = complete_test_df[complete_test_df["SectionID"] == 'R'].reset_index(drop=True)

ramp_train_subset, ramp_test_subset, ramp_preds_subset = create_subset(ramp_train_df, ramp_test_df, target)

# Subsets to concat
subsets_to_concat = []
ramp_subsets_to_concat = []

if plot_train:
    subsets_to_concat.append(train_subset)
    ramp_subsets_to_concat.append(ramp_train_subset)

subsets_to_concat.extend([test_subset, preds_subset])
ramp_subsets_to_concat.extend([ramp_test_subset, ramp_preds_subset])

complete_subset = pd.concat(subsets_to_concat, axis=0, ignore_index=True)
ramp_subset = pd.concat(ramp_subsets_to_concat, axis=0, ignore_index=True)

# Initialize styling dictionaries
color_dict   = {}
marker_dict  = {}
alpha_m_dict = {}
alpha_e_dict = {}
lines_dict   = {}
width_dict   = {}
size_dict    = {}



train_markers = ['x', '*', '^', 'v', '<', '>', 'p', 'h']
test_markers = ['o', 's', 'D', 'P', 'X', 'H'] 


# Assign styles 
if plot_train:
    for idx, rod in enumerate(train_rods):
        
        current_marker = train_markers[idx % len(train_markers)]
        
        color_dict[rod]   = '#a9a9a9'  # light gray
        marker_dict[rod]  = current_marker
        lines_dict[rod]   = '--'       
        width_dict[rod]   = 0.5
        size_dict[rod]    = 2
        alpha_m_dict[rod] = to_rgba('#a9a9a9', 0.7)
        alpha_e_dict[rod] = to_rgba('#a9a9a9', 0.7)
        


for idx, rod in enumerate(test_rods):
    
    pred_rod = f"{rod}_predicted"

    # Update rod_name_map
    name_test_rod = rod_name_map[rod]
    rod_name_map[pred_rod] = f'Predicted {name_test_rod}'
    
    current_marker = test_markers[idx % len(test_markers)]
    
    # For actual values 
    color_dict[rod]   = '#545454'  # dark gray
    marker_dict[rod]  = current_marker
    lines_dict[rod]   = '-'
    width_dict[rod]   = 1.2
    size_dict[rod]    = 4
    alpha_m_dict[rod] = to_rgba('#545454', 1.0)
    alpha_e_dict[rod] = to_rgba('#545454', 1.0)
    
    # For predictions
    color_dict[pred_rod]   = '#000000'  # black
    marker_dict[pred_rod]  = current_marker 
    lines_dict[pred_rod]   = '-'
    width_dict[pred_rod]   = 1.5
    size_dict[pred_rod]    = 4
    alpha_m_dict[pred_rod] = to_rgba('#000000', 1.0)
    alpha_e_dict[pred_rod] = to_rgba('#000000', 1.0)


complete_rods = complete_subset["RodID"].unique().tolist()


def create_pred_plots(subset):
    for rod in complete_rods:
        rod_subset = subset[subset["RodID"] == rod]
        x = rod_subset["AverageBurnup"].values * 1e-3
        y = rod_subset[target].values * 1e6

        rod_name = rod_name_map.get(rod)

        plt.plot(x, y, 
                color = color_dict[rod],
                marker = marker_dict[rod],
                linestyle = lines_dict[rod],
                linewidth = width_dict[rod],
                markersize=size_dict[rod],
                markerfacecolor=alpha_m_dict[rod], 
                markeredgecolor=alpha_e_dict[rod],
                label = f'{rod_name}')


plt.figure(figsize=(13, 10))


# sublot1: complete dataset
ax1 = plt.subplot(2, 1, 1)

create_pred_plots(complete_subset)

plt.xlabel('Burnup (GWd/tU)', labelpad=4)
plt.ylabel('Ridge Height ($\mu$m)', labelpad=4)
plt.title('Predictions of Test Set', weight='bold', pad=8)
y_ticks = np.arange(0, 6.1, 1.0)      
ax1.set_yticks(y_ticks)
plt.grid(True, alpha=0.4)


# sublot2: ramp dataset
ax2 = plt.subplot(2, 1, 2)

create_pred_plots(ramp_subset)

plt.xlabel('Burnup (GWd/tU)', labelpad=4)
plt.ylabel('Ridge Height ($\mu$m)', labelpad=4)
plt.title('Focus on Ramp Section', weight='bold', pad=8)
y_ticks = np.arange(2.5, 6.1, 0.5)      
ax2.set_yticks(y_ticks)
plt.grid(True, alpha=0.4)


plt.legend(
    loc='upper center', 
    bbox_to_anchor=(0.5, -0.25), 
    ncol=4, 
    title_fontsize=10, 
    frameon=True,
    markerscale=2.0
)

plt.tight_layout()

plt.savefig(f'predictions_{file_index}.svg', format='svg', bbox_inches='tight')
#-------------------------------------------



#-------------------------------------------
# PLOT PREDICTIONS VS ACTUAL VALUES

output_folder = "predictions_scatter_plots"
os.makedirs(output_folder, exist_ok=True)

for i in range(X_test.shape[1]):
    feature_test = X_test[:, i]
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(y_test, preds, c=feature_test, cmap='turbo', alpha=0.6, s=20)
    plt.colorbar(sc, label=features[i])
    plt.plot([y_test.min(), y_test.max()], 
              [y_test.min(), y_test.max()], 
              'r--', lw=2, label='Perfect coincidence')
    plt.xlabel('Actual Values')
    plt.ylabel('Predictions')
    plt.title(f'Predictions vs Actual Values\n RMSE = {rmse:.4e}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    file_path = os.path.join(output_folder, f'pred_scatter_{features[i]}_{file_index}.png')
    plt.savefig(file_path, dpi=150)
#-------------------------------------------



#-------------------------------------------
# PLOT PREDICTIONS & ACTUAL VALUES DISTRIBUTIONS

plt.figure(figsize=(10, 6))

counts, bin_edges = np.histogram(y_test, bins='auto')
plt.hist(y_test, bins=bin_edges, density=True, alpha=0.7, color='skyblue', edgecolor='black', label = "Actual Values")

plt.hist(preds, bins=bin_edges, density=True, alpha=0.7, color='lightcoral', edgecolor='black', label = "Predictions")

plt.xlabel('Ridge Height [m]')
plt.ylabel('Frequency')
plt.title(f'Predictions vs Actual Values Distribution')
plt.legend()
plt.grid(True, alpha=0.4)

plt.savefig(f'pred_distr_{file_index}.png', dpi=150)
#-------------------------------------------



##--------------------------------------------------------
# RESIDUALS ANALYSIS
##--------------------------------------------------------

#-------------------------------------------
# PLOT HISTOGRAM OF RESIDUALS (NORMALITY CHECK)

fig = plt.figure(figsize=(10, 8))

fig.suptitle(
    f'Residuals Analysis: Normality and Homoscedasticity Checks\n RMSE = {rmse:.4e}', 
    fontsize=8.5, 
    weight='bold',
    y=0.98          # Adjusts the vertical position so it doesn't crowd the top axis lines
)

ax1 = plt.subplot(1, 2, 1)

ax1_top = ax1.twinx()


## INDIVIDUATE AND SAVE OUTLIERS
lower_bound = -3
upper_bound = 3


outlier_mask = (std_residuals <= lower_bound) | (std_residuals >= upper_bound)

feature_outliers = pd.DataFrame(X_test[outlier_mask], columns = features).reset_index(drop=True)
target_outliers = pd.Series(y_test[outlier_mask], name = target).reset_index(drop=True)
preds_outliers = pd.Series(preds[outlier_mask], name = 'Predictions').reset_index(drop=True)
residuals_outliers = pd.Series(std_residuals[outlier_mask], name='StandardizedResiduals').reset_index(drop=True)

outliers_df = pd.concat([feature_outliers, target_outliers, preds_outliers, residuals_outliers], axis=1)
outliers_df.to_csv(f"difficult_to_predict_{file_index}.csv", sep = ";", index=False)


# FIT WITH NORMAL DISTRIBUTION 
sections = {
    "B": {"data": complete_test_df[complete_test_df["SectionID"] == "B"]["StandardizedResidual"].values, "color": "limegreen", "label": "BI", "linestyle": "--", "linecolor": "green", "ax": ax1},
    "R": {"data": complete_test_df[complete_test_df["SectionID"] == "R"]["StandardizedResidual"].values, "color": "salmon", "label": "Ramp", "linestyle": "-", "linecolor": "red", "ax": ax1_top}
}


for sect_id, info in sections.items():
  data = info["data"]
  color = info["color"]
  label = info["label"]
  linestyle = info["linestyle"]
  linecolor = info["linecolor"]
  target_ax = info["ax"]
  
  (mu, sigma) = norm.fit(data)

  # plot hist
  bins = int(np.sqrt(len(data)))
  observed, bin_edges, _ = target_ax.hist(data, bins=bins, density=True, alpha=0.5, 
                                    color=color, edgecolor='black', label=f"Observed {label}")

  # plot curve normal dist
  x = np.linspace(-3, 3, 100)
  y = norm.pdf(x, mu, sigma)
  target_ax.plot(x, y, linewidth=1.5, linestyle=linestyle, color=linecolor, label=f"Normal Fit\n ($\mu$={mu:.2f}, $\sigma$={sigma:.2f})")

  # goodness-of-fit test
  observed_counts, _ = np.histogram(data, bins=bin_edges)
  cdf_values = norm.cdf(bin_edges, loc=mu, scale=sigma)
  bin_probs = np.diff(cdf_values)
  expected_counts = bin_probs * len(data)

  # validity filter 
  mask = expected_counts >= 5
  if mask.any():
    f_obs = observed_counts[mask]
    f_exp = expected_counts[mask]
    f_exp = f_exp * (f_obs.sum() / f_exp.sum()) # area obs = area exp 

    n_total_bins = len(expected_counts)
    n_kept_bins = len(f_exp)

    print(f"[{label}] Chi-Square Stability Check: Kept {n_kept_bins} out of {n_total_bins} bins.")
    chi2_stat, p_val = chisquare(f_obs=f_obs, f_exp=f_exp, ddof=2)

    dof = len(f_obs) - 1 - 2 
    reduced_chi2 = chi2_stat / dof if dof > 0 else 0

    print(f"\n[{label}]")
    print(f"Fit parameters: mu = {mu:.4f}, sigma={sigma:.4f}")
    print(f"Chi-Square: {chi2_stat:.4f}")
    print(f"Reduced Chi-Square: {reduced_chi2:.4f}")
    print(f"P-value: {p_val:.4e}\n")
 
ax1.set_ylabel('Density', labelpad=4)
ax1.set_ylim(0, 1.2)

ax1_top.set_ylim(-0.6, 0.6) # top half starts from the middle

ax1.set_yticks([0, 0.2, 0.4, 0.6])
ax1_top.set_yticks([0, 0.2, 0.4, 0.6])

ax1.grid(True, which='major', axis='both', alpha=0.4)
ax1_top.grid(True, which='major', axis='y', alpha=0.4)


ax1.set_xlabel('Std Residuals', labelpad=4)
x_ticks = np.arange(-4, 4.1, 1)      
ax1.set_xticks(x_ticks)
ax1.set_xlim(-4, 4)

handles_bottom, labels_bottom = ax1.get_legend_handles_labels()
handles_top, labels_top = ax1_top.get_legend_handles_labels()

all_handles = handles_bottom + handles_top
all_labels = labels_bottom + labels_top

ax1.legend(
    all_handles, all_labels,
    loc='upper center', 
    bbox_to_anchor=(0.5, -0.35), 
    ncol=2,
    title_fontsize=8, 
    frameon=True
)
#-------------------------------------------


# PLOT RESIDUALS VS PREDICTED VALUES (HOMOSCEDASTICITY CHECK)
#-------------------------------------------

ax2 = plt.subplot(1, 2, 2)


# separate test rods with markers and sections with colors 
marker_dict = {'B': 'o', 'R': '^'}
color_dict = {'B': 'green', 'R': 'red'}
section_labels = {'B': 'Base', 'R': 'Ramp'}


for section in ['B', 'R']:

  mask = (complete_test_df["RodID"] == 1) & (complete_test_df["SectionID"] == section)
  subset = complete_test_df[mask]

  preds = subset["PredictedRidgeHeight"].values * 1e6
  res = subset["StandardizedResidual"].values

  plt.scatter(
              preds, 
              res, 
              alpha=0.6, 
              color=color_dict[section], 
              marker=marker_dict[section], 
              s=8, 
              label=f'{section_labels[section]} Section'
          )


plt.axhline(0, color='black', linestyle='--', lw=1.5)
plt.xlabel('Predicted Ridge Heights ($\mu m$)', labelpad=4)
plt.ylabel('Std Residuals', labelpad=4)

y_ticks = np.arange(-5, 5.1, 1)      
ax2.set_yticks(y_ticks)

plt.legend(
    loc='upper center', 
    bbox_to_anchor=(0.5, -0.35), 
    ncol=2,
    title_fontsize=8.5, 
    frameon=True
)

plt.grid(True, alpha=0.4)

plt.tight_layout()

plt.savefig(f'residuals_{file_index}.svg', format='svg', bbox_inches='tight')