import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm, chisquare
import pandas as pd
import plotly.express as px
import os
import seaborn as sns

from balance_data import data_balancing


# Load dataset 
df = pd.read_csv(f'../pcmi_dataset.csv', sep=';')


# Balance the simulation data 
df = data_balancing(df)


# Divide train and test rods
train_rods = [2, 3, 4, 5, 6, 7] 
test_rods = [1] 

rod_name_map = {
    1: 'AN1',
    2: 'AN2',
    3: 'AN3',
    '1_predicted': 'Predicted AN1',
    4: 'AN4',
    5: 'AN8',
    6: 'AN10',
    7: 'AN11'
}
  

train_df = df[df["RodID"].isin(train_rods)].reset_index(drop=True)
test_df = df[df["RodID"].isin(test_rods)].reset_index(drop=True)



# Load predictions csv file and merge with test dataframe
file_index = input("Insert predictions file suffix\n Possible choices: deep - shallow\n >>")

results_df = pd.read_csv(f"predictions_{file_index}.csv", sep = ";")

complete_test_df = pd.concat([test_df, results_df], axis=1)




# Create lists for plots
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

target = "RidgeHeight"


# Individuate training and test data ranges 
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
print("### Range Comparison Training vs Test ###")
print(df_range_compare.to_string(index=False))



# Create lists
X_test = test_df[features].values
y_test = test_df[target].values

preds = results_df["PredictedRidgeHeight"].values

std_residuals = results_df["StandardizedResidual"].values

# Check if the length is consistent 
print(len(y_test))
print(len(preds))




# ---------------------------------------------------------
# Plot temporal evolution of predictions, actual values and train rods targets 
# ---------------------------------------------------------


#--------------------------------------------------------------------
# FUNCTION TO CREATE SUBSETS FOR PLOTTING
def create_subset(train_set, test_set, target):
    train_subset = train_set[['RodID', 'AverageBurnup', target]].copy()
    test_subset = test_set[['RodID', 'AverageBurnup', target]].copy()
    
    preds_subset = test_set[['RodID', 'AverageBurnup', 'PredictedRidgeHeight']].copy()
    preds_subset['RodID'] = preds_subset['RodID'].astype(str) + "_predicted"
    preds_subset.columns = ['RodID', 'AverageBurnup', target]   
    
    return train_subset, test_subset, preds_subset
#--------------------------------------------------------------------




# CREATE INTERACTIVE PLOTS 
#--------------------------
# Choose section to plot 
section_id = input("Choose the section to plot (B for base, R for ramp): ")

sec_train_df = train_df[train_df["SectionID"] == section_id].reset_index(drop=True)
sec_test_df = complete_test_df[complete_test_df["SectionID"] == section_id].reset_index(drop=True)

output_folder = "time_plots_predictions"

sec_train_subset, sec_test_subset, sec_preds_subset = create_subset(sec_train_df, sec_test_df, target)

for test_rod in test_rods:

    rod_output_folder = os.path.join(output_folder, f"rod{test_rod}")

    current_test = sec_test_subset[sec_test_subset['RodID'] == test_rod]

    pred_label = f"{test_rod}_predicted"
    current_pred = sec_preds_subset[sec_preds_subset['RodID'] == pred_label]
  
    df_ridges_rod = pd.concat([sec_train_subset, current_test, current_pred], axis=0, ignore_index=True)
  
    fig = px.line(
            df_ridges_rod,
            x="AverageBurnup",
            y=target,
            color="RodID",
            title=f"Ridge Height: evolution over Burnup (section {section_id})",
            labels={
            "x": "Burnup [kWd/kgU]",  
            "value": "Ridge Height [m]"
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
    print(f"File saved in: {full_path}")





# CREATE STATIC PLOTS 
#--------------------------

### Fix plot parameters ###
# --- SET FONT & PDF CONFIGURATION ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Liberation Serif", "Bitstream Vera Serif"], # Universal fallbacks
    "pdf.fonttype": 42,          
    "ps.fonttype": 42,
    "font.size": 7.5,            # Lowered base size to offset tight cropping stretch
    "axes.labelsize": 8.0,       # X and Y axis labels
    "xtick.labelsize": 7.5,      # Tick numbers
    "ytick.labelsize": 7.5,      
    "legend.fontsize": 7.5,      # Legend labels
})

# --- MATCH DRAWING WINDOW TO A4 TEXT WIDTH ---
# Choice A: Full page width figure (6.27 inches)
# Choice B: Half page width figure (3.0 inches)
fig_width = 6.27 

fig_height = 6.5 
fig = plt.figure(figsize=(fig_width, fig_height))


train_subset, test_subset, preds_subset = create_subset(train_df, complete_test_df, target)


complete_subset = pd.concat([train_subset, test_subset, preds_subset], axis=0, ignore_index=True)
sec_complete_subset = pd.concat([sec_train_subset, sec_test_subset, sec_preds_subset], axis=0, ignore_index=True)

rods = complete_subset["RodID"].unique()
colors = ['#a9a9a9']*6 + ['#545454', '#000000'] 
edge_colors = ['#a9a9a9']*6 + ['#545454', '#000000'] 
markers = ['x', '*', '^', 's', 'D', 'v', 'o', 'o']
line_styles = ['--']*6 + ['-', '-']
line_width = [0.5]*6 + [1.2, 1.5]
size_marker = [2]*7 + [2]
alpha_values = [0.7]*6 + [1.0]*2
alpha_marker = [to_rgba(colors[i], alpha_values[i]) for i in range(8)]
alpha_edge   = [to_rgba(edge_colors[i], alpha_values[i]) for i in range(8)]

color_dict = {rod: colors[i] for i, rod in enumerate(rods)}
edge_dict    = {rod: edge_colors[i] for i, rod in enumerate(rods)}
marker_dict = {rod: markers[i] for i, rod in enumerate(rods)}
alpha_m_dict = {rod: alpha_marker[i] for i, rod in enumerate(rods)}
alpha_e_dict = {rod: alpha_edge[i] for i, rod in enumerate(rods)}
lines_dict = {rod: line_styles[i] for i, rod in enumerate(rods)}
width_dict = {rod: line_width[i] for i, rod in enumerate(rods)}
size_dict = {rod: size_marker[i] for i, rod in enumerate(rods)}


def create_pred_plots(subset):
  for rod in rods:
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

# sublot1: complete dataset
ax1 = plt.subplot(2, 1, 1)


for spine in ax1.spines.values():
    spine.set_linewidth(0.5)

create_pred_plots(complete_subset)

plt.xlabel('Burnup (GWd/tU)', labelpad=4)
plt.ylabel('Ridge Height ($\mu$m)', labelpad=4)
plt.title('Predictions of test rod AN1', fontsize=8.5, weight='bold', pad=8)
y_ticks = np.arange(0, 6.1, 1.0)      
ax1.set_yticks(y_ticks)
plt.grid(True, alpha=0.4)


# sublot2: ramp dataset
ax2 = plt.subplot(2, 1, 2)

for spine in ax2.spines.values():
    spine.set_linewidth(0.5)

create_pred_plots(sec_complete_subset)

plt.xlabel('Burnup (GWd/tU)', labelpad=4)
plt.ylabel('Ridge Height ($\mu$m)', labelpad=4)
plt.title('Focus on Ramp Section', fontsize=8.5, weight='bold', pad=8)
y_ticks = np.arange(2.5, 6.1, 0.5)      
ax2.set_yticks(y_ticks)
plt.grid(True, alpha=0.4)

#plt.legend(bbox_to_anchor=(1.02, 1.15), loc='upper right', borderaxespad=0., prop={'size': 18}, title="Rods", title_fontsize=18, frameon=True)
plt.legend(
    loc='upper center', 
    bbox_to_anchor=(0.5, -0.25), 
    ncol=4, 
    title_fontsize=10, 
    frameon=True,
    markerscale=2.0
)

plt.tight_layout()
#plt.subplots_adjust(hspace=0.3) 

plt.savefig(f'xgb_predictions_{file_index}.svg', format='svg', bbox_inches='tight')



##---------------------------------------------------------
## 3D PLOTS
#
#def plot_3d_binned(xdata, ydata, xbin, ybin, title, xlabel, ylabel, filename):
#  xdata = np.asarray(xdata).ravel()
#  ydata = np.asarray(ydata).ravel()
#  
#  fig = plt.figure(figsize=(12, 8))
#  ax = fig.add_subplot(111, projection='3d')
#
#  # Create bins
#  xbins = np.arange(xdata.min(), xdata.max() + xbin, xbin)
#  ybins = np.arange(ydata.min(), ydata.max() + ybin, ybin)
#
#  # 2d histogram
#  hist, xedges, yedges = np.histogram2d(xdata, ydata, bins=[xbins, ybins])
#
#  # Create meshgrid for plotting
#  xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
#
#  xpos = xpos.ravel()
#  ypos = ypos.ravel()
#  zpos = 0
#
#  # bar dimensions 
#  dx = xbin * 0.9
#  dy = ybin * 0.9
#  dz = hist.ravel()
#
#  mask = dz > 0
#  xpos, ypos, dz = xpos[mask], ypos[mask], dz[mask]
#
#  # Colored according to density
#  cmap = plt.get_cmap('Blues')
#  max_dz = np.max(dz)
#  colors = cmap(dz / max_dz)
#
#  ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, alpha=0.8)
#
#  ax.set_xlabel(xlabel)
#  ax.set_ylabel(ylabel) 
#  ax.set_zlabel('Frequency')
#  ax.set_title(title)
#
#  plt.savefig(filename, dpi=150)
#  #plt.show()
#  #plt.close()
#
#
## Actual vs Predictions 3D density plot
#plot_3d_binned(y_test, preds, 
#               xbin=1e-7, ybin=1e-7, 
#               title='3D Density: Actual vs Predictions', 
#               xlabel='Actual values', ylabel='Predictions', 
#               filename='3d_actual_vs_preds.png')
#
## Homoscedasticity check
#plot_3d_binned(preds, std_residuals,
#               xbin=1e-7, ybin=0.1,
#               title='3D Density: Homoscedasticity Check', 
#               xlabel='Predicted Values', ylabel='Std Residuals', 
#               filename='3d_residuals_density.png')




## Plot predictions vs actual values
#for i in range(X_test.shape[1]):
#    feature_test = X_test[:, i]
#    plt.figure(figsize=(10, 6))
#    sc = plt.scatter(y_test, preds, c=feature_test, cmap='turbo', alpha=0.6, s=20)
#    plt.colorbar(sc, label=features[i])
#    plt.plot([y_test.min(), y_test.max()], 
#              [y_test.min(), y_test.max()], 
#              'r--', lw=2, label='Perfect coincidence')
#    plt.xlabel('Actual values')
#    plt.ylabel('Predictions')
#    plt.title(f'Predictions vs Actual values (XGBoost)\nRMSE = {rmse:.4e}')
#    plt.legend()
#    plt.grid(True, alpha=0.3)
#    plt.savefig(f'xgb_predictions_colored{i}_AN.png', dpi=150)


# Plot predictions & actual values distributions
plt.figure(figsize=(20, 6))

counts, bin_edges = np.histogram(y_test, bins='auto')
plt.hist(y_test, bins=bin_edges, density=True, alpha=0.7, color='skyblue', edgecolor='black', label = "Actual Values")

plt.hist(preds, bins=bin_edges, density=True, alpha=0.7, color='lightcoral', edgecolor='black', label = "Predictions")

plt.xlabel('Ridge Height [m]')
plt.ylabel('Frequency')
plt.title(f'Predictions vs Actual values distribution')
plt.legend()
plt.grid(True, alpha=0.4)

plt.savefig(f'xgb_pred_distr_{file_index}.png', dpi=150)


##--------------------------------------------------------
# RESIDUALS ANALYSIS
#---------------------------------------------------------
# Plot histogram of residuals (normality check)

fig_height = 3.5

fig = plt.figure(figsize=(fig_width, fig_height))

fig.suptitle(
    'Residuals Analysis: Normality and Homoscedasticity Checks', 
    fontsize=8.5, 
    weight='bold',
    y=0.98          # Adjusts the vertical position so it doesn't crowd the top axis lines
)

ax1 = plt.subplot(1, 2, 1)

ax1_top = ax1.twinx()

for spine in ax1.spines.values():
    spine.set_linewidth(0.5)
for spine in ax1_top.spines.values():
    spine.set_linewidth(0.5)


## individuate and save outliers
lower_bound = -3
upper_bound = 3

#clean_mask = (std_residuals >= lower_bound) & (std_residuals <= upper_bound)
#clean_residuals = std_residuals[clean_mask]

outlier_mask = (std_residuals <= lower_bound) | (std_residuals >= upper_bound)

feature_outliers = pd.DataFrame(X_test[outlier_mask], columns = features).reset_index(drop=True)
target_outliers = pd.Series(y_test[outlier_mask], name = target).reset_index(drop=True)
preds_outliers = pd.Series(preds[outlier_mask], name = 'Predictions').reset_index(drop=True)
residuals_outliers = pd.Series(std_residuals[outlier_mask], name='StandardizedResiduals').reset_index(drop=True)

outliers_df = pd.concat([feature_outliers, target_outliers, preds_outliers, residuals_outliers], axis=1)
outliers_df.to_csv(f"difficult_to_predict_{file_index}.csv", sep = ";", index=False)


# Fit with normal distribution 
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

    print(f"[{label}]")
    print(f"Fit parameters: mu = {mu:.4f}, sigma={sigma:.4f}")
    print(f"Chi-Square: {chi2_stat:.4f}")
    print(f"Reduced Chi-Square: {reduced_chi2:.4f}")
    print(f"P-value: {p_val:.4e}")
 
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



# Plot residuals vs predicted values (homoscedasticity check)
ax2 = plt.subplot(1, 2, 2)

for spine in ax2.spines.values():
    spine.set_linewidth(0.5)

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

#plt.savefig(f'xgb_residuals_{file_index}.png', dpi=150)
plt.savefig(f'xgb_residuals_{file_index}.svg', format='svg', bbox_inches='tight')