# MAIN SCRIPT TO BUILD THE SURROGATE MODEL

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
import time
import os

from xgboost import XGBRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import RandomizedSearchCV, LeaveOneGroupOut
from sklearn.metrics import mean_squared_error, r2_score

# Import the customized scripts 
from balance_data import data_balancing
from split_data import data_splitting 


# Individuate current dir 
current_dir = os.path.dirname(os.path.abspath(__file__))

# Path to reach the dataset directory 
data_path = os.path.join(current_dir, "data", "pcmi_dataset.csv")


# Dictionary to memorize execution times
execution_times = {}



# =======================
# 1. Data Processing 
# =======================


# LOAD DATASET 
df = pd.read_csv(data_path, sep=';')


#-------------------------------------------
## FEATURE ENGINEERING 

## Create a column for Young's modulus
#df['YoungsModulus_end'] = (df['CladInnerStress_r_end']/df['CladInnerStrain_r_end']).fillna(0)
#
## Create a column for time since gap closure
## Identify where the gap closes
#first_closure_idx = df.index[df["GapWidth_end"] == 0].min()
#
#df["TimeSinceClosure"] = 0.0
#
#if pd.notna(first_closure_idx):
#    t_start = df.loc[first_closure_idx, "Timesteps"]
#
#    df.loc[first_closure_idx:, "TimeSinceClosure"] = df.loc[first_closure_idx:, "Timesteps"] - t_start
#-------------------------------------------


#-------------------------------------------
# BALANCE THE SIMULATION DATA

# Create a model starting from 11,5 GWd/tU --> to focus on the PCMI phase 
#and remove points from the most represented phase (BASE IRRADIATION) to increase Ramp importance

df = data_balancing(df)
#-------------------------------------------


#-------------------------------------------
# CREATE A TRAIN AND TEST SPLIT

print("----- Dataset splitting -----")
print("\nStarting train-test splitting...")


# Split data into training and test sets
# and get features and target lists
train_df, test_df, features, target, rod_name_map = data_splitting(df)
#-------------------------------------------


#-------------------------------------------
# APPLY SAMPLE WEIGHTS TO INCREASE RAMP IMPORTANCE IN TRAINING 

weight_map = {'R': 30.0, 'B': 1.0}
sec_weights = train_df['SectionID'].map(weight_map).values

# weight increases with FGR accumulation
p = train_df['FGR'].values

p_norm = (p - p.min()) / (p.max() - p.min())
prop_weights = 1.0 + 0.5 * p_norm

weights = sec_weights * prop_weights 
#-------------------------------------------


#-------------------------------------------
# CREATE LISTS FOR FEATURES AND TARGET
X_train = train_df[features].values
y_train = train_df[target].values

X_test = test_df[features].values
y_test = test_df[target].values

print(f"\nTraining set dimension: {y_train.shape}")
print(f"Test set dimension: {y_test.shape}")
#-------------------------------------------


#-------------------------------------------
# SCALE TARGET

# Log transform
#log_y_train = np.log(y_train)

# Robust scaling
scaler = RobustScaler()
y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

median_val = scaler.center_[0]
iqr_val = scaler.scale_[0]

print(f"Median Scaler: {median_val}")
print(f"Interquartile Range (IQR) Scaler: {iqr_val}")
#-------------------------------------------




# =============================
# 2. Hyperparameters Tuning
# =============================

print('\n')
print("\n----- Hypertuning -----")
print("\nStarting Hyperparameter Tuning with RandomizedSearchCV...")
print("\nCV strategy: LOGO (GROUP=ROD)")
 

# CREATE THE XGBOOST MODEL
reg = XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1) # n_jobs=-1 to use all the available CPU cores


#-------------------------------------------
# SET HYPERTUNING

# Parameter Grids
model_arch = input("Choose the model architecture (D for DEEP, S for SHALLOW):\n >> ")
print('\n')

if model_arch == "D":
#
param_grid = {
    'n_estimators': [400, 600, 800, 1000],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.5, 0.7],
    'colsample_bytree': [0.5, 0.7],
    'min_child_weight': [0.1, 0.5, 1.0],
    'gamma': [0, 1e-4, 1e-2, 1],
    'reg_alpha': [0, 1, 10],    # L1 regularization
    'reg_lambda': [1, 10, 100]   # L2 regularization
}
  
elif model_arch == "S":
param_grid = { 
    'n_estimators': [100, 200], 
    'max_depth': [2, 3, 4], 
    'learning_rate': [0.1, 0.3], 
    'subsample': [0.5, 0.7], 
    'colsample_bytree': [0.5, 0.7], 
    'min_child_weight': [0.1, 0.5], 
    'gamma': [0, 1e-6], 
    'reg_alpha': [0, 1],
    'reg_lambda': [1, 10] 
}


# Define CV strategy --> LOGO since there are few rods (=folds) 
#and want to find the parameters leading to the highest score averaged across rods
cv = LeaveOneGroupOut() 


search = RandomizedSearchCV(
    estimator=reg, 
    param_distributions=param_grid,
    n_iter=30,
    scoring='neg_mean_squared_error', 
    cv=cv,
    n_jobs=-1,
    verbose=1,
    random_state=42
)
#-------------------------------------------


#-------------------------------------------
# FIT THE TUNER

groups_train = train_df["RodID"].values

# Fit the model to find the best parameters
start_tuning = time.time()

search.fit(X_train, y_train_scaled, groups=groups_train, sample_weight=weights)

execution_times['Hypertuning'] = (time.time() - start_tuning) / 60

best_params = search.best_params_
print(f"\nBest parameters found:")
for k, v in best_params.items():
    print(f"  {k}: {v}")
print(f"\nBest CV negative MSE: {search.best_score_:.4f}") # a value closer to 0 indicates a better performance 
#-------------------------------------------




# =============================
# 3. Model Validation
# =============================
print('\n')
print("\n----- Model Validation -----")
print("\nStarting Cross Validation (LOGO)...")


## FIXED PARAMETERS 
#best_params = {
#    'subsample': 0.7, 
#    'reg_lambda': 100, 
#    'reg_alpha': 0, 
#    'n_estimators': 1000, 
#    'min_child_weight': 0.5, 
#    'max_depth': 3, 
#    'learning_rate': 0.1, 
#    'gamma': 0.0001, 
#    'colsample_bytree': 0.7
#}

#-------------------------------------------
# RELOAD OPTIMAL PARAMETERS FOR XGBOOST NATIVE API
# XGBoost native API is used for tracking losses in real time

if 'n_estimators' in best_params:

    num_boost_round = best_params.get("n_estimators", 400)
    del best_params['n_estimators']


params_for_train = {
    **best_params,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "n_jobs": -1
}

print('\nCheck if the best parameters are correcly loaded:')
print(params_for_train)
print('\n')
#-------------------------------------------


#-------------------------------------------
# CROSS-VALIDATION LOOP

n_groups = len(np.unique(groups_train)) 

# Matrices to store losses for all folds
train_loss_matrix = np.zeros((n_groups, num_boost_round))
val_loss_matrix = np.zeros((n_groups, num_boost_round))

# List of dictionaries to join rodID and history 
fold_results = []


start_cv = time.time()

for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train_scaled, groups=groups_train)):
    
    print(f"\nProcessing fold {fold_idx + 1}/{n_groups}...")

    # Get Validation RodID for debugging 
    val_rod_id = groups_train[val_idx][0]
    real_rod_name = rod_name_map.get(val_rod_id)
    print(f"\nValidation RodID: {real_rod_name}") 

    # Split data in training and validation sets 
    X_tr, X_v = X_train[train_idx], X_train[val_idx]
    y_tr, y_v = y_train_scaled[train_idx], y_train_scaled[val_idx]

    w_tr = weights[train_idx]

    # Convert dataset into DMatrix (XGBoost API)
    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=features, weight=w_tr)
    dval   = xgb.DMatrix(X_v,  label=y_v,  feature_names=features)

    # Create a watchlist to monitor performance
    watchlist = [(dtrain, 'train'), (dval, 'val')]

    # Callback for storing metrics
    evals_result = {}

    # Train the model
    bst = xgb.train(
            params_for_train,
            dtrain,
            num_boost_round=num_boost_round,
            evals=watchlist,
            evals_result=evals_result,
            verbose_eval=10,  # Print evaluation results every 10 rounds
        )

    # Store losses for averaging later 
    train_loss_matrix[fold_idx, :] = evals_result['train']['rmse']
    val_loss_matrix[fold_idx, :] = evals_result['val']['rmse']

    # Store single rod results
    fold_results.append({
        'rod_id': real_rod_name,
        'rmse': evals_result['val']['rmse']
        })

    # Find best RMSE with relative epoch for each fold
    best_rmse = np.min(evals_result['val']['rmse'])
    best_epoch = np.argmin(evals_result['val']['rmse']) + 1

    print(f"Best RMSE for fold {fold_idx + 1}: {best_rmse} at epoch {best_epoch}")


execution_times['Cross Validation'] = (time.time() - start_cv) / 60
#-------------------------------------------


#-------------------------------------------
# PLOT LEARNING CURVES

# Average losses across folds
avg_train_loss = np.mean(train_loss_matrix, axis=0) 
avg_val_loss = np.mean(val_loss_matrix, axis=0)
std_val_loss = np.std(val_loss_matrix, axis=0)

rounds = range(1, num_boost_round + 1)


# Plot average RMSE curves (over folds)
plt.figure(figsize=(10, 6))
plt.plot(rounds, avg_train_loss, label='Avg training loss', color='blue', linewidth=1)
plt.plot(rounds, avg_val_loss, label='Avg validation loss', color='black', linewidth=1.5)


plt.fill_between(rounds, avg_val_loss - 2*std_val_loss, avg_val_loss + 2*std_val_loss, 
                 alpha=0.1, color='red', label='Val 95.4% CL ($2\sigma$)')
plt.fill_between(rounds, avg_val_loss - std_val_loss, avg_val_loss + std_val_loss, 
                 alpha=0.2, color='red', label='Val 68.2% CL ($1\sigma$)')

# Plot individual fold validation curves
colors = plt.cm.gist_gray(np.linspace(0, 0.7, len(fold_results)))

line_styles = [
    (0, (5, 2, 1, 2)),               
    (0, (5, 5)),           
    (0, (1, 1)),           
    (0, (3, 5, 1, 5)),     
    (0, (5, 1)),           
    (0, (3, 1, 1, 1, 1, 1)) 
]

for i, item in enumerate(fold_results):
    plt.plot(
            rounds, 
            item['rmse'], 
            color=colors[i], 
            alpha=1, 
            linewidth=0.5,
            linestyle=line_styles[i % len(line_styles)],
            label=f"{item['rod_id']} val loss" 
        )

plt.xlabel('N Estimators', labelpad=4)
plt.ylabel('RMSE (Scaled)', labelpad=4)
plt.title(f'Training vs Validation Loss Curves', weight='bold', pad=8)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.ylim(0, 0.5)
plt.yticks(np.arange(0, 0.7, 0.1))
plt.grid(True, alpha=0.4)


plt.savefig(f'loss_curve_{model_arch}.png', bbox_inches='tight', dpi=150)
#-------------------------------------------





# ===================================
# 4. Model Testing
# ===================================

print('\n')
print("\n----- Model Testing -----")
print("\nStarting final training on the entire training set...")

#-------------------------------------------
# CREATE THE XGBOOST MODEL WITH OPTIMAL NUMBER OF ITERATIONS

optimal_num_boost_round = np.argmin(avg_val_loss) + 1

final_xgb = XGBRegressor(
        objective="reg:squarederror", 
        random_state=42, 
        n_jobs=-1, 
        n_estimators=optimal_num_boost_round,
        **best_params
        )
#-------------------------------------------


#-------------------------------------------
# RETRAIN OPTIMAL MODEL ON FULL TRAINING DATA AND SAVE 

start_final_train = time.time()

final_training = final_xgb.fit(X_train, y_train_scaled, verbose=0, sample_weight=weights)

execution_times['Final Training'] = (time.time() - start_final_train) / 60


final_xgb.save_model('final_model.json')
print("\nModel saved!")
#-------------------------------------------


#-------------------------------------------
# MAKE PREDICTIONS AND SAVE

print("\nMaking predictions on test set and saving...")
preds_scaled = final_xgb.predict(X_test).flatten()
preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()


# Print evaluation metrics
rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f"\nFinal RMSE on the test set: {rmse}")

r2 = r2_score(y_test, preds)
print(f"R² Score on the test set: {r2}")


# Compute standardized residuals 
std_residuals = (preds - y_test) / rmse 


# Save results
preds_df = pd.Series(preds, name = "PredictedRidgeHeight").reset_index(drop=True)
res_df = pd.Series(std_residuals, name='StandardizedResidual').reset_index(drop=True)

results_df = pd.concat([preds_df, 
                        res_df], 
                        axis=1)


results_path = os.path.join(parent_dir, "data", f"predictions_{model_arch}.csv")

results_df.to_csv(results_path, sep = ";", index=False)
#-------------------------------------------


# PRINT TRAINING TIMES
print("\n--- Training Times ---")
for key, value in execution_times.items():
    print(f"{key}: {value:.2f} minutes")