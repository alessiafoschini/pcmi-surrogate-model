# pcmi-surrogate-model

## Project Workflow

Follow the steps below to process the raw data, train the model, and analyze the final results.


### 1. Data Preprocessing 
Navigate to the 'preprocessing/' directory:

Inside this folder, the workflow is divided in two phases:

- Data inspection: 'plot_time.py', 'plot_distr.py', 'find_corr.py'
- Data modification: 'create_feat.py', 'balance_data.py', 'split_data.py'


### 2. Model Training, Validation, Testing
Return to the root directory to run 'xgb.py': 

e.g. python3 xgb.py | tee xgb_results_shallow.txt


### 3. Results Analysis
Navigate to the 'postprocessing/' directory to run 'analysis.py'