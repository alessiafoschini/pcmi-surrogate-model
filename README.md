# pcmi-surrogate-model

The project represents an attempt of building a surrogate model to analyze PCMI effects in Light Water Reactors. In the current configuration, the supervised regression model provides an estimate of the cladding diameter deformation due to the 'ridging effect'. 

The PCMI dataset, generated through 2D deterministic simulations, contains the thermomechanical variables related to seven PWR-type rodlets belonging to the same experimental campaign. Therefore, an extension of the sample statistics is necessary to achieve model generalization and reliability. 


## Project Workflow

Follow the steps below to process the raw data, train the model, and analyze the final results.


### 1. Data Preprocessing 
Navigate to the 'preprocessing/' directory:

Inside this folder, the workflow is divided in two phases:

- Data inspection: 'plot_time.py', 'plot_distr.py', 'find_corr.py'
- Data modification: 'create_feat.py', 'balance_data.py', 'split_data.py'


### 2. Model Training and Validation
Return to the root directory to run 'xgb.py': 

>> python3 xgb.py | tee xgb_output_s.txt


### 3. Model Testing and Predictions Analysis
Navigate to the 'postprocessing/' directory to run 'analysis.py':

>> python3 analysis.py | tee results_eval_s.txt


### Skip to the 3rd point 
The file 'xgb_model_d.json' contains a pretrained model with a deep architecture. 
The file 'target_scaler.pkl' contains the scaler parameters. 

Run directly 'analysis.py' if want to test this model and analyze its results. 