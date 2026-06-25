# pcmi-surrogate-model

The project represents an attempt of building a surrogate model (SM) to study PCMI effects in Pressurized Water Reactors during the reactor steady-state operation (Base Irradiation) and transient conditions characterized by power ramps. The objective is to integrate the multidimensional information provided by the SM into a lower dimensional Fuel Performance Code. In the current configuration, a supervised regression model provides an estimate of the cladding diameter deformation due to the 'ridging effect'.

The PCMI dataset, generated through 2D deterministic simulations, contains the thermomechanical variables related to only seven PWR-type rodlets belonging to the same experimental campaign. These rodlets are characterized by specific irradiation histories and structural conditions. Therefore, an extension of the sample statistics is necessary to achieve model robustness and reliability. However, expanding this dataset is constrained by the time required to generate new simulations and, most of all, by the validation process that each new simulated rodlet must undergo to ensure data reliability.

Furthermore, the target engineering code requires a model capable of providing instantaneous predictions; specifically, given a set of thermomechanical variables at a given point in time, it must output the corresponding ridging amplitude. To meet this operational constraint, the dataset was structured by treating individual timesteps as independent data instances, with the instantaneous thermomechanical variables serving as input features.

The tuning of the model hyperparameters was performed through a Group k-fold CV strategy, where each group corresponds to a simulated rodlet. The selected model performance was then assessed via the same strategy by monitoring the evolution of the specific rodlets validation losses, as well as the training and validation errors averaged across folds for each boosting iteration. 

Finally, the optimal model was tested on a held-out rodlet to analyse the residuals of the resulting predictions. Since the data sample statistics is limited, the model is subjected to bias. This bias is mainly related to variations in the input feature space domains among the different rodlets, a factor that contributed into limiting the choice of the test rodlet.




## Project Workflow

First, install the required packages: 

>> pip install -r requirements.txt


Then, follow the steps below to process the raw data, train the model, and analyse the final results.



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

Run directly 'analysis.py' if want to test this model and analyse its results. 