# Big_Market_Sales_Prediction
Analytics Vidhya Hackathon

data : Contains the datasets. All intermediate data generated after transformations are saved in this directory.
Exploratory Data Analysis (EDA) : Latest EDA was Conducted in notebook  1.Exploratory Data Analysis-4.ipynb   (On Local)
Model Training : 
Implemented in:
  •	2.Model_Training_XGB-4.ipynb: Local training, stratified sampling, SHAP analysis
  •	2.Model_Training_XGB-4-Colab.ipynb: Colab-based training with GPU + KFold CV + RandomSearchCV
  •	2_Model_Training_LightGBM_4_Colab: LGBM training
  •	All generated predictions are saved as:
  •	xgb_1_submission.csv
  •	xgb_2_submission.csv
  •	xgb_3_submission.csv
  •	xgb_4_submission.csv
  •	xgb_4_RCV_submission.csv
  •	xgb_final_stratified_submission_colab.csv
  •	xgb_kfold_submission_colab.csv
  •	ensemble_submission_1.csv
  •	lgbm_submission_colab.csv
  
The best model achieved a leaderboard RMSE of 1176.50

