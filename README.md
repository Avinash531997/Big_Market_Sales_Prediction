# Big Market Sales Prediction  
**Analytics Vidhya Hackathon Challenge**  

This project addresses a classical regression problem in the retail domain: predicting **Item Outlet Sales** for products across different store locations. 
Given the limited dataset (1559 unique products across 10 outlets), the task required a strong focus on feature engineering, validation strategy, and model interpretability.

**Documentation**: A complete overview of the analysis and modeling process is provided in the repository.

## Project Structure
- data/Raw and transformed datasets
- label_encoders.pkl # Saved encoders for inference
- submission files/ # Model outputs submitted to leaderboard
- 1.Exploratory Data Analysis-4.ipynb #EDA (Local)
- 2.Model_Training_XGB-4.ipynb # XGBoost (Local)
- 2.Model_Training_XGB-4-Colab.ipynb # XGBoost with GPU + KFold + RCV (Colab)
- 2_Model_Training_LightGBM_4_Colab.ipynb # LightGBM Training (Colab)

## Exploratory Data Analysis (EDA)

Conducted in:  
 `1.Exploratory Data Analysis-4.ipynb`  

Key Findings:
- High sparsity in product distribution (most items appear < 10 times).
- Strong interaction between product and outlet features.
- Target variable (`Item_Outlet_Sales`) was right-skewed → log-transformed to `Log_Sales`.
- Extensive feature engineering added signal to the model.

## Model Training Overview

Model training was conducted using both **local setup** and **Google Colab** (GPU-enabled) environments:

| Notebook                                  | Details                                                                 |
|-------------------------------------------|-------------------------------------------------------------------------|
| `2.Model_Training_XGB-4.ipynb`            | XGBoost on local: Stratified sampling, SHAP-based feature pruning       |
| `2.Model_Training_XGB-4-Colab.ipynb`      | Colab GPU: KFold CV + RandomizedSearchCV                               |
| `2_Model_Training_LightGBM_4_Colab.ipynb` | LightGBM training & evaluation on Colab                                |

## Submission Files

Predictions from different experiments are saved as:

- `xgb_1_submission.csv`  
- `xgb_2_submission.csv`  
- `xgb_3_submission.csv`  
- `xgb_4_submission.csv`  
- `xgb_4_RCV_submission.csv`  
- `xgb_final_stratified_submission_colab.csv`  
- `xgb_kfold_submission_colab.csv`  
- `ensemble_submission_1.csv`  
- `lgbm_submission_colab.csv`  

**Best Leaderboard RMSE Achieved:** `1176.50`  
**Leaderboard Top Score:** `1127.7`

## Key Techniques Applied

- **Label Encoding** for categorical features (with saved encoders)
- **Stratified Sampling** using frequency-based bins for better distribution
- **K-Fold Cross Validation** for robust generalization
- **RandomizedSearchCV** for hyperparameter tuning
- **SHAP Plots** for feature importance and model explainability
- **Feature Pruning** using RFECV and SHAP
  
## Conclusion

Although the model did not surpass the current top leaderboard score (`1127.7`), it achieved a **competitive RMSE of `1176.50`**, reflecting a well-generalized and interpretable solution.  

The SHAP plots confirmed that the model relies on meaningful engineered features like `Item_MRP`, `Sales_to_MRP`, and outlet-level medians.  
