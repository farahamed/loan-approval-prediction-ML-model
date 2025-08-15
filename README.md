Loan Approval Prediction
 Project Overview:
This project predicts whether a loan application will be Approved or Rejected based on applicant and financial information.
It uses Machine Learning (Logistic Regression) to classify applications and handles imbalanced data with SMOTE oversampling to improve fairness for minority classes.

Dataset:
Source: Loan Approval Prediction dataset (Kaggle)
Rows: 4,269
Columns: 13 (loan_id, applicant details, financial assets, loan status, etc.)
Target Variable: loan_status (Approved / Rejected)

Tools & Libraries Used:
Python
Pandas → Data loading, cleaning, and manipulation
Scikit-learn → Model building & evaluation
Imbalanced-learn (SMOTE) → Handle class imbalance
StandardScaler → Feature scaling

Data Preprocessing Steps:
Removed unnecessary columns (loan_id)
Stripped spaces from column names
Label encoding for categorical columns (education, self_employed, loan_status)
Feature scaling using StandardScaler
Oversampling with SMOTE to balance approved and rejected classes

Model Training:
Model: Logistic Regression
Train-Test Split: 80% train / 20% test
Oversampling applied only on the training set to avoid data leakage

Model Evaluation:
Test Set Results:
Metric	Class 0 (Rejected)	Class 1 (Approved)	Overall
Precision	0.95	0.85	
Recall	0.90	0.92	
F1-Score	0.92	0.88	
Accuracy	-	-	0.91

Precision: Measures how many predicted approvals are actually approved

Recall: Measures how many actual approvals were correctly predicted

F1-Score: Balance between precision & recall

Accuracy: Overall correctness of predictions
