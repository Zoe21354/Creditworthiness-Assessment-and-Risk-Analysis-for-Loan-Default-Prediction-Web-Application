import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt                         # Matplotlib and Seaborn is used to create visual graphs
import seaborn as sns  
import warnings                                         # Ignores any future warnings
warnings.filterwarnings('ignore')  


cleaned_raw_data = pd.read_csv('Data/Clean Data/cleaned_credit_risk_raw_data.csv')
cleaned_raw_data_copy = cleaned_raw_data.copy()


# =============================================== FEATURE ENGINEERING MODEL 1 =============================================== #
"""     
    # Feature engineering transforms or combines raw data into a format that can be easily understood by machine learning models.
    # Creates predictive model features, also known as a dimensions or variables, to generate model predictions.
    # This highlights the most important patterns and relationships in the data, which then assists the machine learning model to learn from the data more effectively.
"""
""" 
# Feature 1: Income to Age Ratio
    - This feature could provide insight into the financial maturity of the borrower. 
    - A higher ratio might indicate that the borrower has a high income relative to their age, which could potentially lead to a lower risk of default. 
"""
cleaned_raw_data_copy['income_to_age_ratio'] = cleaned_raw_data_copy['person_income'] / cleaned_raw_data_copy['person_age']


""" 
# Feature 2: Loan Amount to Income Ratio 
    - This feature could provide insight into the borrower’s ability to repay the loan. 
    - A lower ratio might indicate that the borrower has sufficient income to repay the loan, which could potentially lead to a lower risk of default.
"""
cleaned_raw_data_copy['loan_amt_to_income_ratio'] = cleaned_raw_data_copy['loan_amnt'] / cleaned_raw_data_copy['person_income']


""" 
# Feature 3: Employment Length to Age Ratio 
    - This feature could provide insight into the stability of the borrower’s income. 
    - A higher ratio might indicate that the borrower has had a stable source of income for a significant portion of their life, which could potentially lead to a lower risk of default.
"""
cleaned_raw_data_copy['emp_length_to_age_ratio'] = cleaned_raw_data_copy['person_emp_length'] / cleaned_raw_data_copy['person_age']


# Plotting the distribution of the new features
plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)
sns.distplot(cleaned_raw_data_copy['income_to_age_ratio'])
plt.title('Income to Age Ratio')

plt.subplot(2, 2, 2)
sns.distplot(cleaned_raw_data_copy['loan_amt_to_income_ratio'])
plt.title('Loan Amount to Income Ratio')

plt.subplot(2, 2, 3)
sns.distplot(cleaned_raw_data_copy['emp_length_to_age_ratio'])
plt.title('Employment Length to Age Ratio')

plt.tight_layout()
plt.show()


## Remove all features that created the new features
    # The correlation between those old feature and the new features are very high.
    # Logistic regression assume that the variables are not highly correlated.
    # Due to this the excess noise in the datasets are removed.

cleaned_raw_data_copy = cleaned_raw_data_copy.drop(['person_income', 'person_age', 'loan_amnt', 'person_emp_length'], axis=1)

# Store new Features in CSV files
cleaned_raw_data_copy.to_csv('Data/Feature Engineering/new_features_engineered.csv', index=False)