# Import Libraries
import csv                                              # Read and Write to CSV files
import pandas as pd                                     # Manipulation and analysis of data
import numpy as np                                      # Mathematical operations
from scipy import stats                                 # Statistical functions
import matplotlib.pyplot as plt                         # Matplotlib and Seaborn is used to create visual graphs
import seaborn as sns                                   
import warnings                                         # Ignores any future warnings
warnings.filterwarnings('ignore')   

# Read both CSV Files
credit_risk_raw_data = pd.read_csv('Data/Original Data/credit_risk_raw_data.csv')
credit_risk_raw_data_copy = credit_risk_raw_data.copy()

credit_risk_validation_data = pd.read_csv('Data/Original Data/validation_data.csv')
credit_risk_validation_data_copy = credit_risk_validation_data.copy()


# ======================================================= A. DATA CLEANING PROCESSES ======================================================= #

# 1. Attribute Name Standardization Process for both dataset
#Create a dictionary to map the old column names to the new ones to format the column names:
raw_data_column_name_mapping = {
    'cb_person_cred_hist_length': 'person_cred_hist_length'
}

validation_data_column_name_mapping = {
    'cb_person_cred_hist_length': 'person_cred_hist_length'
}

#Replace the original column names with the formatted names: 
credit_risk_raw_data_copy.rename(columns=raw_data_column_name_mapping, inplace=True)
credit_risk_validation_data_copy.rename(columns=validation_data_column_name_mapping, inplace=True)


# 2. For both dataset, check for missing values
print(f"Number of Missing Values in credit_risk_raw_data_copy:\n{credit_risk_raw_data_copy.isnull().sum()}\n")
print(f"Number of Missing Values in credit_risk_validation_data_copy:\n{credit_risk_validation_data_copy.isnull().sum()}\n")

""" Answers:  
Number of Missing Values in credit_risk_raw_data_copy:
    person_age                   0
    person_income                0
    person_home_ownership        0
    person_emp_length           56
    loan_intent                  0
    loan_amnt                    0
    loan_int_rate              156
    loan_percent_income          0
    person_cred_hist_length      0
    loan_status                  0
    dtype: int64

Number of Missing Values in credit_risk_validation_data_copy:
    person_age                 0
    person_income              0
    person_home_ownership      0
    person_emp_length          0
    loan_intent                0
    loan_amnt                  0
    loan_int_rate              0
    loan_percent_income        0
    person_cred_hist_length    0
    dtype: int64
"""

#Fill in missing values
""" 
    # Based on the missing value check above, it was noted that there are missing values in the following attributes for the credit_risk_raw_data_copy dataset:
        - person_emp_length
        - loan_int_rate

    # In order to fill in the missing values the attributes need to be split into Ordinal and Numerical Variables.
        # For Ordinal Variables use the mode of all the values in the attribute to fill in the missing data values:
            - person_emp_length (0 - 23, 27 or 31)
            - 


        # For Numerical Variables use either the mean or median of the values in the attribute to fill in the missing data values
            - loan_int_rate
"""
#=== ORDINAL VARIABLES === 
credit_risk_raw_data_copy['person_emp_length'].fillna(credit_risk_raw_data_copy['person_emp_length'].mode()[0],inplace=True)

#=== NUMERICAL VARIABLES ===
# Median is used instead of mean due to the outliers in the attributes data which could negatively impact the outcome
credit_risk_raw_data_copy['loan_int_rate'].fillna(credit_risk_raw_data_copy['loan_int_rate'].median(),inplace=True)

#Check the file to see whether the missing values have been added
print(f"Number of Missing Values in credit_risk_raw_data_copy:\n{credit_risk_raw_data_copy.isnull().sum()}\n")

"""Answer: 
    - Number of Missing Values in credit_risk_raw_data_copy:       
        person_age                 0
        person_income              0
        person_home_ownership      0
        person_emp_length          0
        loan_intent                0
        loan_amnt                  0
        loan_int_rate              0
        loan_percent_income        0
        person_cred_hist_length    0
        loan_status                0
        dtype: int64
"""

# 3. For both dataset, check for duplicate records and remove them from the dataset
print(f"Number of duplicate rows in raw_data_copy: {credit_risk_raw_data_copy.duplicated().sum()}")
print(f"Number of duplicate rows in validation_data_copy: {credit_risk_validation_data_copy.duplicated().sum()}\n")

""" Answers:
    - Number of duplicate rows in raw_data_copy: 1
    - Number of duplicate rows in validation_data_copy: 0
"""

# There is a duplicate record in the credit_risk_raw_data_copy dataset, as a result it will need to be dropped 
credit_risk_raw_data_copy = credit_risk_raw_data_copy.drop_duplicates()

#Check that the duplicate record was removed
print(f"Number of duplicate rows in raw_data_copy: {credit_risk_raw_data_copy.duplicated().sum()}\n")
""" Answers: Number of duplicate rows in raw_data_copy: 0 """

# 4. Convert the datatype of the attribute 'person_emp_length' in credit_risk_validation_data_copy to float64
credit_risk_validation_data_copy['person_emp_length'] = credit_risk_validation_data_copy['person_emp_length'].astype('float64')
print(f"person_emp_length datatype: {credit_risk_validation_data_copy['person_emp_length'].dtypes}\n")
"""Answer: person_emp_length datatype: float64"""

# 4. For both dataset, perform outliers Data Handling
#person_income
# For the 'credit_risk_validation_data_copy' dataset
credit_risk_validation_data_copy = credit_risk_validation_data_copy[credit_risk_validation_data_copy['person_income'] < 1000000]
credit_risk_raw_data_copy = credit_risk_raw_data_copy[credit_risk_raw_data_copy['person_income'] < 1000000]

print(f"credit_risk_raw_data_copy person_emp_length datatype:\n{credit_risk_raw_data_copy['person_income'].describe()}\n")
print(f"credit_risk_validation_data_copy person_emp_length datatype:\n{credit_risk_validation_data_copy['person_income'].describe()}\n")

"""
Answer:
    - credit_risk_raw_data_copy person_emp_length datatype: 
        count      1523.000000
        mean      60931.252791
        std       66339.170965
        min        4000.000000
        25%       34000.000000
        50%       42000.000000
        75%       60758.000000
        max      900000.000000
        Name: person_income, dtype: float64

    - credit_risk_validation_data_copy person_emp_length datatype: 
        count       469.000000
        mean      68590.279318
        std       69577.787894
        min        9600.000000
        25%       34000.000000
        50%       55000.000000
        75%       77052.000000
        max      762000.000000
        Name: person_income, dtype: float64
"""

# 5. For both datasets, write the new datasets to CSV files
credit_risk_raw_data_copy.to_csv('Data/Clean Data/cleaned_credit_risk_raw_data.csv', index=False)
credit_risk_validation_data_copy.to_csv('Data/Clean Data/cleaned_credit_risk_validation_data.csv', index=False)