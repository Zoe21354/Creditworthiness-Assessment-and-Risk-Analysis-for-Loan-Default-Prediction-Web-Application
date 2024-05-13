import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score

# Load the model
with open('Artifacts/Models/Model_2.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the data
data = pd.read_csv('Data/Clean Data/cleaned_credit_risk_validation_data.csv')
validation_data = data.copy()

# Create new features
validation_data['income_to_age_ratio'] = validation_data['person_income'] / validation_data['person_age']
validation_data['loan_amt_to_income_ratio'] = validation_data['loan_amnt'] / validation_data['person_income']
validation_data['emp_length_to_age_ratio'] = validation_data['person_emp_length'] / validation_data['person_age']

# Drop the original features
validation_data.drop(['person_income', 'person_age', 'loan_amnt', 'person_emp_length'], axis=1, inplace=True)

# Convert categorical variables into dummy/indicator variables (i.e. one-hot encoding)
validation_data = pd.get_dummies(validation_data)

# Get the feature names from the model
try:
    feature_names = model.feature_names_in_
except AttributeError:
    feature_names = ['loan_int_rate', 'income_to_age_ratio', 'loan_amt_to_income_ratio', 'loan_percent_income', 
                        'emp_length_to_age_ratio', 'person_home_ownership_RENT', 'person_cred_hist_length', 
                        'loan_intent_HOMEIMPROVEMENT', 'person_home_ownership_MORTGAGE', 'loan_intent_MEDICAL', 
                        'person_home_ownership_OWN', 'loan_intent_DEBTCONSOLIDATION', 'loan_intent_EDUCATION', 
                        'loan_intent_VENTURE', 'loan_intent_PERSONAL', 'person_home_ownership_OTHER']

# Create a template DataFrame with all the necessary features
template = pd.DataFrame(columns=feature_names)

# Fill in the values from the validation data
for feature in feature_names:
    if feature in validation_data.columns:
        template[feature] = validation_data[feature]
    else:
        template[feature] = 0

# Perform predictions
predictions = model.predict(template)

# Convert predictions from 0 and 1 to 'N' and 'Y'
predictions = ['Y' if prediction == 1 else 'N' for prediction in predictions]

# Append predictions to the validation_data
validation_data['loan_status'] = predictions