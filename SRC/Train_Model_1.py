import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


# ================================================== A. SPLITTING THE RAW DATA INFORMATION ================================================== #
"""     
    # Dummy data is used to convert the categorical data into 0's and 1's  to make it easy to be quantified and compared in the future models
        -> Example: Gender has Male and Female categories
        -> Using 'dummies' from pandas, it converts them into Gender_Male = 1 and Gender_Female = 0
    # Training data set has weight 80% 0r 0.8
    # Testing data set has weight 20% or 0.2
    # 'random_state=42' is used for reproducibility, meaning if the code is run multiple times the same train/test split will occur every time.
"""

# Read Cleaned CSV Files
cleaned_raw_data = pd.read_csv('Data/Clean Data/cleaned_credit_risk_raw_data.csv')
cleaned_raw_data_copy = cleaned_raw_data.copy()

# Define the independent variables (features) and the target variable
X = cleaned_raw_data_copy.drop('loan_status', axis=1)
y = cleaned_raw_data_copy['loan_status']

# Convert categorical variable in the X dataset(all columns except 'loan_status') into dummy variables
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

""" # Create new DataFrames for training and testing sets
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

# Save the training and testing sets to CSV files
train_data.to_csv('Data/Split Data/Model 1/train_data_ML1.csv', index=False)
test_data.to_csv('Data/Split Data/Model 1/test_data_ML1.csv', index=False)
"""

# ================================================== B. TRAIN MODEL 1 ================================================== #
# 1. Train a Logistic Regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# 2. Make predictions with the model
y_pred = logreg.predict(X_test)

# 3. Check the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")
"""Answer: Model Accuracy: 0.7344262295081967 """

# 4. Save the predictions to a csv file
predictions = pd.DataFrame(y_pred, columns=['predictions'])
predictions.to_csv('Artifacts/Predictions/logreg_predictions.csv', index=False)

# 5. Cross Validate the model by building another model
# For this, we'll use a new split of the data
X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split(X, y, test_size=0.2, random_state=0)
logreg_cv = LogisticRegression()
logreg_cv.fit(X_train_cv, y_train_cv)

# 6. Make predictions on the cross validate model
y_pred_cv = logreg_cv.predict(X_test_cv)

# 7. Check the accuracy score of the cross validate model
accuracy_cv = accuracy_score(y_test_cv, y_pred_cv)
print(f"Cross-Validated Model Accuracy: {accuracy_cv}")
"""Answer: Cross-Validated Model Accuracy: 0.7377049180327869 """

# 8. Save the cross validate predictions to csv file
predictions_cv = pd.DataFrame(y_pred_cv, columns=['predictions_cv'])
predictions_cv.to_csv('Artifacts/Predictions/logreg_predictions_cv.csv', index=False)

# 9. Determine which model is the most accurate
if accuracy > accuracy_cv:
    most_accurate_model = logreg
    print("The original model is the most accurate.")
else:
    most_accurate_model = logreg_cv
    print("The cross-validated model is the most accurate.")

"""Answer: The cross-validated model is the most accurate. """

# 10. Hyperparameter tuning
# Define the parameter grid
param_grid = {
    'C': np.logspace(-3, 3, 10),  # exploring a different range and granularity for C
    'penalty': ['l2'],  # only using l2 penalty
    'solver': ['newton-cg', 'lbfgs', 'liblinear']  # adding a new hyperparameter to explore
}

# Create a GridSearchCV object
grid_search = GridSearchCV(most_accurate_model, param_grid, cv=5, scoring='accuracy')

# Fit the GridSearchCV object to the data
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")
""" Answer: Best parameters: {'C': 1000.0, 'penalty': 'l2', 'solver': 'newton-cg'} """

# Get the best score
best_score = grid_search.best_score_
print(f"Best score: {best_score}")
""" Answer: Best score: 0.7717870876340822"""

# 11. Retrain the most accurate model
# Update the most accurate model with the best parameters and retrain
most_accurate_model = LogisticRegression(C=best_params['C'], penalty=best_params['penalty'])
most_accurate_model.fit(X_train, y_train)

# Make predictions with the retrained model
y_pred_updated = most_accurate_model.predict(X_test)

# Check the accuracy of the retrained model
accuracy_updated = accuracy_score(y_test, y_pred_updated)
print(f"Updated Model Accuracy: {accuracy_updated}")
""" Answer: Updated Model Accuracy: 0.7278688524590164 """

# Save the updated predictions to a csv file
predictions_updated = pd.DataFrame(y_pred_updated, columns=['predictions_updated'])
predictions_updated.to_csv('Artifacts/Predictions/logreg_predictions_updated.csv', index=False)

# Determine which model is the most accurate
if accuracy_updated > accuracy and accuracy_updated > accuracy_cv:
    print("The updated model is the most accurate.")
elif accuracy > accuracy_cv:
    print("The original model is the most accurate.")
else:
    print("The cross-validated model is the most accurate.")
""" Answer: The cross-validated model is the most accurate. """

# 12. Create feature importance values for the most accurate model
feature_importance_updated = pd.DataFrame({'feature': X.columns, 'importance': most_accurate_model.coef_[0]})
feature_importance_updated = feature_importance_updated.sort_values('importance', ascending=False)

# 13. Save the updated feature importance values to a csv file
feature_importance_updated.to_csv('Artifacts/Feature Importance/most_accurate_model_feature_importance_updated.csv', index=False)

# 14. Save the model that is the most accurate to a pickle file named 'Model_1.pkl'
with open('Artifacts/Models/Model_1.pkl', 'wb') as file:
    pickle.dump(most_accurate_model, file)