import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
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
cleaned_raw_data = pd.read_csv('Data/Feature Engineering/new_features_engineered.csv')
cleaned_raw_data_copy = cleaned_raw_data.copy()

# Define the independent variables (features) and the target variable
X = cleaned_raw_data_copy.drop('loan_status', axis=1)
y = cleaned_raw_data_copy['loan_status']

# Convert categorical variable in the X dataset(all columns except 'loan_status') into dummy variables
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create new DataFrames for training and testing sets
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

# Save the training and testing sets to CSV files
train_data.to_csv('Data/Split Data/Model 2/train_data_ML2.csv', index=False)
test_data.to_csv('Data/Split Data/Model 2/test_data.ML2.csv', index=False)


# ================================================== B. TRAIN MODEL 2 ================================================== #
# 1. Train a Random Forest model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# 2. Make predictions with the model
y_pred_rf = rf.predict(X_test)

# 3. Check the accuracy of the model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Model Accuracy: {accuracy_rf}")
""" Answer: Model Accuracy: 0.8557377049180328"""

# 4. Save the predictions to a csv file
predictions_rf = pd.DataFrame(y_pred_rf, columns=['predictions'])
predictions_rf.to_csv('Artifacts/Predictions/rf_predictions.csv', index=False)

# 5. Cross Validate the model by building another model
# For this, we'll use a new split of the data
X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split(X, y, test_size=0.2, random_state=0)
rf_cv = RandomForestClassifier()
rf_cv.fit(X_train_cv, y_train_cv)

# 6. Make predictions on the cross validate model
y_pred_cv_rf = rf_cv.predict(X_test_cv)

# 7. Check the accuracy score of the cross validate model
accuracy_cv_rf = accuracy_score(y_test_cv, y_pred_cv_rf)
print(f"Cross-Validated Model Accuracy: {accuracy_cv_rf}")
""" Answer: Cross-Validated Model Accuracy: 0.8459016393442623"""

# 8. Save the cross validate predictions to csv file
predictions_cv_rf = pd.DataFrame(y_pred_cv_rf, columns=['predictions_cv'])
predictions_cv_rf.to_csv('Artifacts/Predictions/rf_predictions_cv.csv', index=False)

# 9. Determine which model is the most accurate
if accuracy_rf > accuracy_cv_rf:
    most_accurate_model_rf = rf
    print("The original model is the most accurate")
else:
    most_accurate_model_rf = rf_cv
    print("The cross-validated model is the most accurate")
""" Answer: The original model is the most accurate """

# 10. Hyperparameter tuning
# Define the parameter grid
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4],
    'bootstrap': [False],
    'criterion': ['gini', 'entropy']
}

# Create a GridSearchCV object
grid_search_rf = GridSearchCV(most_accurate_model_rf, param_grid_rf, cv=5, scoring='accuracy')

# Fit the GridSearchCV object to the data
grid_search_rf.fit(X_train, y_train)

# Get the best parameters
best_params_rf = grid_search_rf.best_params_
print(f"Best parameters: {best_params_rf}")
""" Answer: Best parameters: {'bootstrap': False, 'criterion': 'entropy', 'max_depth': 20, 'min_samples_leaf': 4, 'min_samples_split': 5, 'n_estimators': 100}"""

# Get the best score
best_score_rf = grid_search_rf.best_score_
print(f"Best score: {best_score_rf}")
""" Answer: Best score: 0.834176617418876 """

# 11. Retrain the most accurate model
# Update the most accurate model with the best parameters and retrain
most_accurate_model_rf = RandomForestClassifier(n_estimators=best_params_rf['n_estimators'], 
                                                max_depth=best_params_rf['max_depth'], 
                                                min_samples_split=best_params_rf['min_samples_split'], 
                                                min_samples_leaf=best_params_rf['min_samples_leaf'], 
                                                bootstrap=best_params_rf['bootstrap'])
most_accurate_model_rf.fit(X_train, y_train)

# Make predictions with the retrained model
y_pred_updated_rf = most_accurate_model_rf.predict(X_test)

# Check the accuracy of the retrained model
accuracy_updated_rf = accuracy_score(y_test, y_pred_updated_rf)
print(f"Updated Model Accuracy: {accuracy_updated_rf}")
""" Answer: Updated Model Accuracy: 0.8655737704918033"""

# Save the updated predictions to a csv file
predictions_updated_rf = pd.DataFrame(y_pred_updated_rf, columns=['predictions_updated'])
predictions_updated_rf.to_csv('Artifacts/Predictions/rf_predictions_updated.csv', index=False)

# Determine which model is the most accurate
if accuracy_updated_rf > accuracy_rf and accuracy_updated_rf > accuracy_cv_rf:
    print("The updated model is the most accurate")
elif accuracy_rf > accuracy_cv_rf:
    print("The original model is the most accurate")
else:
    print("The cross-validated model is the most accurate")
""" Answer: The updated model is the most accurate"""

# 12. Create feature importance values for the most accurate model
feature_importance_updated_rf = pd.DataFrame({'feature': X.columns, 'importance': most_accurate_model_rf.feature_importances_})
feature_importance_updated_rf = feature_importance_updated_rf.sort_values('importance', ascending=False)

# Feature Importance graph
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_updated_rf['feature'], feature_importance_updated_rf['importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.gca().invert_yaxis()
plt.show()

# 13. Save the updated feature importance values to a csv file
feature_importance_updated_rf.to_csv('Artifacts/Feature Importance/most_accurate_model_rf_feature_importance_updated.csv', index=False)

# 14. Save the model that is the most accurate to a pickle file
with open('Artifacts/Models/Model_2.pkl', 'wb') as file:
    pickle.dump(most_accurate_model_rf, file)