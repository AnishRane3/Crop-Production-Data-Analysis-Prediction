# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 23:18:44 2024

@author: ranea
"""

from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import learning_curve


df = pd.read_csv('Cleaned_Crop_Production.csv')

data = df.drop(['State_Name'], axis=1)
dummy = pd.get_dummies(data)

x = dummy.drop(["Production", "Yield"], axis=1)
y = dummy["Production"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=5)

print("x_train :", x_train.shape)
print("x_test :", x_test.shape)
print("y_train :", y_train.shape)
print("y_test :", y_test.shape)



xgb_model = xgb.XGBRegressor(
    reg_alpha=10, 
    reg_lambda=0.1,  
    n_estimators=100,  
    objective='reg:squarederror', 
    eval_metric='mae'  
)


xgb_model.fit(x_train, y_train)


y_train_pred = xgb_model.predict(x_train)
y_test_pred = xgb_model.predict(x_test)


train_r2 = r2_score(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

print("XGBoost Training R²:", train_r2)
print("XGBoost Training MAE:", train_mae)
print("XGBoost Test R²:", test_r2)
print("XGBoost Test MAE:", test_mae)


plt.figure(figsize=(10,6))
plt.scatter(y_test, y_test_pred, color='blue', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', lw=2)  # 45-degree line
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('XGBoost: Test Set Actual vs Predicted Values')
plt.grid(True)
plt.show()


train_sizes, train_scores, test_scores = learning_curve(
    xgb_model, x, y, train_sizes=[100, 500, 1000, 5000, 10000], cv=5, scoring='neg_mean_absolute_error'
)

train_errors_mean = -train_scores.mean(axis=1)
test_errors_mean = -test_scores.mean(axis=1)

plt.figure(figsize=(10,6))
plt.plot(train_sizes, train_errors_mean, label='Training Error')
plt.plot(train_sizes, test_errors_mean, label='Test Error')
plt.xlabel('Training Set Size')
plt.ylabel('Mean Absolute Error')
plt.title('Learning Curves')
plt.legend()
plt.grid(True)
plt.show()
