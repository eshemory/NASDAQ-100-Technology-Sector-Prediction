import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

df = pd.read_csv('ndxt.csv')

# Split the dataset into features (X) and target variable (y)
X = df.drop(['Close', 'Date'], axis=1)  # Exclude the column you want to predict
y = df['Close']

# Split the data into training and testing sets (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Create random forest model
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)

# Make predictions using random forest model
rf_y_pred = rf_model.predict(X_test)

# Calculate evaluation metrics for random forest model
rf_mse = mean_squared_error(y_test, rf_y_pred)
rf_rmse = mean_squared_error(y_test, rf_y_pred, squared=False)
rf_r2 = r2_score(y_test, rf_y_pred)

print("Random Forest - Mean Squared Error:", rf_mse)
print("Random Forest - Root Mean Squared Error:", rf_rmse)
print("Random Forest - R-squared:", rf_r2)