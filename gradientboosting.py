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

# Create a gradient boosting model
gb_model = GradientBoostingRegressor()
gb_model.fit(X_train, y_train)

# Make predictions using gradient boosting model
gb_y_pred = gb_model.predict(X_test)

# Calculate evaluation metrics for gradient boosting model
gb_mse = mean_squared_error(y_test, gb_y_pred)
gb_rmse = mean_squared_error(y_test, gb_y_pred, squared=False)
gb_r2 = r2_score(y_test, gb_y_pred)

print("Gradient Boosting - Mean Squared Error:", gb_mse)
print("Gradient Boosting - Root Mean Squared Error:", gb_rmse)
print("Gradient Boosting - R-squared:", gb_r2)