import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

df = pd.read_csv('ndxt.csv')

# Display the first few rows of the dataset
print(df.head())

# Get summary statistics of the dataset
print(df.describe())

# Check the data types of each column
print(df.dtypes)

# Check the dimensions of the dataset (number of rows and columns)
print(df.shape)

# Visualize the data
plt.plot(df['Date'], df['Close'])
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('NASDAQ 100 Technology Sector - Stock Market Performance')
plt.xticks(rotation=45)
plt.show()
