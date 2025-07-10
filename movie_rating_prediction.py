# movie_rating_prediction.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv("IMDb_Indian_Movies.csv")

# Display basic info
print("Initial Dataset Shape:", df.shape)
print(df.head())

# Drop unnecessary columns
df.drop(columns=['Poster_Link', 'Overview'], inplace=True, errors='ignore')

# Handle missing values
df.dropna(inplace=True)

# Convert Votes to numeric if in string format
if df['Votes'].dtype == object:
    df['Votes'] = df['Votes'].str.replace(',', '').astype(float)

# Extract year from Release Date
df['Year'] = pd.to_datetime(df['Release Date'], errors='coerce').dt.year
df.drop(columns=['Release Date'], inplace=True, errors='ignore')

# Drop any remaining nulls
df.dropna(inplace=True)

# One-hot encode categorical features
df = pd.get_dummies(df, columns=['Genre', 'Certificate'], drop_first=True)

# Prepare features and target
X = df.drop('Rating', axis=1)
y = df['Rating']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"\n{name} Performance:")
    print(f"R2 Score  : {r2_score(y_test, y_pred):.3f}")
    print(f"MAE       : {mean_absolute_error(y_test, y_pred):.3f}")
    print(f"RMSE      : {mean_squared_error(y_test, y_pred, squared=False):.3f}")

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
evaluate_model("Linear Regression", lr, X_test, y_test)

# Random Forest
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
evaluate_model("Random Forest", rf, X_test, y_test)

# XGBoost
xgb = XGBRegressor()
xgb.fit(X_train, y_train)
evaluate_model("XGBoost", xgb, X_test, y_test)
