#%%
"""Cleaning the data from 'King_County_House_prices_dataset.csv"""

from pathlib import Path
from matplotlib.pylab import poly
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Load the dataset with pandas
#kc_data = pd.read_csv("data/King_County_House_prices_dataset.csv")
kc_data = pd.read_csv("~/Documents/Courses/AIEng/project01/mle-refractoring-project/data/King_County_House_prices_dataset.csv")

# kc_data.head()

#%%
# FUNCTIONS

"""
data cleaning functions
"""

def bath_bed_ratio_outlier(df):
    """Drop rows with implausible bathroom-to-bedroom ratios.

    Ratios >= 2 or <= 0.10 are treated as outliers for this exercise.
    """

    # Copy first so the caller's DataFrame is not modified in place.
    df = df.copy()
    # Create a temporary helper feature that makes the filtering logic readable.
    df["bath_bed_ratio"] = df["bathrooms"] / df["bedrooms"]

    # Mark the rows that violate the chosen teaching threshold.
    invalid_ratio = (df["bath_bed_ratio"] >= 2) | (df["bath_bed_ratio"] <= 0.10)
    # Keep only the rows whose ratio stays inside the valid range.
    df = df.loc[~invalid_ratio].copy()
    # Drop the helper column so downstream code sees the original schema.
    df.drop(columns=["bath_bed_ratio"], inplace=True)
    return df

def sqft_basement(df):
    """Rebuild basement size from more reliable source columns."""

    df = df.copy()
    # Recompute the feature from two columns that already carry the needed information.
    df["sqft_basement"] = df["sqft_living"] - df["sqft_above"]
    return df


def calculate_last_change(df):
    """Create one `last_known_change` column from build + renovation year columns."""

    df = df.copy()
    last_known_change = []
    for idx, yr_re in df["yr_renovated"].items():
        # Missing or zero renovation years mean "no known renovation",
        # so the build year becomes the last known change.
        if str(yr_re) == "nan" or yr_re == 0.0:
            last_known_change.append(df["yr_built"][idx])
        else:
            # Otherwise preserve the renovation year as the latest known change.
            last_known_change.append(int(yr_re))

    # Add the consolidated feature and remove the redundant source columns.
    df["last_known_change"] = last_known_change
    df.drop("yr_renovated", axis=1, inplace=True)
    df.drop("yr_built", axis=1, inplace=True)
    return df


def fill_missings_view_wf(df):
    """Fill nullable visibility features with explicit zeros.

    Here, `0` means "no view / no waterfront" and is a valid business value.
    """

    df = df.copy()
    # In this dataset, 0 is the explicit business meaning for "not present".
    df["view"] = df["view"].fillna(0)
    df["waterfront"] = df["waterfront"].fillna(0)
    return df

"""
feature engineering functions
"""

def add_column(df, column_name=str):
    """Add a new column to the Dataframe"""
    df = df.copy()
    df[column_name] = (df.price / (df.sqft_living + df.sqft_lot)).round(2)
    return df

def dist(long, lat, ref_long, ref_lat):
    """dist computes the distance in km to a reference location.
    Input: long and lat of the location of interest and ref_long and ref_lat
    as the long and lat of the reference location"""
    delta_long = long - ref_long
    delta_lat = lat - ref_lat
    delta_long_corr = delta_long * np.cos(np.radians(ref_lat))
    return (
        ((delta_long_corr) ** 2 + (delta_lat) ** 2) ** (1 / 2) * 2 * np.pi * 6378 / 360
    )

"""
Modeling functions
"""
def input(df, drop_lst):
    """input function returns the features and target columns"""
    df = df.copy()
    all_features = [x for x in df.columns if x not in drop_lst]
    X = df[all_features]
    y = df.price

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    #print the train and test shapes
    print("X_train (features for the model to learn from): ", X_train.shape)
    print("y_train (labels for the model to learn from): ", y_train.shape)
    print("X_test (features to test the model's accuracy against): ", X_test.shape)
    print("y_test (labels to test the model's accuracy with): ", y_test.shape)

    return X_train, y_train, X_test, y_test

def linear_regression_model(X_train, y_train, X_test, y_test, var_list):
    """linear_regression_model function returns the linear regression model"""
    model_lin_reg = LinearRegression()
    model_lin_reg.fit(X_train[var_list], y_train)

    model_score = model_lin_reg.score(X_test[var_list], y_test)
    X_test_length = X_test.shape[0]

    R_square = round(1-(1-model_score)*(X_test_length-1)/ (X_test_length-len(var_list) - 1),2,)

    # Evaluate how well the model performs on the test data.
    print("Adj. R^2:", R_square)

def polynomial_regression_model(X_train, y_train, X_test, y_test, drop_lst):
    X_train_poly = X_train.copy()
    X_test_poly = X_test.copy()

    # Drop the `id` column.
    X_train_poly = X_train_poly.drop(columns=drop_lst)
    X_test_poly = X_test_poly.drop(columns=drop_lst)

    # Create transformed variables with `poly`.
    X_train_sq = poly.fit_transform(X_train_poly)

    # Apply the same transformation to the test data.
    X_test_sq = poly.transform(X_test_poly)

    # Instantiate the model.
    model_lin_reg = LinearRegression()
    # Train the model with the squared features.
    model_lin_reg.fit(X_train_sq, y_train)

    model_score = model_lin_reg.score(X_test_sq, y_test)
    X_test_length = X_test_sq.shape[0]
    X_test_width = X_test_sq.shape[1]

    R_square = round(1-(1-model_score)*(X_test_length-1)/ (X_test_length-X_test_width - 1),2,)

    # Evaluate how well the model performs on the test data.
    print("Adj. R^2:", R_square)


#%%
# Main code
kc_data_copy = kc_data.copy()

# Clean the datasets
# bed-bath ratio outliers
kc_data_copy = bath_bed_ratio_outlier(kc_data_copy)

# recalculate sqft_basement
kc_data_copy = sqft_basement(kc_data_copy)

# recalculate last_known_change
kc_data_copy = calculate_last_change(kc_data_copy)

# fill missing values for view and waterfront
kc_data_copy = fill_missings_view_wf(kc_data_copy)

# add sqft_price column to the dataset
kc_data_copy = add_column(kc_data_copy, "sqft_price")

#%%
# model data input
drop_lst = ["price", "sqft_price", "date", "delta_lat", "delta_long"]
X_train, y_train, X_test, y_test = input(kc_data_copy, drop_lst)

# linear model
linear_regression_model(X_train, y_train, X_test, y_test, ["grade"])
linear_regression_model(X_train, y_train, X_test, y_test, ["grade", "last_known_change"])

# polynomial model



# %%
