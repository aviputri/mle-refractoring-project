#%%
"""Cleaning the data from 'King_County_House_prices_dataset.csv"""

from math import dist
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
kc_data = pd.read_csv("../data/King_County_House_prices_dataset.csv")

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

# This helper function calculates the distance between one house and a reference location.
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

#%% center and water distance

# center distance
wealth_center = (47.62774, -122.24194)

# Absolute difference in latitude between the center and the property.
kc_data["delta_lat"] = np.absolute(wealth_center[0] - kc_data["lat"])
# Absolute difference in longitude between the center and the property.
kc_data["delta_long"] = np.absolute(wealth_center[1] - kc_data["long"])
# Distance between the center and the property.
kc_data["center_distance"] = (
    (
        (kc_data["delta_long"] * np.cos(np.radians(47.6219))) ** 2
        + kc_data["delta_lat"] ** 2
    )
    ** (1 / 2)
    * 2
    * np.pi
    * 6378
    / 360
)

# water distance
water_distance = []
# Add all waterfront houses to the reference list.
water_list = kc_data.query("waterfront == 1")
# For each row, calculate the distance to the nearest waterfront house.
for idx in kc_data_copy.index:
    ref_list = []
    for x, y in zip(list(water_list["long"]), list(water_list["lat"])):
        ref_list.append(dist(kc_data_copy["long"][idx], kc_data_copy["lat"][idx], x, y).min())
    water_distance.append(min(ref_list))

# Create a new column from the previously computed list.
kc_data_copy["water_distance"] = water_distance

# Create a new column from the previously computed list.
kc_data_copy.describe().round(2)

#%%
#create plots and save them to the plots folder

#Histograms
# Select variables for a closer visual inspection.
columns_histogram = ["price","bathrooms","bedrooms","floors","grade","last_known_change","sqft_living","sqft_lot"]
kc_data_copy[columns_histogram].hist(bins=50, figsize=(20, 15))
plt.savefig("../plots/01_histograms.png", dpi=300, bbox_inches="tight")

#Box plots
fig = px.box(kc_data, y="price", labels={"price": "House Price in $"})
#fig.write_image("../plots/02_price_boxplot.png", width=800, height=600) #still resolving the issue

# Plot scatterplots.
grid = sns.pairplot(kc_data[columns_histogram])
grid.savefig("../plots/03_pairplot.png", dpi=300)

# Heatmap of the Pearson correlation coefficients
numeric_kc_data = kc_data.select_dtypes(include=["number"])
mask = np.triu(numeric_kc_data.corr())
plt.figure(figsize=(20, 15))
ax = sns.heatmap(round(numeric_kc_data.corr(), 2), annot=True, mask=mask, cmap="RdBu_r")
ax.figure.savefig("../plots/04_correlation_heatmap.png", dpi=300, bbox_inches="tight")

# Scatter plot: price versus distance to the center.
scatter = sns.relplot(y="price", x="center_distance", data=kc_data);
scatter.savefig("../plots/05_scatterplot.png", dpi=300, bbox_inches="tight");

# Scatter plot: price per square foot versus distance to the center.
sqft_scatter = sns.relplot(y="sqft_price", x="center_distance", data=kc_data);
sqft_scatter.savefig("../plots/06_sqft_scatterplot.png", dpi=300, bbox_inches="tight");

price_water =sns.relplot(y="price", x="water_distance", data=kc_data);
price_water.savefig("../plots/07_price_water_distance.png", dpi=300, bbox_inches="tight");

sqft_water = sns.relplot(y="sqft_price", x="water_distance", data=kc_data);
sqft_water.savefig("../plots/08_sqft_water_distance.png", dpi=300, bbox_inches="tight");


#%%
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
