"""Cleaning the data from 'King_County_House_prices_dataset.csv"""

from pathlib import Path

import pandas as pd

# Load the dataset with pandas
kc_data = pd.read_csv("data/King_County_House_prices_dataset.csv")

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