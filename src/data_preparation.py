"""Cleaning the data from 'King_County_House_prices_dataset.csv"""

from pathlib import Path

import numpy as np
import pandas as pd


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the King County CSV file."""
    return pd.read_csv(path)


def remove_implausible_bedroom_row(df: pd.DataFrame) -> pd.DataFrame:
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


def rebuild_sqft_basement(df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild basement size from more reliable source columns."""

    df = df.copy()
    # Recompute the feature from two columns that already carry the needed information.
    df["sqft_basement"] = df["sqft_living"] - df["sqft_above"]
    return df


def fill_missing_view_and_waterfront(df: pd.DataFrame) -> pd.DataFrame:
    """Use 0 for missing view and waterfront values."""
    df = df.copy()
    df[["view", "waterfront"]] = (
    df[["view", "waterfront"]].fillna(0)
    )
    return df


def add_last_known_change(df):
    """Create one `last_known_change` column from build + renovation year columns."""

    df = df.copy()
    renovated = df["yr_renovated"]
    df["last_known_change"] = (
    renovated.where(renovated.notna() & renovated.ne(0), df["yr_built"])
    .astype(int)
    )
    return df.drop(columns=["yr_renovated", "yr_built"])


def add_sqft_price(df: pd.DataFrame) -> pd.DataFrame:
    """Add the price ratio used in the notebook."""
    df = df.copy()
    df["sqft_price"] = (
    df["price"] / (df["sqft_living"] + df["sqft_lot"])
    ).round(2)
    return df


def distance_km(longitude, latitude, reference_longitude, reference_latitude):
    """Approximate geographic distance in kilometers."""
    delta_long = longitude - reference_longitude
    delta_lat = latitude - reference_latitude
    corrected_long = delta_long * np.cos(np.radians(reference_latitude))
    return (
    (corrected_long**2 + delta_lat**2) ** 0.5
    * 2 * np.pi * 6378 / 360
    )


CENTER_LAT = 47.62774
CENTER_LONG = -122.24194
CENTER_CORRECTION_LAT = 47.6219


def add_center_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Add the notebook's distance to the reference location."""
    df = df.copy()
    df["delta_lat"] = (CENTER_LAT - df["lat"]).abs()
    df["delta_long"] = (CENTER_LONG - df["long"]).abs()
    df["center_distance"] = (
    (
    (df["delta_long"] * np.cos(np.radians(CENTER_CORRECTION_LAT))) ** 2
    + df["delta_lat"] ** 2
    )
    ** 0.5
    * 2 * np.pi * 6378 / 360
    )
    return df


def add_water_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Add distance to the nearest waterfront reference house."""
    df = df.copy()
    waterfront_houses = df.loc[
    df["waterfront"].eq(1), ["long", "lat"]
    ]
    if waterfront_houses.empty:
        raise ValueError("No waterfront reference houses found.")

    distances = []
    for house in df.itertuples():
        to_references = distance_km(
            house.long,
            house.lat,
            waterfront_houses["long"],
            waterfront_houses["lat"],
        )
        distances.append(to_references.min())
    df["water_distance"] = distances
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run the complete cleaning and feature-engineering workflow."""
    df = remove_implausible_bedroom_row(df)
    df = rebuild_sqft_basement(df)
    df = fill_missing_view_and_waterfront(df)
    df = add_last_known_change(df)
    df = add_sqft_price(df)
    df = add_center_distance(df)
    df = add_water_distance(df)
    return df