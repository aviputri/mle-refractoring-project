#%% IMPORTS ----------------------------------------------------------------
from pathlib import Path
from src.data_preparation import load_data, prepare_data
#plot libraries
from matplotlib.pylab import poly
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
 
#%% MAIN ---------------------------------------------------------------

def main():
    project_dir = Path(__file__).resolve().parent
    csv_path = (
        project_dir / "data" / "King_County_House_prices_dataset.csv"
    )

    print("1/4: Load raw data ...", flush=True)
    raw = load_data(csv_path)
    print("Raw data:", raw.shape)

    print("2/4: Processing data; please wait ...", flush=True)
    prepared = prepare_data(raw) #cleaned data

    print("3/4: Check results ...", flush=True)
    assert raw.shape == (21597, 21), "Raw data shape is incorrect"
    assert prepared.isna().sum().sum() == 0, "Missing values found in prepared data"
    print("Results:", prepared.shape)
    print("Missing Values:", int(prepared.isna().sum().sum()))
    print(prepared.head())
    print("Pipeline completed successfully.")

    print("4/4: Create plots ...", flush=True)
    #Histograms
    # Select variables for a closer visual inspection.
    columns_histogram = ["price","bathrooms","bedrooms","floors","grade","last_known_change","sqft_living","sqft_lot"]
    prepared[columns_histogram].hist(bins=50, figsize=(20, 15))
    plt.savefig("plots/01_histograms.png", dpi=300, bbox_inches="tight")

    #Box plots
    fig = px.box(prepared, y="price", labels={"price": "House Price in $"})
    fig.write_image("plots/02_price_boxplot.png", width=800, height=600) #still resolving the issue

    # Plot scatterplots.
    grid = sns.pairplot(prepared[columns_histogram])
    grid.savefig("plots/03_pairplot.png", dpi=300)

    # Heatmap of the Pearson correlation coefficients
    numeric_kc_data = prepared.select_dtypes(include=["number"])
    mask = np.triu(numeric_kc_data.corr())
    plt.figure(figsize=(20, 15))
    ax = sns.heatmap(round(numeric_kc_data.corr(), 2), annot=True, mask=mask, cmap="RdBu_r")
    ax.figure.savefig("plots/04_correlation_heatmap.png", dpi=300, bbox_inches="tight")

    # Scatter plot: price versus distance to the center.
    scatter = sns.relplot(y="price", x="center_distance", data=prepared);
    scatter.savefig("plots/05_scatterplot.png", dpi=300, bbox_inches="tight");

    # Scatter plot: price per square foot versus distance to the center.
    sqft_scatter = sns.relplot(y="sqft_price", x="center_distance", data=prepared);
    sqft_scatter.savefig("plots/06_sqft_scatterplot.png", dpi=300, bbox_inches="tight");

    price_water =sns.relplot(y="price", x="water_distance", data=prepared);
    price_water.savefig("plots/07_price_water_distance.png", dpi=300, bbox_inches="tight");

    sqft_water = sns.relplot(y="sqft_price", x="water_distance", data=prepared);
    sqft_water.savefig("plots/08_sqft_water_distance.png", dpi=300, bbox_inches="tight");

if __name__ == "__main__":
    main()
# %%
