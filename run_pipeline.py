"""Data preparation for the pipeline."""

from pathlib import Path
from src.data_preparation import load_data, prepare_data



def main():
    project_dir = Path(__file__).resolve().parent
    csv_path = (
        project_dir / "data" / "King_County_House_prices_dataset.csv"
    )

    print("1/3: Load raw data ...", flush=True)
    raw = load_data(csv_path)
    print("Raw data:", raw.shape)

    print("2/3: Process data; please wait ...", flush=True)
    prepared = prepare_data(raw)

    print("3/3: Check results ...", flush=True)
    assert raw.shape == (21597, 21), "Raw data shape is incorrect"
    assert prepared.isna().sum().sum() == 0, "Missing values found in prepared data"
    print("Results:", prepared.shape)
    print("Missing Values:", int(prepared.isna().sum().sum()))
    print(prepared.head())
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()