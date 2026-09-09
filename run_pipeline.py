"""Die Datenaufbereitung für Pipeline."""

from pathlib import Path
from src.data_preparation import load_data, prepare_data



def main():
    project_dir = Path(__file__).resolve().parent
    csv_path = (
        project_dir / "data" / "King_County_House_prices_dataset.csv"
    )

    print("1/3: Rohdaten laden ...", flush=True)
    raw = load_data(csv_path)
    print("Rohdaten:", raw.shape)

    print("2/3: Daten aufbereiten; bitte warten ...", flush=True)
    prepared = prepare_data(raw)

    print("3/3: Ergebnis pruefen ...", flush=True)
    assert raw.shape == (21597, 21), "Rohdatenform weicht ab"
    assert prepared.isna().sum().sum() == 0, "Fehlende Werte"
    print("Ergebnis:", prepared.shape)
    print("Fehlende Werte:", int(prepared.isna().sum().sum()))
    print(prepared.head())
    print("Pipeline erfolgreich abgeschlossen.")


if __name__ == "__main__":
    main()