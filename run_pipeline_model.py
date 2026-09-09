"""Die Datenaufbereitung ohne Notebook starten. Modelle vergleichen. Ergebnisse speichern"""

from datetime import datetime
from pathlib import Path

import pandas as pd

from src.data_preparation import load_data, prepare_data
from src.modeling import (
    select_features_target, split_data, train_linear_model,
    evaluate_model, train_polynomial_model, train_elastic_model,
    build_error_table, save_error_map, save_model,
)

def main() -> None:
    project_dir = Path(__file__).resolve().parent
    csv_path = (
        project_dir / "data" / "King_County_House_prices_dataset.csv"
    )
    print("1/7: Daten laden und aufbereiten ...", flush=True)
    raw = load_data(csv_path)
    prepared = prepare_data(raw)
    assert not prepared.empty, "Keine Daten uebrig"
    assert prepared.isna().sum().sum() == 0, "Fehlende Werte"
    ratio = prepared["bathrooms"] / prepared["bedrooms"]
    assert ratio.between(0.10, 2, inclusive="neither").all()
    print("Rohdaten:", raw.shape, "Aufbereitet:", prepared.shape)

    print("2/7: Features und Train/Test-Split ...", flush=True)
    X, y = select_features_target(prepared)
    assert "price" not in X and "sqft_price" not in X
    X_train, X_test, y_train, y_test = split_data(X, y)
    print("Train:", X_train.shape, "Test:", X_test.shape)
    scores = []

    print("3/7: Zwei lineare Baselines trainieren ...", flush=True)
    for features in [["grade"], ["grade", "last_known_change"]]:
        model = train_linear_model(X_train, y_train, features)
        metrics = evaluate_model(
            model, X_test[features], y_test, len(features)
        )
        scores.append({"model": "linear: " + "+".join(features), **metrics})

    print("4/7: Polynommodell trainieren ...", flush=True)
    X_train_model = X_train.drop(columns=["id"])
    X_test_model = X_test.drop(columns=["id"])
    polynomial = train_polynomial_model(X_train_model, y_train)
    p = polynomial.named_steps["polynomial"].n_output_features_
    metrics = evaluate_model(polynomial, X_test_model, y_test, p)
    scores.append({"model": "polynomial", **metrics})
    polynomial_errors = build_error_table(
        X_test, y_test, polynomial.predict(X_test_model)
    )

    print("5/7: ElasticNet-Suche; bitte warten ...", flush=True)
    search = train_elastic_model(X_train_model, y_train)
    elastic = search.best_estimator_
    print("Beste Parameter:", search.best_params_)
    print("Mittleres CV-R2:", round(search.best_score_, 3))
    metrics = evaluate_model(elastic, X_test_model, y_test)
    scores.append({"model": "elastic", **metrics})
    print("Erste Koeffizienten:")
    print(elastic.named_steps["model"].regressor_.coef_[:5])

    print("6/7: Fehler und Modellvergleich ...", flush=True)
    elastic_errors = build_error_table(
        X_test, y_test, elastic.predict(X_test_model)
    )
    summary = pd.DataFrame(scores)
    print(summary.round(3).to_string(index=False))
    for name, errors in [
        ("polynomial", polynomial_errors), ("elastic", elastic_errors)
    ]:
        print("Groesste relative Ueberschaetzung:", name)
        print(errors.loc[[errors["price_difference_percent"].idxmax()]])

    print("7/7: Ergebnisse speichern ...", flush=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = project_dir / "model" / ("training_" + stamp)
    output_dir.mkdir(parents=True, exist_ok=False)
    summary.to_csv(output_dir / "scores.csv", index=False)
    for name, errors in [
        ("polynomial", polynomial_errors), ("elastic", elastic_errors)
    ]:
        errors.to_csv(output_dir / (name + "_errors.csv"), index=False)
        save_error_map(errors, output_dir / (name + "_map.html"))
    save_model(elastic, output_dir / "model.bin")
    pd.Series(X_train_model.columns, name="feature").to_csv(
        output_dir / "feature_columns.csv", index=False
    )

    print("Training abgeschlossen. Ergebnisse:", output_dir)


if __name__ == "__main__":
    main()