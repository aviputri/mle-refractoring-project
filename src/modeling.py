"""Modelltraining fuer die bereits aufbereiteten King-County-Daten."""

from pathlib import Path
import pandas as pd
import plotly.express as px
import skops.io as sio
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler



def select_features_target(df: pd.DataFrame):
    drop_columns = [
    "price", "sqft_price", "date", "delta_lat", "delta_long"
    ]
    X = df.drop(columns=drop_columns).copy()
    y = df["price"].copy()
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series):
    return train_test_split(
    X, y, test_size=0.3, random_state=42
    )


def train_linear_model(X_train, y_train, features):
    model = LinearRegression()
    model.fit(X_train[features], y_train)
    return model


def evaluate_model(model, X_test, y_test, n_predictors=None):
    r2 = model.score(X_test, y_test)
    result = {"r2": r2}
    if n_predictors is not None:
        n = len(y_test)
        if n <= n_predictors + 1:
            raise ValueError("Zu wenige Testzeilen fuer adjusted R2.")
        result["adjusted_r2"] = (
            1 - (1 - r2) * (n - 1) / (n - n_predictors - 1)
        )
    return result


def train_polynomial_model(X_train, y_train):
    model = Pipeline([
        ("polynomial", PolynomialFeatures(2)),
        ("regression", LinearRegression()),
    ])
    model.fit(X_train, y_train)
    return model


def train_elastic_model(X_train, y_train):
    param_grid = {
        "model__regressor__alpha": [0.01, 0.1, 1],
        "model__regressor__l1_ratio": [0.2, 0.5, 0.8],
    }
    model = Pipeline([
        ("polynomial", PolynomialFeatures(2, include_bias=False)),
        ("scaler", StandardScaler()),
        (
            "model",
            TransformedTargetRegressor(
                regressor=ElasticNet(
                    max_iter=5000, tol=1e-4, precompute=True
                ),
                transformer=StandardScaler(),
            ),
        ),
    ])
    search = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring="r2",
        verbose=2,
        n_jobs=1,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    return search


def build_error_table(X_test, y_test, predictions):
    errors = pd.DataFrame({
        "price": y_test,
        "latitude": X_test["lat"],
        "longitude": X_test["long"],
        "id": X_test["id"],
    }).reset_index(drop=True)
    errors["price_prediction"] = predictions.round(2)
    errors["price_difference"] = (
        errors["price_prediction"] - errors["price"]
    ).round(2)
    errors["price_difference_percent"] = (
        errors["price_difference"] / errors["price"] * 100
    ).round(2)
    return errors


def save_error_map(errors: pd.DataFrame, path: Path):
    fig = px.scatter_map(
        errors,
        lat="latitude",
        lon="longitude",
        hover_data=["price", "price_prediction", "id"],
        color="price_difference_percent",
        color_continuous_scale=["green", "yellow", "red"],
        zoom=7.7,
        height=400,
    )
    fig.update_layout(map_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.write_html(str(path), include_plotlyjs=True, auto_open=False)


def save_model(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        sio.dump(model, file)