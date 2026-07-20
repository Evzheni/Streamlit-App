from __future__ import annotations
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from numpy.typing import ArrayLike
from joblib import dump


def train_regression_model(X_train: ArrayLike, y_train: ArrayLike) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def save_regression_model(model: LinearRegression, filename: str = "linear_regression_model.joblib"):
    dump(model, filename)

def evaluate_regression_model(model: LinearRegression, X_test: ArrayLike, y_test: ArrayLike):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")

def save_initial_datasets(X: ArrayLike, y: ArrayLike):
    X_filename = "X.joblib"
    y_filename = "y.joblib"
    dump(X, X_filename)
    dump(y, y_filename)

if __name__ == '__main__':
    # Generate a dataset
    X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)

    # Scale feature X to range -3..3
    X = np.interp(X, (X.min(), X.max()), (-3, 3))

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a linear regression model
    model = train_regression_model(X_train, y_train)

    # Evaluate the model
    evaluate_regression_model(model, X_test, y_test)

    # Save the model
    save_regression_model(model)

    # Save datasets
    save_initial_datasets(X, y)