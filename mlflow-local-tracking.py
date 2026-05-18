import os
import warnings
import sys
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow
import mlflow.sklearn

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


# Main program starts here
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Load trackling uri - Local or Remote here
    BASE_DIR = Path(__file__).resolve().parent
    db_uri = f"sqlite:///{BASE_DIR / 'mlflow.db'}"
    artifact_uri = (BASE_DIR / "mlruns").resolve().as_uri()
    mlflow.set_tracking_uri(db_uri)
    print("tracking uri:", mlflow.get_tracking_uri())
    print("artifact uri:", artifact_uri)

    # Set experiment metadata
    experiment_name = "wine-quality-experiment"
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        mlflow.create_experiment(
            name=experiment_name,
            artifact_location=artifact_uri
        )
    mlflow.set_experiment(experiment_name)


    # Load dataset
    wine_dataset = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv" # Wine Dataset
    try:
        data = pd.read_csv(wine_dataset, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )
        raise

    # Train, Test
    # Drop the existing output column 'quality' as we are going to recalculate it using the ML model and compare with the actual 'quality' column.
    train, test = train_test_split(data, random_state=42)
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train["quality"]
    test_y = test["quality"]

    # Hyper-parameters for Elasticnet ML algorithm
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5 # Hyper-parameter :Take from command line input or default to 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5 # Hyper-parameter :Take from command line input or default to 0.5


    with mlflow.start_run(): # Start every MLFlow expirment with this command.
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)
        rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)

        print(f"Elasticnet model (alpha={alpha:.6f}, l1_ratio={l1_ratio:.6f}):")
        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  R2: {r2}")

        # Log these params for MLFlow recording
        # mlflow.log_param() is used to track params via mlflow.
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        # mlfow-50-model will be the name of the model as seen on MFLOW UI (http:localhost:6001)
        #     
        mlflow.sklearn.log_model(sk_model=lr, name="mlflow-50-model-local") 