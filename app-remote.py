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

from dotenv import load_dotenv
import dagshub

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

    # Load environment variables from .env file
    BASE_DIR = Path(__file__).resolve().parent
    load_dotenv(BASE_DIR / ".env")
    
    # Set DagsHub credentials from environment
    dagshub_key = os.getenv("DAGSHUB_KEY")
    if dagshub_key:
        os.environ["DAGSHUB_USER_TOKEN"] = dagshub_key
    
    # Configure MLflow tracking with DagsHub
    dagshub.init(
        repo_owner='ankur-israni',
        repo_name='50---mlflow-dagshub-and-bentoml', # This is the name of the repo on Dagshub.com. Daghshub has already connected to this repo.
        mlflow=True
    )

    # Load trackling uri - Local or Remote here
    db_uri = f"sqlite:///{BASE_DIR / 'mlflow.db'}"
    artifact_uri = (BASE_DIR / "mlruns").resolve().as_uri()

    # Load dataset
    wine_dataset = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv" # Dataset
    try:
        data = pd.read_csv(wine_dataset, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )
        raise

    # Train, Test
    train, test = train_test_split(data, random_state=42)
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train["quality"]
    test_y = test["quality"]

    # Hyper-parameters for Elasticnet ML algorithm
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    # Set experiment metadata
    experiment_name = "wine-quality-experiment"
    try:
        mlflow.create_experiment(name=experiment_name)
    except Exception:
        pass  # Experiment may already exist

    # Try to use DagsHub remote tracking, fallback to local if it fails
    try:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run():
            lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
            lr.fit(train_x, train_y)

            predicted_qualities = lr.predict(test_x)
            rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)

            print(f"Elasticnet model (alpha={alpha:.6f}, l1_ratio={l1_ratio:.6f}):")
            print(f"  RMSE: {rmse}")
            print(f"  MAE: {mae}")
            print(f"  R2: {r2}")

            # Log these params for MLFlow recording
            mlflow.log_param("alpha", alpha)
            mlflow.log_param("l1_ratio", l1_ratio)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)

            mlflow.sklearn.log_model(sk_model=lr, name="mlflow-50-model-remote")
    except Exception as e:
        logger.warning(f"DagsHub remote tracking failed: {e}. Falling back to local tracking.")
        # Fallback to local tracking
        mlflow.set_tracking_uri(db_uri)
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run():
            lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
            lr.fit(train_x, train_y)

            predicted_qualities = lr.predict(test_x)
            rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)

            print(f"Elasticnet model (alpha={alpha:.6f}, l1_ratio={l1_ratio:.6f}) [Local Tracking]:")
            print(f"  RMSE: {rmse}")
            print(f"  MAE: {mae}")
            print(f"  R2: {r2}")

            # Log these params for MLFlow recording
            mlflow.log_param("alpha", alpha)
            mlflow.log_param("l1_ratio", l1_ratio)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)

            mlflow.sklearn.log_model(sk_model=lr, name="mlflow-50-model-remote")