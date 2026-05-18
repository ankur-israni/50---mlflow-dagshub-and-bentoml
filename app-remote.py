import os
import warnings
import sys
import logging
from pathlib import Path
import dagshub

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
    if not dagshub_key:
        raise ValueError("DAGSHUB_KEY not found in .env file. Please set your DagsHub API token.")
    
    # Set credentials for DagsHub authentication
    # os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/ankur-israni/50---mlflow-dagshub-and-bentoml.mlflow"
    # os.environ["DAGSHUB_USER_TOKEN"] = dagshub_key
    # os.environ["MLFLOW_TRACKING_USERNAME"] = "ankur-israni"
    # os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_key
    os.environ["DAGSHUB_USER_TOKEN"] = os.getenv("DAGSHUB_KEY")

    # Configure MLflow tracking with DagsHub
    print("Initializing DagsHub MLflow tracking...")
    dagshub.init(repo_owner='ankur-israni', repo_name='50---mlflow-dagshub-and-bentoml', mlflow=True)
    
    # Set the tracking URI with embedded credentials for better compatibility
    # tracking_uri = f"https://ankur-israni:{dagshub_key}@dagshub.com/ankur-israni/50---mlflow-dagshub-and-bentoml.mlflow"
    tracking_uri = "https://dagshub.com/ankur-israni/50---mlflow-dagshub-and-bentoml.mlflow"
    mlflow.set_tracking_uri(tracking_uri)

        # remote_server_uri="https://dagshub.com/krishnaik06/mlflowexperiments.mlflow"
        # mlflow.set_tracking_uri(remote_server_uri)

    
    
    # Verify tracking URI is set correctly
    print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")

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
    mlflow.set_experiment(experiment_name)

    # Train and log with DagsHub remote tracking (will fail if remote tracking unavailable)
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

        # mlflow.sklearn.log_model(sk_model=lr, artifact_path="model")
        mlflow.sklearn.log_model(sk_model=lr, name="model")
    