import os
import sys
import warnings
import logging
from pathlib import Path

import dagshub
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    BASE_DIR = Path(__file__).resolve().parent
    load_dotenv(BASE_DIR / ".env")

    DAGSHUB_USERNAME = "ankur-israni"
    DAGSHUB_REPO_NAME = "50---mlflow-dagshub-and-bentoml"
    DAGSHUB_TOKEN = os.getenv("DAGSHUB_KEY")

    if not DAGSHUB_TOKEN:
        raise ValueError("DAGSHUB_KEY not found in .env file")

    # Clear old MLflow/DagsHub environment values
    for key in [
        "MLFLOW_TRACKING_URI",
        "MLFLOW_TRACKING_USERNAME",
        "MLFLOW_TRACKING_PASSWORD",
        "MLFLOW_EXPERIMENT_NAME",
        "MLFLOW_RUN_ID",
        "DAGSHUB_USER_TOKEN",
    ]:
        os.environ.pop(key, None)

    os.environ["DAGSHUB_USER_TOKEN"] = DAGSHUB_TOKEN

    dagshub.init(
        repo_owner=DAGSHUB_USERNAME,
        repo_name=DAGSHUB_REPO_NAME,
        mlflow=True,
    )

    print("MLflow version:", mlflow.__version__)
    print("MLflow tracking URI:", mlflow.get_tracking_uri())

    # Use DagsHub Default experiment first to avoid remote experiment creation issues
    mlflow.set_experiment("Default")

    wine_dataset = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/"
        "winequality-red.csv"
    )

    try:
        data = pd.read_csv(wine_dataset, sep=";")
    except Exception as e:
        logger.exception("Unable to download CSV. Error: %s", e)
        raise

    train, test = train_test_split(data, random_state=42)

    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train["quality"]
    test_y = test["quality"]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    # Do not pass run_name here. Let DagsHub/MLflow create the run cleanly.
    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)
        rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)

        print(f"ElasticNet model alpha={alpha}, l1_ratio={l1_ratio}")
        print(f"RMSE: {rmse}")
        print(f"MAE: {mae}")
        print(f"R2: {r2}")

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        mlflow.sklearn.log_model(
            sk_model=lr,
            name="mlflow_50_model_remote"
        )

    print("Successfully logged run to DagsHub.")