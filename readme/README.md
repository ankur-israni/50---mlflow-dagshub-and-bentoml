************************************
# 1) Local Run 
************************************
# Create the Vitutal Environment
conda create -p venv_mlops python==3.12

# Start the Virtual Environment
Terminal (/Users/ankur/backup/delta/sn/datascience/workspace) >
conda activate venv_mlops/

# Install dependencies
Terminal ('/Users/ankur/backup/delta/sn/datascience/workspace/11VC_udemy - complete datascience ml_dl_nlp bootcamp/50 - mlflow dagshub and bentoml/resources') > 
pip install -r resources/requirements.txt

# Run the python script
Terminal ('/Users/ankur/backup/delta/sn/datascience/workspace/11VC_udemy - complete datascience ml_dl_nlp bootcamp/50 - mlflow dagshub and bentoml') > 
python app-local.py

# Start the MLFlow UI (using 'mlflow.db' file as backend-store)
Terminal ('/Users/ankur/backup/delta/sn/datascience/workspace/11VC_udemy - complete datascience ml_dl_nlp bootcamp/50 - mlflow dagshub and bentoml') > 
mlflow server --backend-store-uri sqlite:///mlflow.db --port 6001 (Deprecated)
OR
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 6001 (Preferred)


# Start the MLFlow UI (using 'mlruns' folder as the backend-store) - This is deprecated as of Feb 2026
Terminal ('/Users/ankur/backup/delta/sn/datascience/workspace/11VC_udemy - complete datascience ml_dl_nlp bootcamp/50 - mlflow dagshub and bentoml') > 
mlflow server --backend-store-uri ./mlruns --port 6001


# View MLFlow on Browser
Browser > http://localhost:6001


# Stop the Virtual Environment
------------------------------------------------
Terminal (/Users/ankur/backup/delta/sn/datascience/workspace) >
conda deactivate

************************************
# 2) Remote Run (Dagshub) - TODO
************************************

## ML FLow experiements
MLFLOW_TRACKING_URI=https://dagshub.com/krishnaik06/mlflowexperiments.mlflow \
MLFLOW_TRACKING_USERNAME=krishnaik06 \
MLFLOW_TRACKING_PASSWORD=7104284f1bb44ece21e0e2adb4e36a250ae3251f \
python script.py


# 3) Git Repository
------------------------------------------------
This folder '50 - mflow dagshub and bemtoml' has its own .git repository
Remote URL = 
git@github.com:ankur-israni/50---mlflow-dagshub-and-bentoml.git 
OR
https://github.com/ankur-israni/50---mlflow-dagshub-and-bentoml.git

# 4) Querying SQLLite > mlflow.db 
------------------------------------------------
Terminal ('/Users/ankur/backup/delta/sn/datascience/workspace/11VC_udemy - complete datascience ml_dl_nlp bootcamp/50 - mlflow dagshub and bentoml') > 
 - sqlite3 mlflow.db
 - select * from runs
 - select * from experiments;
 - SELECT * FROM metrics WHERE key = 'rmse';
 - SELECT r.run_uuid, m.value FROM runs r JOIN metrics m ON r.run_uuid = m.run_uuid WHERE m.key = 'rmse' ORDER BY m.value ASC;


