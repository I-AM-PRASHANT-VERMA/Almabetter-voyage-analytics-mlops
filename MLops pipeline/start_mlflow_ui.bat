@echo off
REM Launch the local MLflow UI against the repo's file-based tracking folder.
mlflow ui --backend-store-uri "%~dp0mlruns" --host 127.0.0.1 --port 5000
