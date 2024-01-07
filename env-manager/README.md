# toxic-env

## Prerequisites
Please ensure you have these installed before running the repo.
- [docker](https://docs.docker.com/desktop/)
- [docker compose](https://docs.docker.com/compose/install/) 
- [nvidia-container-toolkit](https://www.server-world.info/en/note?os=Ubuntu_20.04&p=nvidia&f=2)

## To run
1. Copy a version of `.env-sample` file and rename to `.env` file.
2. Fill up the missing environment variables in the file. More details of the variables are listed below.
3. To start containers, run `docker-compose up -d --build`.

## Notes
5 containers will be created.
- jupyter: Jupyter notebook and the source codes from toxic-comments repos will be found here
- mlflow: Experiments (metrics, training parameters, etc.)from the train pipeline will be available here
- postgres: DB to store mlflow experiment parameters
- minio/server: Acts as bucket to store experiment artifacts
- minio/mc: Client to initialize minio/server

## Environment variables
```
MLFLOW_NAME: MLflow container name. Default: mlflow
MLFLOW_PORT: MLflow container port. Default: 5000
MLFLOW_DB: MLflow DB name.

POSTGRES_USER: DB name for MLflow. Default: mlflow
POSTGRES_PASSWORD: DB password
POSTGRES_NAME: DB container name. Default: postgres
POSTGRES_PORT: DB container port. Default: 5432

ML_USER: Jupyter notebook container user. Default: user
NB_PORT: Jupyter notebook container port. Default: 8880
NB_NAME: Jupyter notebook container name. Default: jupyter
NB_TOKEN: Jupyter notebook token
NB_PASSWORD: Jupyter notebook password

WORKSPACE_FOLDER: Jupyter notebook workspace folder. This will contain all the relevant scripts. Default: workspace

UID: User UID for Jupyter notebook
GID: User GID for Jupyter notebook

MINIO_SERVER_NAME: MinIO server container name. Default: minio-server
MINIO_CLIENT_NAME: MinIO client container name. Default: minio-client
MLFLOW_BUCKET_NAME: MLflow bucket name in MinIO. Default: mlflow
MINIO_PORT: MinIO container port. Default: 9000
AWS_ACCESS_KEY_ID: MinIO container access ID
AWS_SECRET_ACCESS_KEY: MinIO container access key
```