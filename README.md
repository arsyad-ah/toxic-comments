# Introduction
Online discussions can be difficult in the digital age. The threat of abuse and harassment online may create a barrier for people to stop expressing themselves and give up seeking different opinions online. Platforms struggle to effectively facilitate conversations, leading many of them limiting or completely shutting down user comments.

source: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/overview

# Purpose
This repo aims to create an end-to-end ML pipeline covering data preparation, cleaning & preprocessing, model training & evaluation, and inference.

This repo has been dockerized and should run with other dependency repos/docker containers such as MLflow, postgres, MinIO. To setup the other repos/containers, follow the instructions [here](https://github.com/arsyad-ah/toxic-env.git).

# Pre-requisite
- Data can be found in this [link](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/overview) and placed in `data/` folder
- Embeddings can be found in this [link](https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip) and placed in `embeddings/` folder

# TODO List
The main parts of the pipelines in the repo has been completed. These are the WIPs:
- Change print statements to loggers
- Change / Upload data to read/write from/to DB (using DB image)
- Create a simple frontend for demo
- Create prerequisite files to create folders and download necessary files (e.g. glove embeddings, data)
- Pending mlflow integration with transformer flavour to load, save and log models - [link](https://github.com/mlflow/mlflow/pull/8086) (Completed with mlflow==2.4.1)

# Bugs
- BiLSTMClfTF model will break training pipeline if using the whole dataset in GPU. It is recommended to train a subset of the data if training on a GPU with small VRAM. A larger GPU is needed for training the entire dataset.
