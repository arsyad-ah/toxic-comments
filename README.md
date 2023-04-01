# Introduction
Online discussions can be difficult in the digital age. The threat of abuse and harassment online may create a barrier for people to stop expressing themselves and give up seeking different opinions online. Platforms struggle to effectively facilitate conversations, leading many of them limiting or completely shutting down user comments.

source: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/overview

# TODO List

- Change print statements to loggers
- Change / Upload data to read/write from/to DB (using DB image)
- Create a simple frontend for easy use to train / inference
- Create prerequisite files to create folders and download necessary files (e.g. glove embeddings, data)
- Pending mlflow integration with transformer flavour to load, save and log models - [link](https://github.com/mlflow/mlflow/pull/8086)

# Bugs
- BiLSTM model will break training pipeline if using the whole dataset. Need to relook into dataloader implementation