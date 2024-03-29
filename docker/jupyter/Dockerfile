FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

RUN apt-get update \
  && apt -y install bzip2 curl gcc git vim \
  && apt-get clean

ARG GID
ARG UID
ARG ML_USER
ARG WORKSPACE_FOLDER
ARG WORK_DIR="/home/$ML_USER"
ARG SAVED_MODELS_DIR="$WORK_DIR/$WORKSPACE_FOLDER/saved_models"
ARG CONDA_PATH="$WORK_DIR/miniconda3/bin"
ARG MINI_CONDA_SH="Miniconda3-py39_4.12.0-Linux-x86_64.sh"

RUN mkdir -p $SAVED_MODELS_DIR

RUN groupadd -g $GID $ML_USER && useradd -u $UID -g $GID -m $ML_USER && \
    chown -R $UID:$GID $WORK_DIR $SAVED_MODELS_DIR

USER $ML_USER
WORKDIR $WORK_DIR

RUN curl -O https://repo.anaconda.com/miniconda/$MINI_CONDA_SH \
    && chmod +x $MINI_CONDA_SH \
    && ./$MINI_CONDA_SH -b \
    && rm $MINI_CONDA_SH

ENV PATH $CONDA_PATH:$PATH

COPY conda.yml jupyter-config.sh ./

RUN conda env update -f conda.yml -n base \
    && rm conda.yml

RUN /bin/bash -c "jupyter contrib nbextension install --user"

RUN python -m nltk.downloader punkt \
    && python -m nltk.downloader stopwords \
    && python -m nltk.downloader wordnet

ENTRYPOINT sh jupyter-config.sh \
    && jupyter notebook --port=8888 --no-browser \
    --ip=0.0.0.0 --allow-root --NotebookApp.token=$NB_TOKEN \
    --NotebookApp.password=$NB_PASSWORD --user root
