#!/bin/bash

FILENAME=glove.6B.zip
EMBEDDINGS_DIR=../embeddings

mkdir -p $EMBEDDINGS_DIR
curl https://downloads.cs.stanford.edu/nlp/data/${FILENAME} -O ${EMBEDDINGS_DIR}/${FILENAME}
unzip ${EMBEDDINGS_DIR}/${FILENAME} -d .
