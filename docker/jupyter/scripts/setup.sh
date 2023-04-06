#!/bin/bash
filename=glove.6B.zip
embeddings_dir=../embeddings

curl https://nlp.stanford.edu/data/${filename} -O $embeddings_dir/$filename
unzip $embeddings_dir/$filename -d .
