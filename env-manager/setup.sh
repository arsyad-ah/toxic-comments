#!/bin/bash

REPO_BASE=https://github.com/arsyad-ah
REPOS=(mlflow)

root_path=${PWD}/../..
cd $root_path

for repo in ${REPOS[@]}
do 
    repo_to_clone=$REPO_BASE/$repo
    echo $PWD
    git clone $repo_to_clone
done
