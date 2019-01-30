#!/usr/bin/env bash

if [ -f ./submission.tar.gz ]; then
    echo "Removing submission.tar.gz..."
    docker image rm submission:0.0.1
    rm ./submission.tar.gz
fi

docker build --tag submission:0.0.1 .
docker save submission:0.0.1 | gzip > submission.tar.gz
