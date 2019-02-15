#!/usr/bin/env bash

if [ -f ./submission.tar.gz ]; then
    echo "Removing submission.tar.gz..."
    rm ./submission.tar.gz
fi

docker image rm submission:0.0.1
docker build --tag submission:0.0.1 .
docker save submission:0.0.1 | gzip > submission.tar.gz
