#!/usr/bin/env bash

if [ -f ./submission.tar.gz ]; then
    echo "Removing submission.tar.gz..."
    focker image rm submission:0.0.1
    rm -rf ./submission.tar.gz
fi

docker build --tag submission:0.0.1 .
echo "Build is finished... Start zip file"

docker save submission:0.0.1 | gzip > submission.tar.gz
