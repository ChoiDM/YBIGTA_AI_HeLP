#!/usr/bin/env bash

docker build --tag submission:0.0.1 .

echo "Compressing..."

docker save submission:0.0.1 | gzip > submission.tar.gz
