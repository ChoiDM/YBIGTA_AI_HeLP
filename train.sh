#!/usr/bin/env bash

pip install keras
pip install tensorflow

source activate ai_help && python src/train.py

