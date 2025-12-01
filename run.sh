#!/usr/bin/env bash


echo "Running Machine Learning pipeline..."

# 1. Train and fine-tune models
python3 src/train.py

# 2. Generate model reports and comparison
python3 src/report.py

echo "Done! All modules executed."
