#!/bin/bash

python src/qwen35_hf_sql.py --dataset wikisql --constraint outlines --mode non-thinking --test-size 50
