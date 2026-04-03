#!/bin/bash

MODEL="Qwen/Qwen3-0.6B"
BATCH=10   # CPU-only: keep batches small to avoid RAM pressure

# # ─── WikiSQL ──────────────────────────────────────────────────────────────────

echo "===== WikiSQL | none ====="
python src/nltosql_vllm.py --model $MODEL --dataset wikisql --constraint none --mode non-thinking --test-size 10 --batch-size $BATCH
python src/nltosql_vllm.py --model $MODEL --dataset wikisql --constraint none --mode thinking    --test-size 10 --batch-size $BATCH

echo "===== WikiSQL | outlines ====="
python src/nltosql_vllm.py --model $MODEL --dataset wikisql --constraint outlines --guided-decoding-backend outlines --mode non-thinking --test-size 10 --batch-size $BATCH
python src/nltosql_vllm.py --model $MODEL --dataset wikisql --constraint outlines --guided-decoding-backend outlines --mode thinking    --test-size 10 --batch-size $BATCH

echo "===== WikiSQL | xgrammar ====="
python src/nltosql_vllm.py --model $MODEL --dataset wikisql --constraint xgrammar --guided-decoding-backend xgrammar --mode non-thinking --test-size 10 --batch-size $BATCH
python src/nltosql_vllm.py --model $MODEL --dataset wikisql --constraint xgrammar --guided-decoding-backend xgrammar --mode thinking    --test-size 10 --batch-size $BATCH

# # ─── Spider ───────────────────────────────────────────────────────────────────

echo "===== Spider | none ====="
python src/nltosql_vllm.py --model $MODEL --dataset spider --constraint none --mode non-thinking --test-size 10 --batch-size $BATCH
python src/nltosql_vllm.py --model $MODEL --dataset spider --constraint none --mode thinking    --test-size 10 --batch-size $BATCH

echo "===== Spider | outlines ====="
python src/nltosql_vllm.py --model $MODEL --dataset spider --constraint outlines --guided-decoding-backend outlines --mode non-thinking --test-size 10 --batch-size $BATCH
python src/nltosql_vllm.py --model $MODEL --dataset spider --constraint outlines --guided-decoding-backend outlines --mode thinking    --test-size 10 --batch-size $BATCH

echo "===== Spider | xgrammar ====="
python src/nltosql_vllm.py --model $MODEL --dataset spider --constraint xgrammar --guided-decoding-backend xgrammar --mode non-thinking --test-size 10 --batch-size $BATCH
python src/nltosql_vllm.py --model $MODEL --dataset spider --constraint xgrammar --guided-decoding-backend xgrammar --mode thinking    --test-size 10 --batch-size $BATCH
