#!/bin/bash

MODEL="Qwen/Qwen3-0.6B"
TEST_SIZE=150
BATCH_SIZE=3
N_SHOTS=1

# # # ─── WikiSQL | zero-shot ──────────────────────────────────────────────────────

echo "===== WikiSQL | zero | none ====="
python src/inference/nltosql.py --model $MODEL --dataset wikisql --constraint none --mode non-thinking --train-mode zero --test-size $TEST_SIZE --batch-size $BATCH_SIZE
python src/inference/nltosql.py --model $MODEL --dataset wikisql --constraint none --mode thinking --train-mode zero --test-size $TEST_SIZE --batch-size $BATCH_SIZE

echo "===== WikiSQL | zero | outlines ====="
python src/inference/nltosql.py --model $MODEL --dataset wikisql --constraint outlines --mode non-thinking --train-mode zero --test-size $TEST_SIZE --batch-size $BATCH_SIZE
python src/inference/nltosql.py --model $MODEL --dataset wikisql --constraint outlines --mode thinking --train-mode zero --test-size $TEST_SIZE --batch-size $BATCH_SIZE

echo "===== WikiSQL | zero | xgrammar ====="
python src/inference/nltosql.py --model $MODEL --dataset wikisql --constraint xgrammar --mode non-thinking --train-mode zero --test-size $TEST_SIZE --batch-size $BATCH_SIZE
python src/inference/nltosql.py --model $MODEL --dataset wikisql --constraint xgrammar --mode thinking --train-mode zero --test-size $TEST_SIZE --batch-size $BATCH_SIZE

# # # ─── WikiSQL | few-shot ───────────────────────────────────────────────────────

echo "===== WikiSQL | few | none ====="
python src/inference/nltosql.py --model $MODEL --dataset wikisql --constraint none --mode non-thinking --train-mode few --n-shots $N_SHOTS --test-size $TEST_SIZE --batch-size $BATCH_SIZE
python src/inference/nltosql.py --model $MODEL --dataset wikisql --constraint none --mode thinking --train-mode few --n-shots $N_SHOTS --test-size $TEST_SIZE --batch-size $BATCH_SIZE

echo "===== WikiSQL | few | outlines ====="
python src/inference/nltosql.py --model $MODEL --dataset wikisql --constraint outlines --mode non-thinking --train-mode few --n-shots $N_SHOTS --test-size $TEST_SIZE --batch-size $BATCH_SIZE
python src/inference/nltosql.py --model $MODEL --dataset wikisql --constraint outlines --mode thinking --train-mode few --n-shots $N_SHOTS --test-size $TEST_SIZE --batch-size $BATCH_SIZE

echo "===== WikiSQL | few | xgrammar ====="
python src/inference/nltosql.py --model $MODEL --dataset wikisql --constraint xgrammar --mode non-thinking --train-mode few --n-shots $N_SHOTS --test-size $TEST_SIZE --batch-size $BATCH_SIZE
python src/inference/nltosql.py --model $MODEL --dataset wikisql --constraint xgrammar --mode thinking --train-mode few --n-shots $N_SHOTS --test-size $TEST_SIZE --batch-size $BATCH_SIZE


# # ─── Spider | few-shot ────────────────────────────────────────────────────────

echo "===== Spider | few | outlines ====="
python src/inference/nltosql.py --model $MODEL --dataset spider --constraint outlines --mode non-thinking --train-mode few --n-shots $N_SHOTS --test-size $TEST_SIZE --batch-size $BATCH_SIZE --save-interval 10
python src/inference/nltosql.py --model $MODEL --dataset spider --constraint outlines --mode thinking --train-mode few --n-shots $N_SHOTS --test-size $TEST_SIZE --batch-size $BATCH_SIZE --save-interval 10

echo "===== Spider | few | xgrammar ====="
python src/inference/nltosql.py --model $MODEL --dataset spider --constraint xgrammar --mode non-thinking --train-mode few --n-shots $N_SHOTS --test-size $TEST_SIZE --batch-size $BATCH_SIZE
python src/inference/nltosql.py --model $MODEL --dataset spider --constraint xgrammar --mode thinking --train-mode few --n-shots $N_SHOTS --test-size $TEST_SIZE --batch-size $BATCH_SIZE

echo "===== Spider | few | none ====="
python src/inference/nltosql.py --model $MODEL --dataset spider --constraint none --mode non-thinking --train-mode few --n-shots $N_SHOTS --test-size $TEST_SIZE --batch-size $BATCH_SIZE
python src/inference/nltosql.py --model $MODEL --dataset spider --constraint none --mode thinking --train-mode few --n-shots $N_SHOTS --test-size $TEST_SIZE --batch-size $BATCH_SIZE

# ─── Spider | zero-shot ───────────────────────────────────────────────────────

echo "===== Spider | zero | outlines ====="
python src/inference/nltosql.py --model $MODEL --dataset spider --constraint outlines --mode non-thinking --train-mode zero --test-size $TEST_SIZE --batch-size $BATCH_SIZE --save-interval 10
python src/inference/nltosql.py --model $MODEL --dataset spider --constraint outlines --mode thinking --train-mode zero --test-size $TEST_SIZE --batch-size $BATCH_SIZE --save-interval 10

echo "===== Spider | zero | xgrammar ====="
python src/inference/nltosql.py --model $MODEL --dataset spider --constraint xgrammar --mode non-thinking --train-mode zero --test-size $TEST_SIZE --batch-size $BATCH_SIZE
python src/inference/nltosql.py --model $MODEL --dataset spider --constraint xgrammar --mode thinking --train-mode zero --test-size $TEST_SIZE --batch-size $BATCH_SIZE

echo "===== Spider | zero | none ====="
python src/inference/nltosql.py --model $MODEL --dataset spider --constraint none --mode non-thinking --train-mode zero --test-size $TEST_SIZE --batch-size $BATCH_SIZE
python src/inference/nltosql.py --model $MODEL --dataset spider --constraint none --mode thinking --train-mode zero --test-size $TEST_SIZE --batch-size $BATCH_SIZE
