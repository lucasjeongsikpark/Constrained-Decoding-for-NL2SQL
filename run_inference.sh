#!/bin/bash

MODEL="Qwen/Qwen3-0.6B"
TEST_SIZE=3
BATCH_SIZE=3

# # ─── WikiSQL ──────────────────────────────────────────────────────────────────

# echo "===== WikiSQL | none ====="
# python src/inference/nltosql.py --model $MODEL --dataset wikisql --constraint none --mode non-thinking --test-size $TEST_SIZE --batch-size $BATCH_SIZE
# python src/inference/nltosql.py --model $MODEL --dataset wikisql --constraint none --mode thinking --test-size $TEST_SIZE --batch-size $BATCH_SIZE

# echo "===== WikiSQL | outlines ====="
# python src/inference/nltosql.py --model $MODEL --dataset wikisql --constraint outlines --mode non-thinking --test-size $TEST_SIZE --batch-size $BATCH_SIZE
# python src/inference/nltosql.py --model $MODEL --dataset wikisql --constraint outlines --mode thinking --test-size $TEST_SIZE --batch-size $BATCH_SIZE

# echo "===== WikiSQL | xgrammar ====="
# python src/inference/nltosql.py --model $MODEL --dataset wikisql --constraint xgrammar --mode non-thinking --test-size $TEST_SIZE --batch-size $BATCH_SIZE
# python src/inference/nltosql.py --model $MODEL --dataset wikisql --constraint xgrammar --mode thinking --test-size $TEST_SIZE --batch-size $BATCH_SIZE

# # ─── Spider ───────────────────────────────────────────────────────────────────

# echo "===== Spider | none ====="
# python src/inference/nltosql.py --model $MODEL --dataset spider --constraint none --mode non-thinking --test-size $TEST_SIZE --batch-size $BATCH_SIZE
# python src/inference/nltosql.py --model $MODEL --dataset spider --constraint none --mode thinking --test-size $TEST_SIZE --batch-size $BATCH_SIZE

echo "===== Spider | outlines ====="
python src/inference/nltosql.py --model $MODEL --dataset spider --constraint outlines --mode non-thinking --test-size $TEST_SIZE --batch-size $BATCH_SIZE
python src/inference/nltosql.py --model $MODEL --dataset spider --constraint outlines --mode thinking --test-size $TEST_SIZE --batch-size $BATCH_SIZE

# echo "===== Spider | xgrammar ====="
# python src/inference/nltosql.py --model $MODEL --dataset spider --constraint xgrammar --mode non-thinking --test-size $TEST_SIZE --batch-size $BATCH_SIZE
# python src/inference/nltosql.py --model $MODEL --dataset spider --constraint xgrammar --mode thinking --test-size $TEST_SIZE --batch-size $BATCH_SIZE
