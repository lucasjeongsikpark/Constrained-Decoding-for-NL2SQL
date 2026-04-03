# vLLM offline runner for NL2SQL

This runner uses **vLLM offline API (`LLM.generate`) only** and supports these backends through vLLM guided decoding:

- `none`
- `outlines`
- `lmfe`
- `xgrammar`

It is pinned for **`vllm==0.10.2`**.

## Important

Do **not** install standalone `outlines` in this environment. Use only the backend bundled through vLLM.

This is intended for a Linux GPU environment. macOS local testing is often not representative for real vLLM runs.

## Create environment

```bash
conda create -n nl2sql-vllm310 python=3.10 -y
conda activate nl2sql-vllm310
pip install --upgrade pip setuptools wheel
pip install -r src/requirements_vllm_0102.txt
```

## Quick test

```bash
python src/nltosql_vllm.py \
  --dataset wikisql \
  --constraint none \
  --mode non-thinking \
  --test-size 10
```

## Run with all 3 constrained backends

### Outlines
```bash
python src/nltosql_vllm.py \
  --dataset wikisql \
  --constraint outlines \
  --mode non-thinking \
  --test-size 10
```

### LMFE
```bash
python src/nltosql_vllm.py \
  --dataset wikisql \
  --constraint lmfe \
  --mode non-thinking \
  --test-size 10
```

### XGrammar
```bash
python src/nltosql_vllm.py \
  --dataset wikisql \
  --constraint xgrammar \
  --mode non-thinking \
  --test-size 10
```

## Spider example

```bash
python src/nltosql_vllm.py \
  --dataset spider \
  --constraint xgrammar \
  --mode non-thinking \
  --test-size 50 \
  --max-new-tokens 120
```

## Wrapper script

```bash
bash src/run_nltosql_vllm.sh wikisql none non-thinking 10
bash src/run_nltosql_vllm.sh wikisql outlines non-thinking 10
bash src/run_nltosql_vllm.sh wikisql lmfe non-thinking 10
bash src/run_nltosql_vllm.sh wikisql xgrammar non-thinking 10
```

## Output

Results are saved as:

```bash
result_{dataset}_{mode}_{train_mode}_{constraint}_vllm.json
```

Example:

```bash
result_wikisql_non-thinking_zero_xgrammar_vllm.json
```

## Notes

- `thinking` mode is left available, but for the most stable constrained-decoding runs start with `non-thinking`.
- `few` is kept in the CLI only for filename compatibility; this script currently uses the same prompt path as zero-shot.
- The original `src/nltosql.py` had an indentation bug around the HF xgrammar path. This runner avoids that whole HF path and uses vLLM only.
