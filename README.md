# Constrained Decoding for Natural Language to SQL.

Code, processed data, model predictions, and evaluation outputs for **Constrained Decoding for Robust NL-to-SQL**. The [final report](reports/Constrained%20Decoding%20for%20NL2SQL.pdf) has the full method, metrics, and error analysis.

Free-form LLM decoding often produces SQL that fails to parse, references columns that don't exist, or quietly returns the wrong rows. We study constrained decoding as an inference-time fix that needs no training, and ask three questions:

1. Does constrained decoding reduce invalid and non-executable SQL?
2. Does reasoning ("thinking") mode help, hurt, or interact with grammar enforcement?
3. Do constraints designed for a narrow single-table schema transfer to open, multi-table databases?

We run [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) with greedy decoding under three decoders:

| Decoder    | What constrains the output                                                                                                                            |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `none`     | Nothing (baseline)                                                                                                                                    |
| `outlines` | A regular expression, compiled by [Outlines](https://github.com/dottxt-ai/outlines) into a token-level finite-state machine                           |
| `xgrammar` | A per-question EBNF grammar whose table and column names come from that question's schema, compiled by [XGrammar](https://github.com/mlc-ai/xgrammar) |

The grammars were designed on WikiSQL (one table per question), then transferred to Spider 1.0 (multi-table, unseen databases) with only the additions needed for joins, `ORDER BY`, and `LIMIT`. Every decoder runs in thinking and non-thinking mode, zero-shot and 1-shot, on the same seeded random sample of each test split: 300 WikiSQL questions and 150 Spider questions. That is 24 runs in total.

## Results

Execution accuracy (EA, %): a prediction counts as correct when it returns the same rows as the gold query on the dataset's SQLite database, ignoring row order. The best decoder in each row is in bold.

**WikiSQL** (300 test questions)

| Mode         | Prompt    |   `none` | `outlines` | `xgrammar` |
| ------------ | --------- | -------: | ---------: | ---------: |
| non-thinking | zero-shot |     39.3 |       36.0 |   **45.3** |
| non-thinking | 1-shot    |     62.7 |       49.3 |   **66.7** |
| thinking     | zero-shot | **55.0** |       42.7 |       54.0 |
| thinking     | 1-shot    |     66.3 |       50.0 |   **68.3** |

**Spider 1.0** (150 test questions)

| Mode         | Prompt    |   `none` | `outlines` | `xgrammar` |
| ------------ | --------- | -------: | ---------: | ---------: |
| non-thinking | zero-shot | **28.7** |       10.7 |        8.7 |
| non-thinking | 1-shot    | **30.7** |       11.3 |       12.0 |
| thinking     | zero-shot | **43.3** |        9.3 |       13.3 |
| thinking     | 1-shot    | **35.3** |       12.7 |       12.0 |

> [!NOTE]
> Spider EA here is 2–9 points higher than in the final report's Table 4 (for example, 43.3 instead of 36.0 for thinking, zero-shot, `none`). The report's Spider EA was computed while 8 of the 40 Spider test databases were unavailable, so the 24 sampled questions from those databases could never be scored as correct. The table above scores the same predictions with all databases present. WikiSQL numbers match the report exactly, and the report's trends and conclusions do not change.

What the numbers show:

- **Whether constraints help depends on the dataset.** On WikiSQL, XGrammar has the best EA in three of four settings (1 point behind `none` in the fourth) and the highest valid-SQL and executable rates, up to 98.7% (report Table 3). On Spider, both constrained decoders lower EA by 18–34 points. In 103 of the 150 sampled Spider questions (69%), the gold query uses a construct neither grammar can produce: `GROUP BY`, `HAVING`, a nested query, a set operation, a table alias, or more than one `SELECT` column.
- **Outlines is below the unconstrained baseline in every setting, and below XGrammar on WikiSQL.** Its regex only approximates the XGrammar grammar. On WikiSQL it accepts any identifier of the form `[A-Za-z_][A-Za-z0-9_ ]*` instead of the question's real column names, so in 69 of the 300 sampled questions (23%) it cannot spell the column the gold query uses (for example `Pick #` or `U.S. airdate`).
- **Thinking mode stays compatible with constraints**, because both the regex and the EBNF accept an optional `<think>…</think>` block before the SQL. On WikiSQL, thinking beats non-thinking under both decoders, and thinking + XGrammar + 1-shot is the best result overall (68.3%). The cost is 14–25× more generated tokens per question (about 470–1,260 versus 23–70 on average).
- **One retrieved example helps on WikiSQL but not on Spider.** A single TF-IDF-retrieved training example adds 7–23 EA points on WikiSQL. Spider's training and test databases don't overlap, so the retrieved example (shown without its schema) always comes from a different database, and the effect ranges from −8 to +3 points.

With 150–300 questions per run, differences of a few points are within sampling noise. The report also covers valid-SQL rate, executable rate, latency and throughput, and error analysis.

## Repository layout

```text
.
├── run_inference.sh             # the 24-run sweep
├── requirements.txt
├── src/
│   ├── inference/               # nltosql.py (main script) and prototype notebooks
│   ├── data_processing/         # builds the processed WikiSQL and Spider splits
│   ├── evaluation/              # execution accuracy: results/ → output/
│   └── arxiv/                   # archived scripts and notes
├── data/
│   ├── wikisql_processed/       # processed WikiSQL splits
│   ├── spider_data_processed/   # processed Spider splits
│   ├── data/                    # raw WikiSQL + SQLite DBs (downloaded, not committed)
│   └── spider_data/             # raw Spider + SQLite DBs (downloaded, not committed)
├── results/                     # predictions for all 24 runs
├── logs/                        # one log per run
├── output/                      # scored predictions from src/evaluation (generated)
├── evaluation/
│   ├── scripts/                 # executable rate, error analysis, pairwise comparison, plots
│   └── output/                  # scored predictions, comparison and error reports, figures
├── reports/                     # final report (PDF)
└── arxiv/                       # archived notebooks (Outlines, XGrammar, LMFE)
```

## Setup

```bash
git clone https://github.com/lucasjeongsikpark/CSCI566-GroupProject.git
cd CSCI566-GroupProject

conda create -n nl2sql-vllm310 python=3.10 -y
conda activate nl2sql-vllm310
pip install -r requirements.txt
```

The first run downloads `Qwen/Qwen3-0.6B` from the Hugging Face Hub; later runs load it from the local cache. The model is loaded with `device_map="auto"`. The report's runs used an Apple M1 Pro (32 GB) with PyTorch's `mps` backend, and the full sweep took about 33 hours, most of it in thinking mode.

## Data

The processed splits used for inference are committed:

| Dataset                                    |  Train |   Val |   Test |
| ------------------------------------------ | -----: | ----: | -----: |
| WikiSQL (`data/wikisql_processed/`)        | 56,355 | 8,421 | 15,878 |
| Spider 1.0 (`data/spider_data_processed/`) |  8,659 | 1,034 |  2,147 |

Spider's train split combines `train_spider.json` (7,000) and `train_others.json` (1,659) and is the pool for few-shot retrieval.

Execution accuracy needs the original SQLite databases, which are not committed:

- **WikiSQL:** run `python src/data_processing/preprocess_wikisql.py --raw-dir data`. It downloads the official [WikiSQL](https://github.com/salesforce/WikiSQL) archive, extracts it to `data/data/` (including `test.db`), and rebuilds `data/wikisql_processed/`.
- **Spider 1.0:** download the dataset from the [Spider site](https://yale-lily.github.io/spider) and extract it so that `data/spider_data/test_database/<db_id>/<db_id>.sqlite` exists for all 40 test databases. The same folder is the input to `src/data_processing/preprocess_spider.py` if you want to rebuild the processed splits.

## Run inference

Run the full sweep from the repository root:

```bash
bash run_inference.sh
```

The script uses `Qwen/Qwen3-0.6B`, 1-shot retrieval, batch size 3, and `TEST_SIZE=150` for both datasets. The report's WikiSQL runs used 300 questions (`--test-size 300`).

To run one configuration:

```bash
python src/inference/nltosql.py \
  --model Qwen/Qwen3-0.6B \
  --dataset wikisql \
  --constraint xgrammar \
  --mode thinking \
  --train-mode few --n-shots 1 \
  --test-size 300 \
  --batch-size 3
```

| Flag               | Default                      | Description                                                                                                                                               |
| ------------------ | ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--model`          | `Qwen/Qwen2.5-0.5B-Instruct` | Hugging Face model ID or local path. Pass `Qwen/Qwen3-0.6B` to match the report; thinking mode needs a chat template with `enable_thinking`, as in Qwen3. |
| `--dataset`        | required                     | `wikisql` or `spider`                                                                                                                                     |
| `--constraint`     | required                     | `none`, `outlines`, or `xgrammar`                                                                                                                         |
| `--mode`           | `non-thinking`               | `thinking` sets `enable_thinking=True` in the chat template                                                                                               |
| `--train-mode`     | `zero`                       | `few` prepends training examples retrieved by TF-IDF similarity                                                                                           |
| `--n-shots`        | `3`                          | Examples to retrieve in `few` mode (the report uses 1)                                                                                                    |
| `--test-size`      | whole split                  | Size of the seeded random sample (seed 42)                                                                                                                |
| `--batch-size`     | `8`                          | Generation batch size; Outlines runs one question at a time                                                                                               |
| `--max-new-tokens` | `150`                        | Generation budget; thinking mode always gets at least 1,500                                                                                               |
| `--save-interval`  | `10`                         | Save a checkpoint every N questions (`0` saves only at the end)                                                                                           |

`--data-path`, `--train-data-path`, and `--output` override the default input and output files.

Each run writes two files:

- `results/result_{dataset}_{mode}_{train_mode}_{constraint}.json`: one record per question with `instance_id`, `question`, `gold_sql`, `predicted_sql` (after post-processing), `latency_ms`, and `num_tokens_generated`. WikiSQL records also carry the table schema.
- `logs/log_{dataset}_{mode}_{train_mode}_{constraint}.txt`: a sample prompt, the first three predictions, and progress.

Re-running a configuration resumes from its result file and skips questions that are already answered. Delete the file to start that run over.

## Evaluate

From the repository root, with the databases from [Data](#data) in place:

```bash
mkdir -p output
python src/evaluation/test_ea.py          # every WikiSQL run in results/
python src/evaluation/test_ea_spider.py   # every Spider run in results/
```

Each script prints EA per run and writes scored copies to `output/eval_{dataset}_{mode}_{train_mode}_{constraint}.json`, adding `gold_answer`, `predicted_answer`, and `execution_match` to every record. Existing files in `output/` are overwritten.

- Rows are compared as sorted lists of stringified rows, so row order doesn't matter.
- For WikiSQL, the script points `FROM table` at the question's table in `test.db`, renames columns to `col<i>`, and compares string literals with `COLLATE NOCASE`.
- A gold query that fails to execute counts as a miss for every decoder: 1 Spider question (its gold SQL references a table that is missing from the database) and 2 WikiSQL questions (an apostrophe in a string value breaks the literal rewrite).

`evaluation/scripts/` holds further analysis scripts:

| Script                                 | Purpose                                                                                                                 |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `er_wiki.py`, `er_spider.py`           | EA plus executable rate                                                                                                 |
| `errors_wiki.py`, `errors_spider.py`   | Error categories; with `--valid_exec_gap`, valid-SQL rate and queries that parse but fail to execute (needs `sqlparse`) |
| `compare_wiki.py`, `compare_spider.py` | Question-level wins and losses between two runs (`--a`, `--b`): exact SQL match for WikiSQL, execution match for Spider |
| `find_case_studies.py`                 | WikiSQL questions where thinking + `none` is correct and thinking + `outlines` is not                                   |
| `plot_diagram.py`                      | Draws `evaluation/output/fig*.png` from hard-coded numbers from the midterm's 50-question pilot                         |
| `test_ea_wiki.py`, `test_ea_spider.py` | Variants of the EA scripts in `src/evaluation/`                                                                         |

These scripts hard-code their paths relative to the working directory, and several expect per-dataset folders (`results/wiki/`, `results/spider/`) instead of the flat `results/` that inference writes. Check the path constants at the top of a script before running it.

## Where the report maps to code

| Report                               | Code                                                                                          |
| ------------------------------------ | --------------------------------------------------------------------------------------------- |
| §4.2 Data preprocessing              | `src/data_processing/preprocess_wikisql.py`, `src/data_processing/preprocess_spider.py`       |
| §4.4 Prompts and few-shot retrieval  | `build_system_prompt`, `build_user_message`, `FewShotRetriever` in `src/inference/nltosql.py` |
| §4.5 and Appendix A: grammars        | `build_wikisql_regex`, `build_spider_regex`, `build_sql_ebnf`, `build_xgrammar`               |
| §4.5 Outlines integration            | `OutlinesRunner`                                                                              |
| §4.5 and Appendix C: post-processing | `clean_sql`                                                                                   |
| §4.6 Evaluation                      | `src/evaluation/`, `evaluation/scripts/`                                                      |

## Earlier work

- `arxiv/` and `src/arxiv/` archive midterm-era work: WikiSQL notebooks for Outlines, XGrammar, and LM Format Enforcer (LMFE), an older inference script, notes for a vLLM-based runner, and a guide to serving models on USC's CARC cluster. LMFE was dropped from the final sweep because it took about 12 minutes per question.
- `reports/` holds the [final report](reports/Constrained%20Decoding%20for%20NL2SQL.pdf).

## Authors

Jeongsik Park, Haiyun Lyu, Hao-Tien Cheng, Manu Goel, Wanting Jiang, Zihan Xie, and Zhuoning Wu, University of Southern California.
