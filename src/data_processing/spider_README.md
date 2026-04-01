# CSCI566-GroupProject

## Spider Analysis-Friendly Dataset

This repository includes a preprocessing script to build Spider 1.0 into
analysis-friendly `train/val/test` splits.

### Target Format

Each record is generated with:

```json
{
  "instance_id": "spider_train_000001",
  "split": "train",
  "source_split": "train_spider",
  "db_id": "department_management",
  "input": {
    "question": "...",
    "db_id": "department_management",
    "schema_json": {
      "table_names_original": [],
      "column_names_original": [],
      "column_types": [],
      "primary_keys": [],
      "foreign_keys": []
    }
  },
  "output": {
    "gold_sql_query": "SELECT ..."
  }
}
```

### Build Command

Run from repo root:

```bash
/opt/homebrew/bin/python3 code/data/preprocess_spider.py
```

### Output Files

- `data/spider_data/spider_data_processed/spider_train.json`
- `data/spider_data/spider_data_processed/spider_val.json`
- `data/spider_data/spider_data_processed/spider_test.json`
- `data/spider_data/spider_data_processed/stats.json`
- `data/spider_data/spider_data_processed/validation_errors.json`

### Official Split Policy

- `train = train_spider.json + train_others.json`
- `val = dev.json`
- `test = test.json`
