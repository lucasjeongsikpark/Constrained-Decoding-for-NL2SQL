#!/usr/bin/env python3
# Run: /opt/homebrew/bin/python3 code/data/preprocess_spider.py
"""Build analysis-friendly Spider train/val/test JSON files.

Outputs:
- spider_train.json
- spider_val.json
- spider_test.json
- stats.json
- validation_errors.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


@dataclass
class SplitSpec:
    name: str
    source_files: List[Tuple[str, Path]]


SCHEMA_KEYS = [
    "table_names_original",
    "column_names_original",
    "column_types",
    "primary_keys",
    "foreign_keys",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_schema_index(*tables_paths: Path) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for tables_path in tables_paths:
        if not tables_path.exists():
            continue
        raw_tables = load_json(tables_path)
        for entry in raw_tables:
            db_id = entry["db_id"]
            index[db_id] = {k: entry[k] for k in SCHEMA_KEYS}
    return index


def make_record(
    *,
    split_name: str,
    source_split: str,
    running_idx: int,
    example: Dict[str, Any],
    schema_index: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    db_id = example["db_id"]
    schema_json = schema_index[db_id]
    return {
        "instance_id": f"spider_{split_name}_{running_idx:06d}",
        "split": split_name,
        "source_split": source_split,
        "db_id": db_id,
        "input": {
            "question": example["question"],
            "db_id": db_id,
            "schema_json": schema_json,
        },
        "output": {
            "gold_sql_query": example["query"],
        },
    }


def write_json(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def validate_rows(rows: List[Dict[str, Any]], schema_index: Dict[str, Dict[str, Any]]) -> List[str]:
    errors: List[str] = []
    for i, row in enumerate(rows):
        prefix = f"row={i} instance_id={row.get('instance_id')}"
        input_obj = row.get("input", {})
        output_obj = row.get("output", {})

        if not isinstance(input_obj.get("question"), str) or not input_obj["question"].strip():
            errors.append(f"{prefix}: missing/empty input.question")
        if not isinstance(input_obj.get("db_id"), str) or not input_obj["db_id"].strip():
            errors.append(f"{prefix}: missing/empty input.db_id")
        if input_obj.get("db_id") not in schema_index:
            errors.append(f"{prefix}: input.db_id not found in schema index")
        if not isinstance(input_obj.get("schema_json"), dict):
            errors.append(f"{prefix}: missing input.schema_json")
        if not isinstance(output_obj.get("gold_sql_query"), str) or not output_obj["gold_sql_query"].strip():
            errors.append(f"{prefix}: missing/empty output.gold_sql_query")

    return errors


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    db_ids = {r["db_id"] for r in rows}
    return {
        "records": len(rows),
        "unique_db_ids": len(db_ids),
    }


def process_split(spec: SplitSpec, schema_index: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    running_idx = 1
    for source_split, path in spec.source_files:
        examples = load_json(path)
        for ex in examples:
            db_id = ex.get("db_id")
            if db_id not in schema_index:
                raise ValueError(f"db_id '{db_id}' in {path} is missing from schema files")
            rows.append(
                make_record(
                    split_name=spec.name,
                    source_split=source_split,
                    running_idx=running_idx,
                    example=ex,
                    schema_index=schema_index,
                )
            )
            running_idx += 1
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Spider analysis-friendly splits")
    parser.add_argument(
        "--spider-dir",
        default="data/spider_data",
        help="Directory containing Spider source files",
    )
    parser.add_argument(
        "--output-dir",
        default="data/spider_data_processed",
        help="Output directory for processed files",
    )
    args = parser.parse_args()

    spider_dir = Path(args.spider_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    schema_index = build_schema_index(
        spider_dir / "tables.json",
        spider_dir / "test_tables.json",
    )

    split_specs = [
        SplitSpec(
            name="train",
            source_files=[
                ("train_spider", spider_dir / "train_spider.json"),
                ("train_others", spider_dir / "train_others.json"),
            ],
        ),
        SplitSpec(name="val", source_files=[("dev", spider_dir / "dev.json")]),
        SplitSpec(name="test", source_files=[("test", spider_dir / "test.json")]),
    ]

    output_name_map = {
        "train": "spider_train.json",
        "val": "spider_val.json",
        "test": "spider_test.json",
    }

    stats: Dict[str, Any] = {}
    validation_errors: Dict[str, List[str]] = {}

    for spec in split_specs:
        rows = process_split(spec, schema_index)
        write_json(output_dir / output_name_map[spec.name], rows)

        errors = validate_rows(rows, schema_index)
        validation_errors[spec.name] = errors
        stats[spec.name] = summarize(rows)

    stats["expected_counts"] = {
        "train": len(load_json(spider_dir / "train_spider.json")) + len(load_json(spider_dir / "train_others.json")),
        "val": len(load_json(spider_dir / "dev.json")),
        "test": len(load_json(spider_dir / "test.json")),
    }
    stats["all_validation_ok"] = all(not errs for errs in validation_errors.values())
    stats["validation_error_counts"] = {k: len(v) for k, v in validation_errors.items()}

    write_json(output_dir / "stats.json", stats)
    write_json(output_dir / "validation_errors.json", validation_errors)

    print("=== Spider preprocessing complete ===")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
