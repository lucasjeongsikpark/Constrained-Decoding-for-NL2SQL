#!/usr/bin/env python3
"""Download WikiSQL and build train/val/test JSON files in project format.

Output row format (matching data/wikisql_train_sample50.json):
{
  "instance_id": "wikisql_train_1",
  "db": "1-10015132-11",
  "question": "...",
  "schema": {
    "table_names_original": ["table"],
    "column_names_original": [[0, "col_a"], [0, "col_b"]],
    "column_types": ["text", "real"]
  },
  "gold_sql_query": "SELECT ..."
}
"""

from __future__ import annotations

import argparse
import json
import tarfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


WIKISQL_ARCHIVE_URL = "https://github.com/salesforce/WikiSQL/raw/master/data.tar.bz2"
AGG_OPS = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
COND_OPS = ["=", ">", "<", "OP"]


def safe_extract_tar_bz2(archive_path: Path, out_dir: Path) -> None:
    """Safely extract tar archive to avoid path traversal."""

    def is_within_directory(directory: Path, target: Path) -> bool:
        try:
            target.resolve().relative_to(directory.resolve())
            return True
        except ValueError:
            return False

    with tarfile.open(archive_path, "r:bz2") as tar:
        for member in tar.getmembers():
            member_path = out_dir / member.name
            if not is_within_directory(out_dir, member_path):
                raise RuntimeError(f"Unsafe tar member path: {member.name}")
        tar.extractall(path=out_dir)


def ensure_wikisql_raw(raw_root: Path, force_download: bool = False) -> Path:
    """Download and extract official WikiSQL archive, returning raw data dir."""
    raw_data_dir = raw_root / "data"
    required_files = [
        raw_data_dir / "train.jsonl",
        raw_data_dir / "train.tables.jsonl",
        raw_data_dir / "dev.jsonl",
        raw_data_dir / "dev.tables.jsonl",
        raw_data_dir / "test.jsonl",
        raw_data_dir / "test.tables.jsonl",
    ]

    if not force_download and all(p.exists() for p in required_files):
        return raw_data_dir

    raw_root.mkdir(parents=True, exist_ok=True)
    archive_path = raw_root / "data.tar.bz2"

    print(f"Downloading WikiSQL archive from: {WIKISQL_ARCHIVE_URL}")
    urllib.request.urlretrieve(WIKISQL_ARCHIVE_URL, archive_path)

    print(f"Extracting archive to: {raw_root}")
    safe_extract_tar_bz2(archive_path, raw_root)

    if not all(p.exists() for p in required_files):
        missing = [str(p) for p in required_files if not p.exists()]
        raise FileNotFoundError(f"Missing expected WikiSQL files after extraction: {missing}")

    return raw_data_dir


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_column_type(raw_type: Any) -> str:
    if not isinstance(raw_type, str):
        return "text"

    t = raw_type.strip().lower()
    if t in {"real", "number", "float", "int", "integer", "numeric"}:
        return "real"
    return "text"


def build_schema(table_obj: Dict[str, Any]) -> Dict[str, Any]:
    headers = table_obj.get("header") if isinstance(table_obj.get("header"), list) else []
    raw_types = table_obj.get("types") if isinstance(table_obj.get("types"), list) else ["text"] * len(headers)

    column_names_original = [[0, str(h)] for h in headers]
    column_types = [
        normalize_column_type(raw_types[i] if i < len(raw_types) else "text")
        for i in range(len(headers))
    ]

    return {
        "table_names_original": ["table"],
        "column_names_original": column_names_original,
        "column_types": column_types,
    }


def is_number_like(value: Any) -> bool:
    if isinstance(value, (int, float)):
        return True
    if not isinstance(value, str):
        return False

    text = value.strip()
    if not text:
        return False

    try:
        float(text)
        return True
    except ValueError:
        return False


def to_sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"

    if is_number_like(value):
        return str(value).strip()

    text = str(value).replace("'", "''")
    return f"'{text}'"


def sql_from_struct(sql_obj: Dict[str, Any], columns: List[str]) -> str:
    """Convert WikiSQL structured sql dict into SQL text."""
    sel_idx = int(sql_obj.get("sel", 0))
    agg_idx = int(sql_obj.get("agg", 0))

    if not columns:
        select_col = "*"
    elif 0 <= sel_idx < len(columns):
        select_col = columns[sel_idx]
    else:
        select_col = columns[0]

    agg_fn = AGG_OPS[agg_idx] if 0 <= agg_idx < len(AGG_OPS) else ""
    select_expr = f"{agg_fn}({select_col})" if agg_fn else select_col

    conds = sql_obj.get("conds", [])
    cond_parts: List[str] = []
    if isinstance(conds, Iterable):
        for cond in conds:
            if not isinstance(cond, (list, tuple)) or len(cond) != 3:
                continue
            col_idx, op_idx, raw_val = cond
            col_idx = int(col_idx)
            op_idx = int(op_idx)

            if not columns:
                col_name = "*"
            elif 0 <= col_idx < len(columns):
                col_name = columns[col_idx]
            else:
                continue

            op = COND_OPS[op_idx] if 0 <= op_idx < len(COND_OPS) else "="
            cond_parts.append(f"{col_name} {op} {to_sql_literal(raw_val)}")

    query = f"SELECT {select_expr} FROM table"
    if cond_parts:
        query += " WHERE " + " AND ".join(cond_parts)
    return query


def make_record(
    *,
    split_name: str,
    idx: int,
    example: Dict[str, Any],
    table_index: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    table_id = str(example.get("table_id", ""))
    table_obj = table_index.get(table_id)
    if table_obj is None:
        raise KeyError(f"table_id '{table_id}' missing from table index")

    schema = build_schema(table_obj)
    columns = [pair[1] for pair in schema["column_names_original"]]

    sql_obj = example.get("sql") if isinstance(example.get("sql"), dict) else {}
    gold_sql_query = sql_from_struct(sql_obj, columns)

    question = example.get("question") if isinstance(example.get("question"), str) else ""

    return {
        "instance_id": f"wikisql_{split_name}_{idx}",
        "db": table_id,
        "question": question,
        "schema": schema,
        "gold_sql_query": gold_sql_query,
    }


def write_json(path: Path, rows: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def preprocess(
    *,
    raw_data_dir: Path,
    output_dir: Path,
    sample50_path: Optional[Path],
) -> Dict[str, int]:
    split_file_map = {
        "train": (raw_data_dir / "train.jsonl", raw_data_dir / "train.tables.jsonl"),
        "val": (raw_data_dir / "dev.jsonl", raw_data_dir / "dev.tables.jsonl"),
        "test": (raw_data_dir / "test.jsonl", raw_data_dir / "test.tables.jsonl"),
    }

    counts: Dict[str, int] = {}
    split_rows_cache: Dict[str, List[Dict[str, Any]]] = {}

    for split_name in ("train", "val", "test"):
        split_path, tables_path = split_file_map[split_name]
        examples = read_jsonl(split_path)
        table_rows = read_jsonl(tables_path)
        table_index = {str(t["id"]): t for t in table_rows}

        rows = [
            make_record(
                split_name=split_name,
                idx=i + 1,
                example=ex,
                table_index=table_index,
            )
            for i, ex in enumerate(examples)
        ]

        split_rows_cache[split_name] = rows
        write_json(output_dir / f"wikisql_{split_name}.json", rows)
        counts[split_name] = len(rows)

    if sample50_path is not None:
        train_rows = split_rows_cache["train"][:50]
        write_json(sample50_path, train_rows)
        counts["train_sample50"] = len(train_rows)

    write_json(output_dir / "stats.json", counts)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess WikiSQL to project JSON format")
    parser.add_argument(
        "--raw-dir",
        default="data/wikisql_raw",
        help="Directory for downloaded/extracted official WikiSQL files",
    )
    parser.add_argument(
        "--output-dir",
        default="data/wikisql_processed",
        help="Directory to save wikisql_train.json / wikisql_val.json / wikisql_test.json",
    )
    parser.add_argument(
        "--write-train-sample50",
        action="store_true",
        help="Also write data/wikisql_train_sample50.json",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download archive even when raw files already exist",
    )
    args = parser.parse_args()

    raw_root = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    sample50_path = Path("data/wikisql_train_sample50.json") if args.write_train_sample50 else None

    raw_data_dir = ensure_wikisql_raw(raw_root, force_download=args.force_download)
    counts = preprocess(raw_data_dir=raw_data_dir, output_dir=output_dir, sample50_path=sample50_path)

    print("=== WikiSQL preprocessing complete ===")
    print(json.dumps(counts, indent=2))
    print(f"Raw files dir: {raw_data_dir}")
    print(f"Saved split files to: {output_dir}")
    if sample50_path is not None:
        print(f"Updated sample file: {sample50_path}")


if __name__ == "__main__":
    main()
