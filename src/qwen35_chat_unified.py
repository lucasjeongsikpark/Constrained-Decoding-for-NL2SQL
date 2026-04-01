"""Qwen3.5-0.8B chat-completions runner with dataset-specific prompts for WikiSQL and Spider.

Features
- Two prompt templates: one for WikiSQL (single table), one for Spider (multi-table with PK/FK hints).
- Fixed generation hyperparameters per user request (no CLI knobs):
  max_tokens=81920, temperature=1.0, top_p=0.95, presence_penalty=1.5, extra_body={top_k:20, enable_thinking:bool}.
- CLI flags: --dataset {wikisql, spider}, --mode {thinking, non-thinking}, --train-mode {zero, few}, --max-examples N, --output path.
- Defaults to processed validation/test paths:
  data/wikisql_processed/wikisql_test.json
  data/spider_data_processed/spider_test.json
- Outputs JSON with keys: instance_id, question, gold_sql, predicted_sql, latency_ms, num_tokens_generated (None because API does not return it).
Environment
- Set OPENAI_API_KEY or DASHSCOPE_API_KEY for the Qwen endpoint.
- Optionally set QWEN_BASE_URL (defaults to DashScope compatible URL).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from openai import OpenAI

MODEL_NAME = "Qwen/Qwen3.5-0.8B"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

DEFAULT_DATA_PATHS = {
    "wikisql": "data/wikisql_processed/wikisql_test.json",
    "spider": "data/spider_data_processed/spider_test.json",
}

# Fixed decoding hyperparameters (do not expose via CLI per user request).
CHAT_PARAMS_BASE = {
    "model": MODEL_NAME,
    "max_tokens": 81920,
    "temperature": 1.0,
    "top_p": 0.95,
    "presence_penalty": 1.5,
    "extra_body": {
        "top_k": 20,
        # enable_thinking will be set per --mode below
    },
}


@dataclass
class Example:
    instance_id: str
    question: str
    schema_text: str
    gold_sql: str


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _chunk_pairs(pairs: Iterable[Tuple[str, str]]) -> str:
    return "\n".join(f"- {name} [{ctype}]" for name, ctype in pairs) if pairs else "(none)"


def format_wikisql_schema(schema: Dict[str, Any]) -> str:
    table_names = schema.get("table_names_original") or []
    columns = schema.get("column_names_original") or []
    column_types = schema.get("column_types") or []

    lines: List[str] = []
    for table_idx, table_name in enumerate(table_names):
        lines.append(f"TABLE: {table_name}")
        pairs: List[Tuple[str, str]] = []
        for idx, pair in enumerate(columns):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            tid, col_name = pair
            if tid != table_idx or col_name == "*":
                continue
            ctype = column_types[idx] if idx < len(column_types) else "text"
            pairs.append((str(col_name), str(ctype)))
        lines.append(_chunk_pairs(pairs))
        lines.append("")
    text = "\n".join(lines).strip()
    return text or "No schema provided."


def _group_spider_columns(schema_json: Dict[str, Any]) -> Dict[int, List[Tuple[str, str]]]:
    grouped: Dict[int, List[Tuple[str, str]]] = {}
    columns = schema_json.get("column_names_original") or []
    column_types = schema_json.get("column_types") or []
    for idx, pair in enumerate(columns):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        tid, col_name = pair
        ctype = column_types[idx] if idx < len(column_types) else "text"
        grouped.setdefault(int(tid), []).append((str(col_name), str(ctype)))
    return grouped


def format_spider_schema(schema_json: Dict[str, Any]) -> str:
    tables = schema_json.get("table_names_original") or []
    pk_set = {int(x) for x in (schema_json.get("primary_keys") or [])}
    fk_pairs = [(int(a), int(b)) for a, b in (schema_json.get("foreign_keys") or []) if isinstance(a, int) and isinstance(b, int)]
    fk_lookup: Dict[int, List[int]] = {}
    for a, b in fk_pairs:
        fk_lookup.setdefault(a, []).append(b)
        fk_lookup.setdefault(b, []).append(a)

    grouped = _group_spider_columns(schema_json)
    lines: List[str] = []
    for table_idx, table_name in enumerate(tables):
        lines.append(f"TABLE: {table_name}")
        lines.append("COLUMNS:")
        for col_name, ctype in grouped.get(table_idx, []):
            if col_name == "*":
                continue
            col_idx = None
            for i, pair in enumerate(schema_json.get("column_names_original") or []):
                if pair and pair[0] == table_idx and pair[1] == col_name:
                    col_idx = i
                    break
            suffixes: List[str] = []
            if col_idx is not None:
                if col_idx in pk_set:
                    suffixes.append("PK")
                if col_idx in fk_lookup:
                    suffixes.append("FK")
            suffix = f" ({', '.join(suffixes)})" if suffixes else ""
            lines.append(f"- {col_name} [{ctype}]{suffix}")
        lines.append("")

    if fk_pairs:
        lines.append("FOREIGN KEYS (column index pairs in column_names_original):")
        for a, b in fk_pairs:
            lines.append(f"- {a} <-> {b}")

    text = "\n".join(lines).strip()
    return text or "No schema provided."


def build_rules(dataset: str) -> str:
    rules = [
        "Produce exactly one SQL query and nothing else.",
        "Start the query with SELECT.",
        "Only use tables and columns from the provided schema.",
        "Quote text values with single quotes; do not invent values.",
        "Do not add commentary or explanation.",
    ]
    if dataset == "wikisql":
        rules.append("All queries target the single table named 'table'.")
        rules.append("Prefer the simplest valid WikiSQL-style query.")
    else:
        rules.append("Join tables only when needed and only through valid foreign keys.")
        rules.append("Keep the query concise and valid for Spider multi-table schemas.")
    return "\n".join(f"- {r}" for r in rules)


def build_user_prompt(dataset: str, ex: Example) -> str:
    rules_text = build_rules(dataset)
    return (
        f"You are a precise text-to-SQL system for {dataset}.\n"
        f"Task: Convert the question into exactly one SQL query for the schema.\n\n"
        f"Rules:\n{rules_text}\n\n"
        f"Instance ID: {ex.instance_id}\n\n"
        f"Schema:\n{ex.schema_text}\n\n"
        f"Question:\n{ex.question}\n\n"
        f"SQL:"
    )


def clean_sql(text: str) -> str:
    if text is None:
        return ""
    text = text.strip()
    text = text.replace("```sql", "").replace("```", "").strip()
    text = " ".join(text.split())
    lower = text.lower()
    if "select" not in lower:
        return ""
    text = text[lower.find("select") :]
    if ";" in text:
        text = text.split(";", 1)[0]
    if not text.lower().startswith("select"):
        return ""
    bad_suffixes = ("where", "and", "or", "(", "=", ">", "<", ">=", "<=", "!=")
    if text.lower().endswith(bad_suffixes):
        return ""
    return text.strip()


def make_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY or DASHSCOPE_API_KEY for Qwen.")
    base_url = os.environ.get("QWEN_BASE_URL", DEFAULT_BASE_URL)
    return OpenAI(api_key=api_key, base_url=base_url)


def call_qwen(client: OpenAI, messages: Sequence[Dict[str, str]], enable_thinking: bool) -> Tuple[str, float]:
    start = time.perf_counter()
    params = dict(CHAT_PARAMS_BASE)
    extra_body = dict(params["extra_body"])
    extra_body["enable_thinking"] = enable_thinking
    params["extra_body"] = extra_body
    resp = client.chat.completions.create(messages=messages, **params)
    latency_ms = (time.perf_counter() - start) * 1000.0
    content = resp.choices[0].message.content if resp.choices else ""
    return content or "", latency_ms


def to_examples(dataset: str, rows: List[Dict[str, Any]]) -> List[Example]:
    examples: List[Example] = []
    if dataset == "wikisql":
        for row in rows:
            examples.append(
                Example(
                    instance_id=str(row.get("instance_id", "")),
                    question=str(row.get("question", "")),
                    schema_text=format_wikisql_schema(row.get("schema", {})),
                    gold_sql=str(row.get("gold_sql_query", "")),
                )
            )
    else:
        for row in rows:
            input_obj = row.get("input", {})
            output_obj = row.get("output", {})
            examples.append(
                Example(
                    instance_id=str(row.get("instance_id", "")),
                    question=str(input_obj.get("question", "")),
                    schema_text=format_spider_schema(input_obj.get("schema_json", {})),
                    gold_sql=str(output_obj.get("gold_sql_query", "")),
                )
            )
    return examples


def build_output_filename(dataset: str, mode: str, train_mode: str) -> str:
    return f"result_{dataset}_{mode}_{train_mode}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.5-0.8B chat SQL generation for WikiSQL and Spider")
    parser.add_argument("--dataset", choices=["wikisql", "spider"], default="wikisql")
    parser.add_argument("--mode", choices=["thinking", "non-thinking"], default="non-thinking")
    parser.add_argument("--train-mode", choices=["zero", "few"], default="zero")
    parser.add_argument("--data-path", default=None, help="Override dataset JSON path")
    parser.add_argument("--output", default=None, help="Output JSON path; default derives from dataset/mode/train-mode")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit number of processed examples")
    args = parser.parse_args()

    data_path = args.data_path or DEFAULT_DATA_PATHS[args.dataset]
    rows = load_json(data_path)
    if args.max_examples is not None:
        rows = rows[: args.max_examples]

    examples = to_examples(args.dataset, rows)
    client = make_client()

    results: List[Dict[str, Any]] = []
    enable_thinking = args.mode == "thinking"

    for idx, ex in enumerate(examples, start=1):
        messages = [
            {"role": "system", "content": "You are a precise text-to-SQL assistant."},
            {"role": "user", "content": build_user_prompt(args.dataset, ex)},
        ]
        raw_text, latency_ms = call_qwen(client, messages, enable_thinking)
        predicted_sql = clean_sql(raw_text)
        result = {
            "instance_id": ex.instance_id,
            "question": ex.question,
            "gold_sql": ex.gold_sql,
            "predicted_sql": predicted_sql,
            "latency_ms": round(latency_ms, 3),
            "num_tokens_generated": None,
        }
        results.append(result)
        if idx <= 3:
            print("=" * 60)
            print(f"Example {idx}")
            print("Question:", ex.question)
            print("Gold SQL:", ex.gold_sql)
            print("Predicted SQL:", predicted_sql)
            print("Latency (ms):", round(latency_ms, 2))
        if idx % 20 == 0 or idx == len(examples):
            print(f"Processed {idx}/{len(examples)}")

    out_path = args.output or build_output_filename(args.dataset, args.mode, args.train_mode)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Saved to:", out_path)


if __name__ == "__main__":
    main()
