"""Chat-completions runner for Qwen3.5-0.8B on WikiSQL and Spider.

Usage:
    python src/qwen35_chat_sql.py --dataset wikisql --max-examples 5
    python src/qwen35_chat_sql.py --dataset spider --max-examples 5

Environment:
    OPENAI_API_KEY (or DASHSCOPE_API_KEY) must be set for the Qwen endpoint.
    Optionally set QWEN_BASE_URL to override the base URL (defaults to
    https://dashscope.aliyuncs.com/compatible-mode/v1).
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

# Fixed generation hyperparameters requested by user.
CHAT_PARAMS = {
    "model": MODEL_NAME,
    "max_tokens": 81920,
    "temperature": 1.0,
    "top_p": 0.95,
    "presence_penalty": 1.5,
    "extra_body": {
        "top_k": 20,
        "enable_thinking": True,
    },
}

DEFAULT_DATA_PATHS = {
    "wikisql": "data/wikisql_processed/wikisql_val.json",
    "spider": "data/spider_data_processed/spider_val.json",
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


def chunk_pairs(pairs: Iterable[Tuple[str, str]]) -> str:
    lines = [f"- {name} [{ctype}]" for name, ctype in pairs]
    return "\n".join(lines) if lines else "(none)"


def format_wikisql_schema(schema: Dict[str, Any]) -> str:
    table_names = schema.get("table_names_original") or []
    columns = schema.get("column_names_original") or []
    column_types = schema.get("column_types") or []

    lines: List[str] = []
    for table_idx, table_name in enumerate(table_names):
        lines.append(f"TABLE: {table_name}")
        col_pairs: List[Tuple[str, str]] = []
        for idx, pair in enumerate(columns):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            tid, col_name = pair
            if tid != table_idx or col_name == "*":
                continue
            ctype = column_types[idx] if idx < len(column_types) else "text"
            col_pairs.append((str(col_name), str(ctype)))
        lines.append(chunk_pairs(col_pairs))
        lines.append("")
    return "\n".join(lines).strip() or "No schema provided."


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
    fk_pairs = [(int(a), int(b)) for a, b in (schema_json.get("foreign_keys") or []) if isinstance(a, int)]
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
        lines.append("FOREIGN KEYS (by column index in column_names_original):")
        for a, b in fk_pairs:
            lines.append(f"- {a} <-> {b}")

    return "\n".join(lines).strip() or "No schema provided."


def build_common_rules(dataset: str) -> str:
    rules = [
        "Produce exactly one SQL query and nothing else.",
        "Start the query with SELECT.",
        "Only use tables and columns from the provided schema.",
        "Quote text values with single quotes and do not invent values.",
        "Do not add commentary or explanation.",
    ]
    if dataset == "wikisql":
        rules.append("All queries target the single table named 'table'.")
        rules.append("Prefer the simplest valid WikiSQL-style query.")
    else:
        rules.append("Join tables only through valid foreign keys when needed.")
        rules.append("Keep the query as short as possible while correct for Spider.")
    return "\n".join(f"- {r}" for r in rules)


def build_user_prompt(dataset: str, example: Example) -> str:
    rules_text = build_common_rules(dataset)
    return (
        f"You are a precise text-to-SQL system for {dataset}.\n"
        f"Task: Convert the question into exactly one SQL query for the schema.\n\n"
        f"Rules:\n{rules_text}\n\n"
        f"Instance ID: {example.instance_id}\n\n"
        f"Schema:\n{example.schema_text}\n\n"
        f"Question:\n{example.question}\n\n"
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


def call_qwen(client: OpenAI, messages: Sequence[Dict[str, str]]) -> Tuple[str, float]:
    start = time.perf_counter()
    resp = client.chat.completions.create(messages=messages, **CHAT_PARAMS)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen3.5-0.8B chat completions for text-to-SQL")
    parser.add_argument("--dataset", choices=["wikisql", "spider"], default="wikisql")
    parser.add_argument("--data-path", help="Override dataset JSON path; defaults to processed val split")
    parser.add_argument("--output", default=None, help="Where to write JSON results (default derives from dataset)")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit number of processed examples")
    args = parser.parse_args()

    data_path = args.data_path or DEFAULT_DATA_PATHS[args.dataset]
    rows = load_json(data_path)
    if args.max_examples is not None:
        rows = rows[: args.max_examples]

    examples = to_examples(args.dataset, rows)
    client = make_client()

    results: List[Dict[str, Any]] = []
    for idx, ex in enumerate(examples, start=1):
        messages = [
            {"role": "system", "content": "You are a precise text-to-SQL assistant."},
            {"role": "user", "content": build_user_prompt(args.dataset, ex)},
        ]
        raw_text, latency_ms = call_qwen(client, messages)
        predicted_sql = clean_sql(raw_text)
        result = {
            "instance_id": ex.instance_id,
            "question": ex.question,
            "gold_sql": ex.gold_sql,
            "predicted_sql": predicted_sql,
            "latency_ms": round(latency_ms, 3),
            "num_tokens_generated": None,  # Not returned by chat-completions API
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

    out_path = args.output or f"result_qwen35_chat_{args.dataset}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Saved to:", out_path)


if __name__ == "__main__":
    main()
