"""OpenAI-compatible Qwen3.5 runner for text-to-SQL with constraints.

Matches the CLI, prompts, and outputs of qwen35_hf_sql.py but calls a remote
FastAPI server (see src/server/deploy_qwen.sh) through the OpenAI client. All
constraint modes are supported server-side: none, outlines, lmfe, xgrammar.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from openai import OpenAI

MODEL_NAME = "Qwen/Qwen3.5-0.8B"
DEFAULT_BASE_URL = "http://localhost:8082/v1"
DEFAULT_DATA_PATHS = {
    "wikisql": "data/wikisql_processed/wikisql_test.json",
    "spider": "data/spider_data_processed/spider_test.json",
}

DEFAULT_MAX_NEW_TOKENS = 100


@dataclass
class Example:
    instance_id: str
    question: str
    schema: Dict[str, Any]
    gold_sql: str


# --------------------------- utils ---------------------------

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_default_sampling() -> Dict[str, Any]:
    # Deterministic by default; temperature=0 maps to greedy on the server.
    return {"temperature": 0.0, "top_p": 1.0}


def build_generation_kwargs(max_new_tokens: int) -> Dict[str, Any]:
    kwargs = get_default_sampling()
    kwargs["max_tokens"] = max_new_tokens
    return kwargs


# --------------------------- schema formatting ---------------------------

def chunk_pairs(pairs: Iterable[Tuple[str, str]]) -> str:
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
        lines.append(chunk_pairs(pairs))
        lines.append("")
    return "\n".join(lines).strip() or "No schema provided."


def group_spider_columns(schema_json: Dict[str, Any]) -> Dict[int, List[Tuple[str, str]]]:
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
    grouped = group_spider_columns(schema_json)
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
    return "\n".join(lines).strip() or "No schema provided."


# --------------------------- prompts ---------------------------

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
    schema_text = format_wikisql_schema(ex.schema) if dataset == "wikisql" else format_spider_schema(ex.schema)
    rules_text = build_rules(dataset)
    return (
        f"You are a precise text-to-SQL system for {dataset}.\n"
        f"Task: Convert the question into exactly one SQL query for the schema.\n\n"
        f"Rules:\n{rules_text}\n\n"
        f"Instance ID: {ex.instance_id}\n\n"
        f"Schema:\n{schema_text}\n\n"
        f"Question:\n{ex.question}\n\n"
        f"SQL:"
    )


def build_messages(dataset: str, ex: Example) -> List[Dict[str, str]]:
    user_prompt = build_user_prompt(dataset, ex)
    return [
        {"role": "system", "content": "You are a precise text-to-SQL assistant."},
        {"role": "user", "content": user_prompt},
    ]


# --------------------------- cleaning ---------------------------

def clean_sql(text: str, mode: str = "non-thinking") -> str:
    if text is None:
        return ""
    text = text.strip()
    if mode == "thinking" and "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    text = re.sub(r"<\|.*?\|>", "", text)
    text = text.replace("```sql", "").replace("```", "").strip()
    text = " ".join(text.split())
    lower = text.lower()
    if "select" not in lower:
        return ""
    text = text[lower.find("select") :]
    if ";" in text:
        text = text.split(";", 1)[0].strip()
    bad_suffixes = ("where", "and", "or", "(", "=", ">", "<", ">=", "<=", "!=")
    if text.lower().endswith(bad_suffixes):
        return ""
    if text.count("(") != text.count(")"):
        return ""
    if not text.lower().startswith("select"):
        return ""
    return text


# --------------------------- constraint helpers ---------------------------

def build_wikisql_regex() -> str:
    ident = r"[A-Za-z_][A-Za-z0-9_ ]*"
    number = r"-?\d+(?:\.\d+)?"
    quoted = r"'[^']*'"
    bare = r"[A-Za-z0-9_./()\-]+"
    value = rf"(?:{quoted}|{number}|{bare})"
    op = r"(?:=|!=|<|>|<=|>=|LIKE)"
    agg = rf"(?:COUNT\(\*\)|COUNT\({ident}\)|MAX\({ident}\)|MIN\({ident}\)|SUM\({ident}\)|AVG\({ident}\))"
    select_expr = rf"(?:\*|{ident}|{agg})"
    condition = rf"{ident}\s+{op}\s+{value}"
    where_opt = rf"(?:\s+WHERE\s+{condition}(?:\s+(?:AND|OR)\s+{condition})*)?"
    order_opt = rf"(?:\s+ORDER\s+BY\s+{ident}(?:\s+(?:ASC|DESC))?)?"
    limit_opt = rf"(?:\s+LIMIT\s+\d+)?"
    return rf"SELECT\s+(?:DISTINCT\s+)?{select_expr}\s+FROM\s+table{where_opt}{order_opt}{limit_opt}\s*;?"


def build_spider_regex(table_names: List[str]) -> str:
    table_union = "|".join(re.escape(t) for t in table_names) or "[A-Za-z_][A-Za-z0-9_]*"
    ident = r"[A-Za-z_][A-Za-z0-9_ ]*"
    number = r"-?\d+(?:\.\d+)?"
    quoted = r"'[^']*'"
    bare = r"[A-Za-z0-9_./()\-]+"
    value = rf"(?:{quoted}|{number}|{bare})"
    op = r"(?:=|!=|<|>|<=|>=|LIKE|IN|NOT IN)"
    select_expr = rf"(?:\*|{ident}|{ident}\. {ident}|{ident}\({ident}\))"
    condition = rf"{ident}(?:\.{ident})?\s+{op}\s+{value}"
    join_part = rf"(?:\s+JOIN\s+(?:{table_union})\s+ON\s+{condition})*"
    where_opt = rf"(?:\s+WHERE\s+{condition}(?:\s+(?:AND|OR)\s+{condition})*)?"
    order_opt = rf"(?:\s+ORDER\s+BY\s+{ident}(?:\.{ident})?(?:\s+(?:ASC|DESC))?)?"
    limit_opt = rf"(?:\s+LIMIT\s+\d+)?"
    return rf"SELECT\s+(?:DISTINCT\s+)?{select_expr}\s+FROM\s+(?:{table_union}){join_part}{where_opt}{order_opt}{limit_opt}\s*;?"


def escape_ebnf_literal(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\"')


def build_sql_ebnf(columns: List[str], tables: List[str], *, allow_or: bool, allow_join: bool) -> str:
    col_rules = " | ".join(f'"{escape_ebnf_literal(c)}"' for c in columns if c != "*") or '"col"'
    table_rules = " | ".join(f'"{escape_ebnf_literal(t)}"' for t in tables) or '"table"'
    and_or = '("AND" | "OR")' if allow_or else '"AND"'
    join_ref = ' join_clause' if allow_join else ''
    join_clause = r"""
join_clause  ::= ( ws "JOIN" ws table_name ws "ON" ws condition )*
""" if allow_join else ""
    grammar = rf"""
root         ::= select_stmt

select_stmt  ::= "SELECT" ws select_expr ws "FROM" ws table_name{join_ref} where_clause order_clause limit_clause

select_expr  ::= agg_expr | column_name | "*"
agg_expr     ::= agg_func ws "(" ws ("*" | column_name) ws ")"
agg_func     ::= "COUNT" | "MAX" | "MIN" | "SUM" | "AVG"

where_clause ::= "" | ws "WHERE" ws condition (ws {and_or} ws condition)*
condition    ::= column_name ws op ws value
op           ::= "=" | "!=" | ">" | "<" | ">=" | "<=" | "LIKE" | "IN" | "NOT" ws "IN" | "IS" ws "NULL" | "IS" ws "NOT" ws "NULL"

value        ::= number | quoted_string | bare_word | "NULL"
number       ::= "-"? [0-9]+ ("." [0-9]+)?
quoted_string ::= "\"" dq_char* "\"" | "'" sq_char* "'"
dq_char      ::= [^"\\] | "\\" ["\\/bfnrt]
sq_char      ::= [^'\\] | "\\" ['\\/bfnrt]
bare_word    ::= [A-Za-z0-9_.%+\-]+"

order_clause ::= "" | ws "ORDER" ws "BY" ws column_name (ws ("ASC" | "DESC"))?
limit_clause ::= "" | ws "LIMIT" ws [0-9]+
{join_clause}
table_name   ::= {table_rules}
column_name  ::= {col_rules}
ws           ::= [ \t]+
"""
    return grammar.strip()


def build_xgrammar(dataset: str, example: Dict[str, Any]) -> str:
    if dataset == "wikisql":
        schema = example.get("schema", {})
        cols = [c[1] for c in schema.get("column_names_original", [])]
        tables = schema.get("table_names_original", ["table"])
        return build_sql_ebnf(cols, tables, allow_or=True, allow_join=False)
    input_obj = example.get("input", {})
    schema_json = input_obj.get("schema_json", {})
    cols = [c[1] for c in schema_json.get("column_names_original", [])]
    tables = schema_json.get("table_names_original", [])
    return build_sql_ebnf(cols, tables, allow_or=True, allow_join=True)


# --------------------------- data conversion ---------------------------

def to_examples(dataset: str, rows: List[Dict[str, Any]]) -> List[Example]:
    examples: List[Example] = []
    if dataset == "wikisql":
        for row in rows:
            examples.append(
                Example(
                    instance_id=str(row.get("instance_id", "")),
                    question=str(row.get("question", "")),
                    schema=row.get("schema", {}),
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
                    schema=input_obj.get("schema_json", {}),
                    gold_sql=str(output_obj.get("gold_sql_query", "")),
                )
            )
    return examples


# --------------------------- generation core ---------------------------

def build_output_filename(dataset: str, mode: str, train_mode: str, constraint: str) -> str:
    return f"result_{dataset}_{mode}_{train_mode}_{constraint}.json"


def call_openai(client: OpenAI, model_name: str, messages: List[Dict[str, str]], *, constraint: str, regex_pattern: Optional[str], grammar: Optional[str], mode: str, generation_kwargs: Dict[str, Any]) -> Tuple[str, int, float]:
    extra_body: Dict[str, Any] = {
        "constraint": constraint,
        "regex_pattern": regex_pattern,
        "grammar": grammar,
        "mode": mode,
    }
    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=generation_kwargs.get("max_tokens"),
        temperature=generation_kwargs.get("temperature"),
        top_p=generation_kwargs.get("top_p"),
        extra_body=extra_body,
    )
    latency_ms = getattr(resp, "latency_ms", None)
    if latency_ms is None:
        latency_ms = (time.perf_counter() - start) * 1000.0
    text = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    num_tokens_generated = getattr(usage, "completion_tokens", None)
    if num_tokens_generated is None:
        num_tokens_generated = len(text.split())
    return text, int(num_tokens_generated), float(latency_ms)


# --------------------------- main ---------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.5 OpenAI text-to-SQL with constraints")
    parser.add_argument("--dataset", choices=["wikisql", "spider"], required=True)
    parser.add_argument("--constraint", choices=["none", "outlines", "lmfe", "xgrammar"], required=True)
    parser.add_argument("--mode", choices=["thinking", "non-thinking"], default="non-thinking")
    parser.add_argument("--train-mode", choices=["zero", "few"], default="zero")
    parser.add_argument("--data-path", default=None, help="Override dataset JSON path")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None, help="Randomly sample this many examples (seed=42)")
    parser.add_argument("--output", default=None)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model-name", default=MODEL_NAME)
    args = parser.parse_args()

    data_path = args.data_path or DEFAULT_DATA_PATHS[args.dataset]
    rows = load_json(data_path)
    print(f"Loaded {len(rows)} rows from {data_path}")
    if args.test_size is not None:
        random.seed(42)
        take_k = min(args.test_size, len(rows))
        rows = random.sample(rows, k=take_k)
        print(f"Sampled {len(rows)} rows (test_size={args.test_size}, seed=42)")
    if args.max_examples is not None:
        rows = rows[: args.max_examples]
        print(f"Truncated to first {len(rows)} rows due to max_examples={args.max_examples}")
    examples = to_examples(args.dataset, rows)

    client = OpenAI(base_url=args.base_url, api_key=os.getenv("OPENAI_API_KEY", "EMPTY"))

    generation_kwargs = build_generation_kwargs(args.max_new_tokens)

    results: List[Dict[str, Any]] = []
    for idx, (ex, raw_row) in enumerate(zip(examples, rows), start=1):
        messages = build_messages(args.dataset, ex)

        regex_pattern = None
        grammar = None
        if args.constraint in {"outlines", "lmfe"}:
            regex_pattern = build_wikisql_regex() if args.dataset == "wikisql" else build_spider_regex(ex.schema.get("table_names_original", []))
        elif args.constraint == "xgrammar":
            grammar = build_xgrammar(args.dataset, raw_row)

        raw_text, num_tokens, latency_ms = call_openai(
            client,
            args.model_name,
            messages,
            constraint=args.constraint,
            regex_pattern=regex_pattern,
            grammar=grammar,
            mode=args.mode,
            generation_kwargs=generation_kwargs,
        )
        predicted_sql = clean_sql(raw_text, mode=args.mode)
        result = {
            "instance_id": ex.instance_id,
            "question": ex.question,
            "gold_sql": ex.gold_sql,
            "predicted_sql": predicted_sql,
            "latency_ms": round(float(latency_ms), 3),
            "num_tokens_generated": int(num_tokens),
        }
        results.append(result)
        if idx <= 3:
            print("=" * 60)
            print(f"Example {idx}")
            print("Question:", ex.question)
            print("Gold SQL:", ex.gold_sql)
            print("Predicted SQL:", predicted_sql)
            print("Constraint:", args.constraint)
            print("Latency (ms):", round(latency_ms, 2))
            print("Generated tokens:", num_tokens)
        if idx % 20 == 0 or idx == len(examples):
            print(f"Processed {idx}/{len(examples)}")

    out_path = args.output or build_output_filename(args.dataset, args.mode, args.train_mode, args.constraint)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Saved to:", out_path)


if __name__ == "__main__":
    main()
