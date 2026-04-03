from __future__ import annotations

import argparse
import json
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_DATA_PATHS = {
    "wikisql": "data/wikisql_processed/wikisql_test.json",
    "spider": "data/spider_data_processed/spider_test.json",
}
DEFAULT_MAX_NEW_TOKENS = 150           # non-thinking: SQL fits easily in 150 tokens
DEFAULT_MAX_NEW_TOKENS_THINKING = 1500  # thinking: <think> chain adds ~600 tokens on average
SEED = 42


@dataclass
class Example:
    instance_id: str
    question: str
    schema: Dict[str, Any]
    gold_sql: str


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
            if ctype == "real":
                ctype = "number"
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
        if not isinstance(tid, int):
            continue
        ctype = column_types[idx] if idx < len(column_types) else "text"
        grouped.setdefault(tid, []).append((str(col_name), str(ctype)))
    return grouped


def format_spider_schema(schema_json: Dict[str, Any]) -> str:
    tables = schema_json.get("table_names_original") or []
    pk_set = {int(x) for x in (schema_json.get("primary_keys") or [])}
    fk_pairs = [
        (int(a), int(b))
        for a, b in (schema_json.get("foreign_keys") or [])
        if isinstance(a, int) and isinstance(b, int)
    ]
    fk_lookup: Dict[int, List[int]] = {}
    for a, b in fk_pairs:
        fk_lookup.setdefault(a, []).append(b)
        fk_lookup.setdefault(b, []).append(a)
    grouped = group_spider_columns(schema_json)
    lines: List[str] = []
    original_cols = schema_json.get("column_names_original") or []

    for table_idx, table_name in enumerate(tables):
        lines.append(f"TABLE: {table_name}")
        lines.append("COLUMNS:")
        for col_name, ctype in grouped.get(table_idx, []):
            if col_name == "*":
                continue
            col_idx = None
            for i, pair in enumerate(original_cols):
                if (
                    isinstance(pair, (list, tuple))
                    and len(pair) == 2
                    and pair[0] == table_idx
                    and pair[1] == col_name
                ):
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
        lines.append("FOREIGN KEYS:")
        for a, b in fk_pairs:
            if 0 <= a < len(original_cols) and 0 <= b < len(original_cols):
                ta_idx, ca = original_cols[a]
                tb_idx, cb = original_cols[b]
                ta = tables[ta_idx] if 0 <= ta_idx < len(tables) else "?"
                tb = tables[tb_idx] if 0 <= tb_idx < len(tables) else "?"
                lines.append(f"- {ta}.{ca} = {tb}.{cb}")
            else:
                lines.append(f"- column_{a} = column_{b}")
    return "\n".join(lines).strip() or "No schema provided."


def format_schema_text(schema: Dict[str, Any], dataset: str) -> str:
    if dataset == "wikisql":
        return format_wikisql_schema(schema)
    return format_spider_schema(schema)


def build_system_prompt(dataset: str) -> str:
    """Role + task description + dataset-specific rules — stable across all examples; belongs in system turn."""
    if dataset == "wikisql":
        dataset_context = (
            "You are working with the WikiSQL dataset. "
            "Each database has exactly one table always named 'table'. "
            "Queries are simple: single-table SELECT with optional aggregation, WHERE conditions, and ORDER BY. "
            "No JOINs, subqueries, or UNION are needed."
        )
        rules = [
            "1. Output ONLY the SQL query — no explanation, no markdown, no code fences, nothing else.",
            "2. The query must start with SELECT.",
            "3. Always use the table name: table.",
            "4. Use only columns that appear in the schema.",
            "5. Do not use JOINs, subqueries, or UNION.",
            "6. Preserve string values exactly from the question when used in WHERE conditions.",
            "7. Put text values in single quotes.",
            "8. Do not invent columns, values, or conditions not implied by the question.",
            "9. Prefer the simplest correct query.",
        ]
    else:
        dataset_context = (
            "You are working with the Spider dataset. "
            "Each database has multiple tables. "
            "Queries may require JOINs across tables — use only the foreign key relationships listed in the schema. "
            "Queries range from simple single-table lookups to multi-table JOINs with aggregation and ORDER BY."
        )
        rules = [
            "1. Output ONLY the SQL query — no explanation, no markdown, no code fences, nothing else.",
            "2. The query must start with SELECT.",
            "3. Use only tables and columns that appear in the schema.",
            "4. Join tables only when necessary, and only through valid foreign key relationships shown in the schema.",
            "5. Qualify column names with the table name (e.g. table.column) when the column exists in multiple tables.",
            "6. Preserve string values exactly from the question when used in WHERE conditions.",
            "7. Put text values in single quotes.",
            "8. Do not invent columns, values, joins, or conditions not implied by the question.",
            "9. Keep the query concise and correct for the schema.",
        ]

    rules_text = "\n".join(rules)
    return f"""{dataset_context}

Your task: convert the user's natural language question into exactly one valid SQL query.

Rules:
{rules_text}"""


def build_user_message(dataset: str, example: Dict[str, Any]) -> str:
    """Per-instance data (schema + question) — belongs in user turn."""
    question = example.get("question", "")
    schema_text = format_schema_text(example.get("schema", {}), dataset)
    return f"""Schema:
{schema_text}

Question:
{question}

SQL:
"""


def build_qwen_prompt(tokenizer, dataset: str, ex: Example, mode: str) -> str:
    messages = [
        {"role": "system", "content": build_system_prompt(dataset)},
        {"role": "user", "content": build_user_message(dataset, {"question": ex.question, "schema": ex.schema})},
    ]
    # enable_thinking=False explicitly disables thinking for non-thinking mode.
    # The except branch is for old tokenizers that don't support enable_thinking at all
    # (i.e. no thinking capability) — omitting the flag is safe there.
    enable_thinking = (mode == "thinking")
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def clean_sql(text: str, mode: str = "non-thinking") -> str:
    if text is None:
        return ""

    text = text.strip()

    if mode == "thinking":
        if "</think>" in text:
            text = text.split("</think>", 1)[1].strip()
        elif "select" in text.lower():
            text = text[text.lower().rfind("select"):].strip()

    text = re.sub(r"<\|.*?\|>", "", text)
    text = text.replace("```sql", "").replace("```", "").strip()
    text = text.replace("\n", " ")
    text = " ".join(text.split())

    lower_text = text.lower()
    if "select" not in lower_text:
        return ""

    text = text[lower_text.find("select"):].strip()

    if ";" in text:
        text = text.split(";", 1)[0].strip()

    # Reject obviously incomplete SQL
    bad_suffixes = ("where", "and", "or", "(", "=", ">", "<", ">=", "<=", "!=")
    if text.lower().endswith(bad_suffixes):
        return ""
    if text.count("(") != text.count(")"):
        return ""

    if not text.lower().startswith("select"):
        return ""

    return text


# Matches an optional <think>...</think> block at the start of generation.
# Allows any content inside <think> except the literal "</think>".
# Used to make constrained decoding compatible with thinking mode.
# Outlines uses an FSM-based regex compiler (interegular) that does NOT support
# lookaheads.  Use <[^/] instead of <(?!/think>) — identical logic but FSM-safe:
# the pattern stops consuming at "</think>" because "/" is excluded after "<".
_THINK_PREFIX_RE = r"(?:<think>(?:[^<]|<[^/])*</think>\s*)?"


def build_wikisql_regex(mode: str = "non-thinking") -> str:
    ident = r"[A-Za-z_][A-Za-z0-9_ ]*"
    number = r"-?\d+(?:\.\d+)?"
    quoted = r"'[^']*'"
    bare = r"[A-Za-z0-9_./()\ -]+"
    value = rf"(?:{quoted}|{number}|{bare})"
    op = r"(?:=|!=|<|>|<=|>=|LIKE)"
    agg = rf"(?:COUNT\(\*\)|COUNT\({ident}\)|MAX\({ident}\)|MIN\({ident}\)|SUM\({ident}\)|AVG\({ident}\))"
    # WikiSQL gold never uses SELECT * - exclude it to force column/agg selection
    select_expr = rf"(?:{agg}|{ident})"
    condition = rf"{ident}\s+{op}\s+{value}"
    # max 3 conditions to prevent hallucinated repetition loops
    where_opt = rf"(?:\s+WHERE\s+{condition}(?:\s+AND\s+{condition}){{0,2}})?"
    order_opt = rf"(?:\s+ORDER\s+BY\s+{ident}(?:\s+(?:ASC|DESC))?)?"
    # bounded LIMIT: 1-6 digits (up to 999999)
    limit_opt = r"(?:\s+LIMIT\s+[1-9][0-9]{0,5})?"
    sql = rf"SELECT\s+(?:DISTINCT\s+)?{select_expr}\s+FROM\s+table{where_opt}{order_opt}{limit_opt}\s*;?"
    prefix = _THINK_PREFIX_RE if mode == "thinking" else ""
    return prefix + sql


def build_spider_regex(table_names: List[str], mode: str = "non-thinking") -> str:
    table_union = "|".join(re.escape(t) for t in table_names) or "[A-Za-z_][A-Za-z0-9_]*"
    ident = r"[A-Za-z_][A-Za-z0-9_ ]*"
    number = r"-?\d+(?:\.\d+)?"
    quoted = r"'[^']*'"
    bare = r"[A-Za-z0-9_./()\-]+"
    value = rf"(?:{quoted}|{number}|{bare})"
    op = r"(?:=|!=|<|>|<=|>=|LIKE|IN|NOT IN)"
    select_expr = rf"(?:\*|{ident}|{ident}\.{ident}|{ident}\({ident}\))"
    condition = rf"{ident}(?:\.{ident})?\s+{op}\s+{value}"
    join_part = rf"(?:\s+JOIN\s+(?:{table_union})\s+ON\s+{condition}){{0,3}}"
    # max 4 conditions to prevent repetition loops
    where_opt = rf"(?:\s+WHERE\s+{condition}(?:\s+(?:AND|OR)\s+{condition}){{0,3}})?"
    order_opt = rf"(?:\s+ORDER\s+BY\s+{ident}(?:\.{ident})?(?:\s+(?:ASC|DESC))?)?"
    # bounded LIMIT: 1-6 digits
    limit_opt = r"(?:\s+LIMIT\s+[1-9][0-9]{0,5})?"
    sql = rf"SELECT\s+(?:DISTINCT\s+)?{select_expr}\s+FROM\s+(?:{table_union}){join_part}{where_opt}{order_opt}{limit_opt}\s*;?"
    prefix = _THINK_PREFIX_RE if mode == "thinking" else ""
    return prefix + sql


def escape_ebnf_literal(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def build_sql_ebnf(
    columns: List[str],
    tables: List[str],
    *,
    allow_or: bool,
    allow_join: bool,
    allow_star_select: bool = True,
    max_conditions: int = 5,
    mode: str = "non-thinking",
) -> str:
    col_rules = " | ".join(f'"{escape_ebnf_literal(c)}"' for c in columns if c != "*") or '"col"'
    table_rules = " | ".join(f'"{escape_ebnf_literal(t)}"' for t in tables) or '"table"'
    and_or = '("AND" | "OR")' if allow_or else '"AND"'
    join_ref = ' join_clause' if allow_join else ''
    join_clause = r'''
join_clause  ::= ( ws "JOIN" ws table_name ws "ON" ws condition )*
''' if allow_join else ""

    # select_expr: optionally exclude bare "*" (WikiSQL gold never uses SELECT *)
    if allow_star_select:
        select_expr_rule = 'agg_expr | column_name | "*"'
    else:
        select_expr_rule = 'agg_expr | column_name'

    # where_clause: explicit fixed-count alternatives to avoid ambiguity and
    # prevent infinite condition repetition during constrained decoding.
    _cond_alts = ['""']
    for _n in range(1, max_conditions + 1):
        _parts = ['ws "WHERE" ws condition'] + [f'ws {and_or} ws condition'] * (_n - 1)
        _cond_alts.append(' '.join(_parts))
    where_clause_rule = '\n    | '.join(_cond_alts)

    # For thinking mode: include an optional <think>...</think> prefix rule so the
    # grammar allows the reasoning chain before the SQL.  think_text matches any
    # char that is not "<", or a "<" not followed by "/" (to avoid triggering
    # "</think>" prematurely).
    if mode == "thinking":
        think_rules = r'''
think_block  ::= "<think>" think_text "</think>" ws_opt
think_text   ::= ( [^<] | "<" [^/] )*
ws_opt       ::= [ \t\n\r]*
'''
        root_rule = 'root         ::= think_block select_stmt | select_stmt'
    else:
        think_rules = ""
        root_rule = 'root         ::= select_stmt'

    grammar = rf'''
{root_rule}

select_stmt  ::= "SELECT" ws select_expr ws "FROM" ws table_name{join_ref} where_clause order_clause limit_clause

select_expr  ::= {select_expr_rule}
agg_expr     ::= agg_func ws? "(" ws? ("*" | column_name) ws? ")"
agg_func     ::= "COUNT" | "MAX" | "MIN" | "SUM" | "AVG"

where_clause ::= {where_clause_rule}
condition    ::= column_name ws op ws value
op           ::= "=" | "!=" | ">" | "<" | ">=" | "<=" | "LIKE" | "IN" | "NOT" ws "IN" | "IS" ws "NULL" | "IS" ws "NOT" ws "NULL"

value        ::= number | quoted_string | bare_word | "NULL"
number       ::= "-"? digit+ ("." digit+)?
quoted_string ::= "\"" dq_char* "\"" | "'" sq_char* "'"
dq_char      ::= [^"\\] | "\\" ["\\/bfnrt]
sq_char      ::= [^'\\] | "\\" ['\\/bfnrt]
bare_word    ::= bare_char+
bare_char    ::= [A-Za-z0-9_.%+\-]

order_clause ::= "" | ws "ORDER" ws "BY" ws column_name (ws ("ASC" | "DESC"))?
limit_clause ::= "" | ws "LIMIT" ws limit_val
limit_val    ::= [1-9] [0-9]? [0-9]? [0-9]? [0-9]? [0-9]?
{join_clause}
table_name   ::= {table_rules}
column_name  ::= {col_rules}
digit        ::= [0-9]
ws           ::= [ \t]+
{think_rules}'''
    return grammar.strip()


def build_xgrammar(dataset: str, raw_row: Dict[str, Any], mode: str = "non-thinking") -> str:
    if dataset == "wikisql":
        schema = raw_row.get("schema", {})
        cols = [c[1] for c in schema.get("column_names_original", []) if isinstance(c, (list, tuple)) and len(c) == 2]
        tables = schema.get("table_names_original", ["table"])
        # WikiSQL: never SELECT *, no OR, max 3 conditions to prevent repetition loops
        return build_sql_ebnf(cols, tables, allow_or=False, allow_join=False, allow_star_select=False, max_conditions=3, mode=mode)
    input_obj = raw_row.get("input", {})
    schema_json = input_obj.get("schema_json", {})
    cols = [c[1] for c in schema_json.get("column_names_original", []) if isinstance(c, (list, tuple)) and len(c) == 2]
    tables = schema_json.get("table_names_original", [])
    return build_sql_ebnf(cols, tables, allow_or=True, allow_join=True, allow_star_select=True, max_conditions=4, mode=mode)


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


def build_output_filename(dataset: str, mode: str, train_mode: str, constraint: str) -> str:
    return f"result_{dataset}_{mode}_{train_mode}_{constraint}_vllm.json"


def build_log_filename(dataset: str, mode: str, train_mode: str, constraint: str) -> str:
    return f"log_{dataset}_{mode}_{train_mode}_{constraint}_vllm.txt"


def make_sampling_params(
    max_new_tokens: int,
    constraint: str,
    raw_row: Dict[str, Any],
    ex: Example,
    dataset: str,
    mode: str = "non-thinking",
) -> SamplingParams:
    # For constrained decoding, cap tokens generously but not so high that the
    # model fills the budget with repeated grammar-legal tokens (e.g., LIMIT digits).
    # Thinking mode generates a <think>...</think> chain before SQL.  Measured
    # results show a mean of ~608 generated tokens even for WikiSQL, so 100-200
    # tokens is far too small.  Always use at least DEFAULT_MAX_NEW_TOKENS_THINKING
    # for thinking mode regardless of the constraint backend.
    if mode == "thinking":
        effective_tokens = max(max_new_tokens, DEFAULT_MAX_NEW_TOKENS_THINKING)
    else:
        effective_tokens = max_new_tokens

    params = SamplingParams(temperature=0.0, max_tokens=effective_tokens)

    if constraint == "none":
        return params

    if constraint == "outlines":
        regex_pattern = build_wikisql_regex(mode) if dataset == "wikisql" else build_spider_regex(ex.schema.get("table_names_original", []), mode)
        params.guided_decoding = GuidedDecodingParams(regex=regex_pattern)
        return params

    if constraint == "xgrammar":
        grammar = build_xgrammar(dataset, raw_row, mode)
        params.guided_decoding = GuidedDecodingParams(grammar=grammar)
        return params

    raise ValueError(f"Unsupported constraint: {constraint}")


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM text-to-SQL with backends: none / outlines / xgrammar")
    parser.add_argument("--dataset", choices=["wikisql", "spider"], required=True)
    parser.add_argument("--constraint", choices=["none", "outlines", "xgrammar"], required=True)
    parser.add_argument("--mode", choices=["thinking", "non-thinking"], default="non-thinking")
    parser.add_argument("--train-mode", choices=["zero", "few"], default="zero")
    parser.add_argument("--data-path", default=None, help="Override dataset JSON path")
    parser.add_argument(
        "--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Max new tokens (default {DEFAULT_MAX_NEW_TOKENS}). For --mode thinking the "
             f"effective minimum is {DEFAULT_MAX_NEW_TOKENS_THINKING} tokens.",
    )
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None, help="Randomly sample this many examples (seed=42)")
    parser.add_argument("--output", default=None)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Prompts to submit to vLLM per call (default: all at once)",
    )
    parser.add_argument(
                "--guided-decoding-backend",
                choices=["auto", "outlines", "xgrammar"],
                default=None,
                help="Structured decoding backend to use when constraint != none",
            )
    args = parser.parse_args()

    data_path = args.data_path or DEFAULT_DATA_PATHS[args.dataset]
    rows = load_json(data_path)
    print(f"Loaded {len(rows)} rows from {data_path}")
    if args.test_size is not None:
        random.seed(SEED)
        take_k = min(args.test_size, len(rows))
        rows = random.sample(rows, k=take_k)
        print(f"Sampled {len(rows)} rows (test_size={args.test_size}, seed={SEED})")
    if args.max_examples is not None:
        rows = rows[: args.max_examples]
        print(f"Truncated to first {len(rows)} rows due to max_examples={args.max_examples}")
    examples = to_examples(args.dataset, rows)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    llm_kwargs = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "seed": SEED,
    }

    try:
        import torch
        if torch.cuda.is_available():
            llm_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
    except Exception:
        pass

    # Fix backend at engine init time, because request-level backend override is not supported.
    if args.constraint != "none":
        backend_map = {
            "outlines": "outlines",
            "xgrammar": "xgrammar",
        }
        llm_kwargs["guided_decoding_backend"] = (
            args.guided_decoding_backend or backend_map[args.constraint]
        )

    llm_kwargs["dtype"] = "float32"
    llm = LLM(**llm_kwargs)

    print(f"Loaded model via vLLM: {args.model}")
    print(f"Constraint backend: {args.constraint}")

    log_path = build_log_filename(args.dataset, args.mode, args.train_mode, args.constraint)
    log_fh = open(log_path, "w", encoding="utf-8")

    def log(msg: str = "") -> None:
        """Print to both terminal and log file."""
        print(msg)
        print(msg, file=log_fh)

    try:
        # ── Build all prompts & per-example sampling params ──────────────────────
        all_prompts: List[str] = []
        all_params: List[SamplingParams] = []
        for ex, raw_row in zip(examples, rows):
            all_prompts.append(build_qwen_prompt(tokenizer, args.dataset, ex, args.mode))
            all_params.append(make_sampling_params(args.max_new_tokens, args.constraint, raw_row, ex, args.dataset, args.mode))

        log("=" * 60)
        log("PROMPT (example 1):")
        log(all_prompts[0])
        log("=" * 60)

        # ── Batched generation ────────────────────────────────────────────────────
        batch_size = args.batch_size or len(all_prompts)
        raw_texts: List[str] = []
        num_tokens_list: List[int] = []
        latency_list: List[float] = []

        for batch_start in range(0, len(all_prompts), batch_size):
            batch_prompts = all_prompts[batch_start : batch_start + batch_size]
            batch_params  = all_params [batch_start : batch_start + batch_size]
            t0 = time.perf_counter()
            outputs = llm.generate(batch_prompts, sampling_params=batch_params, use_tqdm=False)
            batch_ms = (time.perf_counter() - t0) * 1000.0
            per_ms = batch_ms / len(batch_prompts)
            for out in outputs:
                o = out.outputs[0]
                raw_texts.append(o.text)
                num_tokens_list.append(len(o.token_ids))
                latency_list.append(per_ms)
            end_idx = min(batch_start + batch_size, len(all_prompts))
            log(f"Generated {end_idx}/{len(all_prompts)} "
                f"(batch {batch_start // batch_size + 1}: {batch_ms:.0f} ms "
                f"for {len(batch_prompts)} examples, {per_ms:.0f} ms/example amortized)")

        log("=" * 60)
        log("RAW OUTPUT (example 1):")
        log(raw_texts[0])
        log("=" * 60)

        # ── Post-process ──────────────────────────────────────────────────────────
        results: List[Dict[str, Any]] = []
        for idx, (ex, raw_text, num_tokens, latency_ms) in enumerate(
            zip(examples, raw_texts, num_tokens_list, latency_list), start=1
        ):
            predicted_sql = clean_sql(raw_text, mode=args.mode)
            thinking_suppressed = (
                args.mode == "thinking"
                and args.constraint != "none"
                and "<think>" not in raw_text
            )
            result = {
                "instance_id": ex.instance_id,
                "question": ex.question,
                "gold_sql": ex.gold_sql,
                "predicted_sql": predicted_sql,
                "latency_ms": round(float(latency_ms), 3),
                "num_tokens_generated": int(num_tokens),
                "constraint": args.constraint,
                "mode": args.mode,
                "thinking_suppressed": thinking_suppressed,
            }
            results.append(result)
            if idx <= 3:
                log("=" * 60)
                log(f"Example {idx}")
                log(f"Question: {ex.question}")
                log(f"Gold SQL: {ex.gold_sql}")
                log(f"Predicted SQL: {predicted_sql}")
                log(f"Constraint: {args.constraint}")
                if thinking_suppressed:
                    log("[NOTE] thinking suppressed despite grammar allowing it — model skipped <think>")
                log(f"Latency (ms, amortized): {round(latency_ms, 2)}")
                log(f"Generated tokens: {num_tokens}")

        out_path = args.output or build_output_filename(args.dataset, args.mode, args.train_mode, args.constraint)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        log(f"Saved results to: {out_path}")
        log(f"Saved log     to: {log_path}")
    finally:
        log_fh.close()


if __name__ == "__main__":
    main()