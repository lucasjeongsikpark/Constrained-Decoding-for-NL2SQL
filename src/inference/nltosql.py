from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import xgrammar as xgr

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
        lines.append("COLUMNS:")
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
        return "You are a precise text-to-SQL assistant."

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
    return (
        "You are working with the Spider dataset. "
        "Each database has multiple tables. "
        "Queries may require JOINs across tables — use only the foreign key relationships listed in the schema. "
        "Queries range from simple single-table lookups to multi-table JOINs with aggregation and ORDER BY.\n\n"
        "Your task: convert the user's natural language question into exactly one valid SQL query.\n\n"
        f"Rules:\n{rules_text}"
    )


def build_user_message(dataset: str, example: Dict[str, Any]) -> str:
    """Per-instance data (schema + question) — belongs in user turn."""
    question = example.get("question", "")
    schema_text = format_schema_text(example.get("schema", {}), dataset)
    if dataset == "wikisql":
        instance_id = example.get("instance_id", "")
        return f"""You are a text-to-SQL system.

Task:
Convert the question into exactly one SQL query for the given schema.

Rules:
1. Output exactly one SQL query and nothing else.
2. The query must start with SELECT.
3. Use only the table name: table
4. Use only columns that appear in the schema, using the exact column name as listed.
5. Always wrap string values in single quotes.
6. Lowercase all string values in WHERE conditions (e.g. 'john smith' not 'John Smith').
7. Only add WHERE conditions for constraints explicitly stated in the question.
8. SELECT only the single column that directly answers the question — do not add extra columns.
9. Do not invent columns, values, or conditions not present in the question.
10. Do not add explanation.
11. Prefer the simplest correct WikiSQL-style query.

Instance ID: {instance_id}

Schema:
{schema_text}

Question:
{question}

SQL:
"""
    return f"""Schema:
{schema_text}

Question:
{question}

SQL:
"""


def build_qwen_prompt(tokenizer, dataset: str, ex: Example, mode: str) -> str:
    messages = [
        {"role": "system", "content": build_system_prompt(dataset)},
        {"role": "user", "content": build_user_message(dataset, {"instance_id": ex.instance_id, "question": ex.question, "schema": ex.schema})},
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
    # Matches notebook's build_better_wikisql_regex logic.
    ident = r"[A-Za-z_][A-Za-z0-9_ ]*"
    number = r"-?\d+(?:\.\d+)?"
    string = r"'[^']*'"
    value = rf"(?:{string}|{number})"
    op = r"(?:=|!=|>|<|>=|<=)"
    agg = rf"(?:COUNT\(\*\)|COUNT\({ident}\)|MAX\({ident}\)|MIN\({ident}\)|SUM\({ident}\)|AVG\({ident}\))"
    select_expr = rf"(?:\*|{ident}|{agg})"
    condition = rf"{ident}\s+{op}\s+{value}"
    sql = (
        rf"SELECT\s+{select_expr}\s+FROM\s+table"
        rf"(?:\s+WHERE\s+{condition}"
        rf"(?:\s+(?:AND|OR)\s+{condition}){{0,1}})?"
        rf"\s*;?"
    )
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


def build_sql_ebnf_extend(
    columns: List[str],
    table: str,
    *,
    allow_or: bool = True,
    allow_join: bool = False,
    multi_table: Optional[List[str]] = None,
) -> str:
    """EBNF grammar matching the notebook's build_sql_ebnf_extend logic."""
    col_rules = " | ".join(f'"{escape_ebnf_literal(c)}"' for c in columns if c != "*") or '"col"'
    tables = multi_table if multi_table else [table]
    table_rules = " | ".join(f'"{escape_ebnf_literal(t)}"' for t in tables) or '"table"'
    and_or = '("AND" | "OR")' if allow_or else '"AND"'

    if allow_join and multi_table:
        join_clause = (
            '\njoin_clause  ::= "" | (ws join_type ws "JOIN" ws table_name ws "ON" ws column_name ws "=" ws column_name)+'
            '\njoin_type    ::= "" | "INNER" | "LEFT" | "RIGHT" | "FULL OUTER"'
        )
        join_ref = " join_clause"
    else:
        join_clause = ""
        join_ref = ""

    grammar = rf"""
root         ::= select_stmt

select_stmt  ::= "SELECT" ws select_expr ws "FROM" ws table_name{join_ref} where_clause order_clause limit_clause

select_expr  ::= agg_expr | column_name | "*"
agg_expr     ::= agg_func ws "(" ws ("*" | column_name) ws ")"
agg_func     ::= "COUNT" | "MAX" | "MIN" | "SUM" | "AVG"

where_clause ::= "" | ws "WHERE" ws condition (ws {and_or} ws condition)*
condition    ::= column_name ws op ws value

op           ::= "=" | "!=" | ">" | "<" | ">=" | "<=" | "LIKE" | "IN" | "NOT IN" | "IS NULL" | "IS NOT NULL"

value        ::= number | quoted_string | bare_word | "NULL"
number       ::= "-"? [0-9]+ ("." [0-9]+)?
quoted_string ::= "\"" dq_char* "\"" | "'" sq_char* "'"
dq_char      ::= [^"\\ ] | "\\" ["\\/bfnrt]
sq_char      ::= [^'\\ ] | "\\" ['\\/bfnrt]
bare_word    ::= [A-Za-z0-9_.%+\-]+

order_clause ::= "" | ws "ORDER" ws "BY" ws column_name (ws ("ASC" | "DESC"))?
limit_clause ::= "" | ws "LIMIT" ws [0-9]+{join_clause}

table_name   ::= {table_rules}
column_name  ::= {col_rules}

ws           ::= [ \t]+"""
    return grammar.strip()


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
        table = (schema.get("table_names_original") or ["table"])[0]
        return build_sql_ebnf(cols, [table], allow_or=False, allow_join=False, allow_star_select=True, max_conditions=4, mode=mode)
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
    return f"results/result_{dataset}_{mode}_{train_mode}_{constraint}.json"


def build_log_filename(dataset: str, mode: str, train_mode: str, constraint: str) -> str:
    return f"logs/log_{dataset}_{mode}_{train_mode}_{constraint}.txt"


class OutlinesRunner:
    """Outlines wrapper: builds a regex logits processor and runs HF model.generate() directly.
    This avoids outlines' internal generation loop which crashes on special tokens (e.g. Qwen3 EOS=0).
    """
    def __init__(self, model, tokenizer):
        from outlines.backends.outlines_core import OutlinesCoreBackend
        from outlines.models.transformers import Transformers

        self._model = model
        self._tokenizer_hf = tokenizer
        outlines_model = Transformers(model, tokenizer)
        self._backend = OutlinesCoreBackend(outlines_model)

    def get_logits_processor(self, regex_pattern: str):
        lp = self._backend.get_regex_logits_processor(regex_pattern)
        eos_id = self._tokenizer_hf.eos_token_id

        class _SafeLP:
            """Wrap outlines logits processor: if FSM has no transition (e.g. for EOS token ID),
            fall back to allowing only EOS so generation can terminate cleanly."""
            def __init__(self, inner, eos_token_id):
                self._inner = inner
                self._eos = eos_token_id

            def __call__(self, input_ids, scores):
                try:
                    return self._inner(input_ids, scores)
                except (ValueError, RuntimeError):
                    result = torch.full_like(scores, float("-inf"))
                    result[:, self._eos] = 0.0
                    return result

        return _SafeLP(lp, eos_id)

    def generate_regex(self, prompt: str, regex_pattern: str, max_new_tokens: int = 128, generation_kwargs: dict = None):
        lp = self.get_logits_processor(regex_pattern)
        inputs = self._tokenizer_hf(prompt, return_tensors="pt").to(self._model.device)
        kwargs = {"do_sample": False, "max_new_tokens": max_new_tokens}
        if generation_kwargs:
            kwargs.update(generation_kwargs)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self._model.generate(**inputs, logits_processor=[lp], **kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        gen_tokens = out[0][inputs["input_ids"].shape[1]:]
        text = self._tokenizer_hf.decode(gen_tokens, skip_special_tokens=True).strip()
        num_tokens = int(gen_tokens.shape[0])
        return text, num_tokens, latency_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="HuggingFace text-to-SQL with backends: none / outlines / xgrammar")
    parser.add_argument("--dataset", choices=["wikisql", "spider"], required=True)
    parser.add_argument("--constraint", choices=["none", "outlines", "xgrammar"], required=True)
    parser.add_argument("--mode", choices=["thinking", "non-thinking"], default="non-thinking")
    parser.add_argument("--train-mode", choices=["zero", "few"], default="zero")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", default=None)
    parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args()

    data_path = args.data_path or DEFAULT_DATA_PATHS[args.dataset]
    rows = load_json(data_path)
    print(f"Loaded {len(rows)} rows from {data_path}")
    if args.test_size is not None:
        random.seed(SEED)
        rows = random.sample(rows, k=min(args.test_size, len(rows)))
        print(f"Sampled {len(rows)} rows (seed={SEED})")
    if args.max_examples is not None:
        rows = rows[: args.max_examples]
    examples = to_examples(args.dataset, rows)

    effective_tokens = (
        max(args.max_new_tokens, DEFAULT_MAX_NEW_TOKENS_THINKING)
        if args.mode == "thinking" else args.max_new_tokens
    )
    generation_kwargs: Dict[str, Any] = {"do_sample": False, "max_new_tokens": effective_tokens}

    # Resolve Hub model ID to local snapshot path to avoid network version checks
    _model_path = args.model
    if not os.path.isdir(args.model):
        _cache_base = os.path.join(
            os.path.expanduser("~"), ".cache", "huggingface", "hub",
            "models--" + args.model.replace("/", "--"), "snapshots"
        )
        if os.path.isdir(_cache_base):
            _snapshots = sorted(os.listdir(_cache_base))
            if _snapshots:
                _model_path = os.path.join(_cache_base, _snapshots[-1])
    _hf_kwargs = {"local_files_only": True} if _model_path != args.model else {}
    tokenizer = AutoTokenizer.from_pretrained(_model_path, trust_remote_code=True, **_hf_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"  # left-padding required for batched generation
    model = AutoModelForCausalLM.from_pretrained(
        _model_path,
        trust_remote_code=True,
        dtype="auto",
        device_map="auto",
        **_hf_kwargs,
    )
    model.eval()
    print(f"Loaded model: {args.model}  device: {model.device}")
    print(f"Constraint: {args.constraint}  batch_size: {args.batch_size}")

    # ── Init constraint backend ───────────────────────────────────────────────
    grammar_compiler = None
    outlines_runner = None
    if args.constraint == "xgrammar":
        config = AutoConfig.from_pretrained(_model_path, **_hf_kwargs)
        info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=config.vocab_size)
        grammar_compiler = xgr.GrammarCompiler(info, max_threads=8)
    elif args.constraint == "outlines":
        outlines_runner = OutlinesRunner(model, tokenizer)

    log_path = build_log_filename(args.dataset, args.mode, args.train_mode, args.constraint)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_fh = open(log_path, "w", encoding="utf-8")

    def log(msg: str = "") -> None:
        print(msg)
        print(msg, file=log_fh)

    try:
        prompt0 = build_qwen_prompt(tokenizer, args.dataset, examples[0], args.mode)
        log("=" * 60)
        log("PROMPT (example 1):")
        log(prompt0)
        log("=" * 60)
        log(f"Dataset: {args.dataset}  Mode: {args.mode}  Constraint: {args.constraint}")

        results: List[Dict[str, Any]] = []
        global_idx = 0

        for batch_start in range(0, len(examples), args.batch_size):
            batch_exs  = examples[batch_start : batch_start + args.batch_size]
            batch_rows = rows[batch_start : batch_start + args.batch_size]
            batch_prompts = [
                build_qwen_prompt(tokenizer, args.dataset, ex, args.mode)
                for ex in batch_exs
            ]

            if args.constraint == "xgrammar":
                grammars = [build_xgrammar(args.dataset, r, args.mode) for r in batch_rows]
                compiled_list = [grammar_compiler.compile_grammar(g) for g in grammars]
                # xgrammar supports List[CompiledGrammar] for per-item grammars in a batch
                lp_arg = compiled_list if len(compiled_list) > 1 else compiled_list[0]
                logits_processor = xgr.contrib.hf.LogitsProcessor(lp_arg)
                inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
                t0 = time.perf_counter()
                with torch.no_grad():
                    out = model.generate(**inputs, **generation_kwargs, logits_processor=[logits_processor])
                batch_latency_ms = (time.perf_counter() - t0) * 1000.0
                per_latency = batch_latency_ms / len(batch_exs)
                input_len = inputs["input_ids"].shape[1]
                batch_raw, batch_ntok, batch_latencies = [], [], []
                for i in range(len(batch_exs)):
                    gen_tokens = out[i][input_len:]
                    batch_raw.append(tokenizer.decode(gen_tokens, skip_special_tokens=True).strip())
                    batch_ntok.append(int(gen_tokens.shape[0]))
                    batch_latencies.append(per_latency)

            elif args.constraint == "outlines":
                batch_raw, batch_ntok, batch_latencies = [], [], []
                for ex, raw_row in zip(batch_exs, batch_rows):
                    prompt = build_qwen_prompt(tokenizer, args.dataset, ex, args.mode)
                    regex = (
                        build_wikisql_regex(args.mode)
                        if args.dataset == "wikisql"
                        else build_spider_regex(ex.schema.get("table_names_original", []), args.mode)
                    )
                    raw_text, num_tokens, latency_ms = outlines_runner.generate_regex(
                        prompt, regex, max_new_tokens=effective_tokens
                    )
                    batch_raw.append(raw_text)
                    batch_ntok.append(num_tokens)
                    batch_latencies.append(latency_ms)

            else:  # none
                inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
                t0 = time.perf_counter()
                with torch.no_grad():
                    out = model.generate(**inputs, **generation_kwargs)
                batch_latency_ms = (time.perf_counter() - t0) * 1000.0
                per_latency = batch_latency_ms / len(batch_exs)
                input_len = inputs["input_ids"].shape[1]
                batch_raw, batch_ntok, batch_latencies = [], [], []
                for i in range(len(batch_exs)):
                    gen_tokens = out[i][input_len:]
                    batch_raw.append(tokenizer.decode(gen_tokens, skip_special_tokens=True).strip())
                    batch_ntok.append(int(gen_tokens.shape[0]))
                    batch_latencies.append(per_latency)

            for ex, raw_row, raw_text, num_tokens, latency_ms in zip(
                batch_exs, batch_rows, batch_raw, batch_ntok, batch_latencies
            ):
                global_idx += 1
                predicted_sql = clean_sql(raw_text, mode=args.mode)
                thinking_suppressed = (
                    args.mode == "thinking"
                    and args.constraint != "none"
                    and "<think>" not in raw_text
                )
                result: Dict[str, Any] = {
                    "instance_id": ex.instance_id,
                    "question": ex.question,
                    "gold_sql": ex.gold_sql,
                    "predicted_sql": predicted_sql,
                    "latency_ms": round(float(latency_ms), 3),
                    "num_tokens_generated": num_tokens,
                    "constraint": args.constraint,
                    "mode": args.mode,
                    "thinking_suppressed": thinking_suppressed,
                }
                if args.dataset == "wikisql":
                    result["db"] = raw_row.get("db", "")
                    result["schema"] = raw_row.get("schema", {})
                results.append(result)

                if global_idx <= 3:
                    log("=" * 60)
                    log(f"Example {global_idx}")
                    log(f"Question: {ex.question}")
                    log(f"Gold SQL: {ex.gold_sql}")
                    log(f"Predicted SQL: {predicted_sql}")
                    log(f"Latency (ms): {round(latency_ms, 2)}")
                    log(f"Generated tokens: {num_tokens}")
                if global_idx % 10 == 0 or global_idx == len(examples):
                    log(f"Done {global_idx}/{len(examples)}")

        out_path = args.output or build_output_filename(args.dataset, args.mode, args.train_mode, args.constraint)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        log(f"Saved results to: {out_path}")
        log(f"Saved log     to: {log_path}")
    finally:
        log_fh.close()


if __name__ == "__main__":
    main()