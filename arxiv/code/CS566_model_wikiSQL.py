"""
This python file is designed to supports these experiment modes:
- zero-shot + thinking
- zero-shot + non-thinking
- zero-shot + thinking + outlines
- zero-shot + non-thinking + outlines

The input 
- data wikisql_test / spider_test
- model none / checkpoint
- constrained none / xgrammar / lmfe / outlines 

The output JSON always contains these six fields:

- `instance_id`
- `question`
- `gold_sql`
- `predicted_sql`
- `latency_ms`
- `num_tokens_generated`

!pip install -q transformers==4.51.3 accelerate sentencepiece
!pip install -q outlines
!pip install -q lm-format-enforcer
!pip install -q xgrammar
"""

import argparse
import os
import re
import json
import time
from typing import Dict, Any, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import xgrammar as xgr

# For SQL generation, deterministic decoding is usually more stable.
TEMPERATURE = None
TOP_P = None
TOP_K = None
MIN_P = None
MAX_NEW_TOKENS = 100
MODE = "non-thinking"

# Load dataset function
def load_dataset(input_path: str) -> List[Dict[str, Any]]:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples.")
    if len(data) > 0:
        print("Sample keys:", list(data[0].keys()))
    return data

# Load model functions
def load_model(model_type: str, model_name: str, checkpoint_path: str = None, trust_remote_code=True):
    if model_type == "none":
        model_path = model_name

    elif model_type == "checkpoint":
        if checkpoint_path is None:
            raise ValueError("checkpoint_path must be provided for checkpoint model")
        model_path = checkpoint_path

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        torch_dtype="auto",
        device_map="auto",
    )

    model.eval()
    print(f"Loaded {model_type} model from: {model_path}")

    return tokenizer, model

# Outlines wrapper
class OutlinesRunner:
    """
    A version-compatible Outlines wrapper.
    """
    def __init__(self, model, tokenizer):
        import outlines
        from outlines.types import Regex

        self.outlines = outlines
        self.Regex = Regex
        self.tokenizer = tokenizer
        self.outlines_model = outlines.from_transformers(model, tokenizer)

    def generate_regex(
        self,
        prompt: str,
        regex_pattern: str,
        max_new_tokens: int = 128,
    ):
        start_time = time.perf_counter()

        output = self.outlines_model(
            prompt,
            self.Regex(regex_pattern),
            max_new_tokens=max_new_tokens,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        text = output if isinstance(output, str) else str(output)
        num_tokens_generated = len(
            self.tokenizer.encode(text, add_special_tokens=False)
        )

        return text, num_tokens_generated, latency_ms
    
# LMFE wrapper
class LMFERunner:
    """
    LMFE-style constrained decoding wrapper.
    """
    def __init__(self, model, tokenizer):
        from lmformatenforcer import RegexParser
        from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

        self.tokenizer = tokenizer
        self.LMFE_model = model
        self.RegexParser = RegexParser
        self.build_transformers_prefix_allowed_tokens_fn = build_transformers_prefix_allowed_tokens_fn

    def generate_with_prefix_fn(
        self,
        prompt: str,
        regex_pattern: str,
        generation_kwargs: Dict[str, Any],
        max_new_tokens: int = 128,
        ):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.LMFE_model.device)
        input_len = inputs["input_ids"].shape[1]

        start_time = time.perf_counter()

        parser = self.RegexParser(regex_pattern)
        prefix_fn = self.build_transformers_prefix_allowed_tokens_fn(self.tokenizer, parser)

        with torch.no_grad():
            outputs = self.LMFE_model.generate(
                **inputs,
                **generation_kwargs,
                prefix_allowed_tokens_fn=prefix_fn,
            )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        generated_ids = outputs[0][input_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        num_tokens_generated = int(generated_ids.shape[0])

        return text, num_tokens_generated, latency_ms
    

# XGrammar warapper
class XGrammarRunner:
    """
    XGrammar-style constrained decoding wrapper.
    """
    def __init__(self, tokenizer, model_name):
        config = AutoConfig.from_pretrained(model_name)
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=config.vocab_size)
        self.grammar_compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=8)

    def generate_xgrammar_extend(self, model, tokenizer, device, prompt: str, example: dict, generation_kwargs: Dict[str, Any]):
        ebnf = from_wikisql(example)
        compiled_grammar = self.grammar_compiler.compile_grammar(ebnf)
        xgr_logits_processor = xgr.contrib.hf.LogitsProcessor(compiled_grammar)

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        t0 = time.time()
        outputs = model.generate(
            **inputs,
            **generation_kwargs,
            logits_processor=[xgr_logits_processor],
        )
        latency = time.time() - t0
        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        pred_sql = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        return pred_sql, int(gen_tokens.shape[0]), latency

def get_default_sampling(mode: str) -> Dict[str, Any]:
    """
    Use deterministic decoding for more stable SQL generation.
    """
    return {
        "do_sample": False,
    }

def build_generation_kwargs(
    mode: str,
    max_new_tokens: int,
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    min_p: Optional[float],
) -> Dict[str, Any]:
    kwargs = get_default_sampling(mode)
    kwargs["max_new_tokens"] = max_new_tokens

    if kwargs.get("do_sample", False):
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if top_k is not None:
            kwargs["top_k"] = top_k
        if min_p is not None:
            kwargs["min_p"] = min_p

    return kwargs

def format_schema_text(schema: Dict[str, Any]) -> str:
    """
    Convert schema metadata into a simple text format that is easy for the model to read.
    """
    if not schema:
        return "No schema provided."

    table_names = schema.get("table_names_original") or schema.get("table_names") or []
    column_names = schema.get("column_names_original") or schema.get("column_names") or []
    column_types = schema.get("column_types") or ["unknown"] * len(column_names)

    lines = []

    for table_id, table_name in enumerate(table_names):
        lines.append(f"TABLE: {table_name}")
        lines.append("COLUMNS:")
        for idx, pair in enumerate(column_names):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            tid, col_name = pair
            if tid != table_id:
                continue
            if col_name == "*":
                continue

            col_type = column_types[idx] if idx < len(column_types) else "unknown"
            lines.append(f"- {col_name} [{col_type}]")
        lines.append("")

    text = "\n".join(lines).strip()
    return text if text else "No schema provided."

def build_plain_prompt(example: Dict[str, Any]) -> str:
    """
    Build a stronger prompt for text-to-SQL generation.
    """
    instance_id = example.get("instance_id", "")
    question = example.get("question", "")
    schema_text = format_schema_text(example.get("schema", {}))

    prompt = f"""You are a text-to-SQL system.

Task:
Convert the question into exactly one SQL query for the given schema.

Rules:
1. Output exactly one SQL query and nothing else.
2. The query must start with SELECT.
3. Use only the table name: table
4. Use only columns that appear in the schema.
5. Preserve string values exactly from the question when possible.
6. Put text values in single quotes.
7. Do not invent columns, values, or conditions.
8. Do not add explanation.
9. Prefer the simplest correct WikiSQL-style query.

Instance ID: {instance_id}

Schema:
{schema_text}

Question:
{question}

SQL:
"""
    return prompt

def build_qwen_prompt(tokenizer, example: Dict[str, Any], mode: str) -> str:
    """
    Build a chat-formatted prompt for Qwen.
    """
    user_prompt = build_plain_prompt(example)

    messages = [
        {"role": "system", "content": "You are a precise text-to-SQL assistant."},
        {"role": "user", "content": user_prompt},
    ]

    enable_thinking = (mode == "thinking")

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return prompt

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

def build_better_wikisql_regex() -> str:
    # Regex upgraded using the teammate grammar idea while staying in outlines.
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

def generate_plain(
    model,
    tokenizer,
    prompt: str,
    generation_kwargs: Dict[str, Any],
):
    """
    Plain Hugging Face generation without constrained decoding.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    start_time = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generation_kwargs,
        )

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    generated_ids = outputs[0][input_len:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    num_tokens_generated = int(generated_ids.shape[0])

    return text, num_tokens_generated, latency_ms

def validate_output_record(record: Dict[str, Any]) -> None:
    """
    Ensure required keys are included in the output record.
    """
    required_keys = [
        "instance_id",
        "question",
        "gold_sql",
        "predicted_sql",
        "latency_ms",
        "num_tokens_generated",
    ]

    missing = [k for k in required_keys if k not in record]
    if missing:
        raise ValueError(f"Missing required keys: {missing}")
    
def build_output_filename(
    dataset_name: str,
    mode: str,
    train_mode: str,
    constraint_method: str,
) -> str:
    """
    Required naming convention:
    result_{dataset name}_{thinking mode}_{zero/ft}_{constrained decoding methods}.json
    """
    return f"result_{dataset_name}_{mode}_{train_mode}_{constraint_method}.json"

def escape_ebnf_literal(s: str) -> str:
    escaped = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'

def build_sql_ebnf_extend(
    columns: list[str],
    table: str,
    *,
    allow_or: bool = True,
    allow_subquery: bool = False,
    allow_join: bool = False,
    multi_table: list[str] | None = None,
) -> str:
    """
    通用 NL2SQL grammar，schema 信息动态注入。

    Parameters
    ----------
    columns      : 当前 schema 的列名列表
    table        : 主表名（单表场景）
    allow_or     : WHERE 是否允许 OR（WikiSQL 不需要，Spider 需要）
    allow_join   : 是否允许 JOIN（多表场景）
    multi_table  : JOIN 场景下的所有表名，None 则只用 table
    """
    col_rules   = " | ".join(escape_ebnf_literal(c) for c in columns if c != "*")
    tables      = multi_table if multi_table else [table]
    table_rules = " | ".join(escape_ebnf_literal(t) for t in tables)

    # WHERE
    and_or = '("AND" | "OR")' if allow_or else '"AND"'

    # JOIN
    if allow_join and multi_table:
        join_clause = r"""
join_clause  ::= "" | (ws join_type ws "JOIN" ws table_name ws "ON" ws column_name ws "=" ws column_name)+
join_type    ::= "" | "INNER" | "LEFT" | "RIGHT" | "FULL OUTER"
"""
        join_ref = " join_clause"
    else:
        join_clause = ""
        join_ref    = ""

    grammar = rf"""
root         ::= select_stmt

select_stmt  ::= "SELECT" ws select_expr ws "FROM" ws table_name{join_ref} where_clause order_clause limit_clause

# ── SELECT ───────────────────────────────────────────────────
select_expr  ::= agg_expr | column_name | "*"
agg_expr     ::= agg_func ws "(" ws ("*" | column_name) ws ")"
agg_func     ::= "COUNT" | "MAX" | "MIN" | "SUM" | "AVG"

# ── WHERE ────────────────────────────────────────────────────
where_clause ::= "" | ws "WHERE" ws condition (ws {and_or} ws condition)*
condition    ::= column_name ws op ws value

op           ::= "=" | "!=" | ">" | "<" | ">=" | "<=" | "LIKE" | "IN" | "NOT IN" | "IS NULL" | "IS NOT NULL"

# ── value ────────────────────────────────────────────────────
value        ::= number | quoted_string | bare_word | "NULL"
number       ::= "-"? [0-9]+ ("." [0-9]+)?
quoted_string ::= "\"" dq_char* "\"" | "'" sq_char* "'"
dq_char      ::= [^"\\] | "\\" ["\\/bfnrt]
sq_char      ::= [^'\\] | "\\" ['\\/bfnrt]
bare_word    ::= [A-Za-z0-9_.%+\-]+

# ── ORDER BY ─────────────────────────────────────────────────
order_clause ::= "" | ws "ORDER" ws "BY" ws column_name (ws ("ASC" | "DESC"))?

# ── LIMIT ────────────────────────────────────────────────────
limit_clause ::= "" | ws "LIMIT" ws [0-9]+

# ── JOIN（可选）──────────────────────────────────────────────{join_clause}
# ── 白名单 ───────────────────────────────────────────────────
table_name   ::= {table_rules}
column_name  ::= {col_rules}

ws           ::= [ \t]+
"""
    return grammar.strip()

def from_wikisql(example: dict) -> str:
    schema  = example["schema"]
    columns = [c[1] for c in schema["column_names_original"]]
    table   = schema["table_names_original"][0]
    return build_sql_ebnf_extend(columns, table, allow_or=False, allow_join=False)


# inference logic
def run_baseline(constraint_type, model, tokenizer, data, output_path):
    generation_kwargs = build_generation_kwargs(
        mode=MODE,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        min_p=MIN_P,
    )
    outlines_runner = None
    LMFE_runner = None
    xgrammar_runner = None
    if constraint_type == "outlines":   
        outlines_runner = OutlinesRunner(model, tokenizer)
    elif constraint_type == "lmfe":
        LMFE_runner = LMFERunner(model, tokenizer)
    elif constraint_type == "xgrammar":
        xgrammar_runner = XGrammarRunner(tokenizer, "Qwen/Qwen3-0.6B")


    results = []
    for idx, example in enumerate(data, start=1):
        prompt = build_qwen_prompt(tokenizer, example, MODE)

        if constraint_type == "outlines":
            regex_pattern = build_better_wikisql_regex()
            raw_text, num_tokens_generated, latency_ms = outlines_runner.generate_regex(
                prompt=prompt,
                regex_pattern=regex_pattern,
                max_new_tokens=MAX_NEW_TOKENS,
            )
        elif constraint_type == "lmfe":
            regex_pattern = build_better_wikisql_regex()
            raw_text, num_tokens_generated, latency_ms = LMFE_runner.generate_with_prefix_fn(
                prompt=prompt,
                regex_pattern=regex_pattern,
                generation_kwargs=generation_kwargs,
                max_new_tokens=MAX_NEW_TOKENS,
            )
        elif constraint_type == "xgrammar":
            raw_text, num_tokens_generated, latency_ms = xgrammar_runner.generate_xgrammar_extend(
            model=model,
            tokenizer=tokenizer,
            device=model.device,
            prompt=prompt,
            example=example,
            generation_kwargs=generation_kwargs,
        )
        elif constraint_type == "none":
            raw_text, num_tokens_generated, latency_ms = generate_plain(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                generation_kwargs=generation_kwargs,
            )

        predicted_sql = clean_sql(raw_text, mode=MODE)

        # Keep all original fields from the input example.
        result = dict(example)

        # Map gold_sql_query -> gold_sql
        gold_sql_value = example.get("gold_sql_query", "")

        result.update({
            "instance_id": example.get("instance_id", ""),
            "question": example.get("question", ""),
            "gold_sql": gold_sql_value,
            "predicted_sql": predicted_sql,
            "latency_ms": round(float(latency_ms), 3),
            "num_tokens_generated": int(num_tokens_generated),
        })

        # Remove the old field if it exists
        if "gold_sql_query" in result:
            del result["gold_sql_query"]

        validate_output_record(result)
        results.append(result)

        if idx <= 3:
            print("=" * 80)
            print(f"Example {idx}")
            print("Question:", result["question"])
            print("Gold SQL:", result["gold_sql"])
            print("Predicted SQL:", result["predicted_sql"])
            print("Latency (ms):", result["latency_ms"])
            print("Generated tokens:", result["num_tokens_generated"])
        # if idx == 3:
        #   break

        if idx % 5 == 0 or idx == len(data):
            print(f"Done {idx}/{len(data)}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Saved JSON to:", output_path)

# Main function
def main():
    parser = argparse.ArgumentParser(
        description="Text-to-SQL Inference Entry"
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        choices=["wikisql_test", "spider_test"]
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["none", "checkpoint"]
    )

    parser.add_argument(
        "--constrained",
        type=str,
        required=True,
        choices=["none", "xgrammar", "lmfe", "outlines"]
    )

    args = parser.parse_args()

    # ===== model choice =====
    if args.model == "checkpoint":
        print("Using fine-tuned checkpoint model")
    else:
        print("Using base model")

    print(f"Dataset: {args.data}")
    print(f"Constraint: {args.constrained}")

    # ===== data loading =====
    if args.data == "wikisql_test":
        dataset = load_dataset("../data/wikisql_train_sample50.json")
    elif args.data == "spider_test":
        dataset = load_dataset("../data/spider_test.json")
    else:
        raise ValueError("Unsupported dataset")

    # ===== model loading =====
    if args.model == "none":
        tokenizer, model = load_model("none", "Qwen/Qwen3-0.6B")
    elif args.model == "checkpoint":
        tokenizer, model = load_model("checkpoint", "Qwen/Qwen3-0.6B", "checkpoint_path")
    else:
        raise ValueError("Unsupported model")
    
    # ===== output setup =====
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = build_output_filename(
        dataset_name=args.data,
        mode=MODE,
        train_mode=args.model,
        constraint_method=args.constrained,
        )
    output_path = os.path.join(output_dir, output_filename)

    # ===== inference logic =====
    if args.constrained == "none":
        run_baseline("none", model, tokenizer, dataset, output_path)
    elif args.constrained == "xgrammar":
        run_baseline("xgrammar", model, tokenizer, dataset, output_path)
    elif args.constrained == "lmfe":
        run_baseline("lmfe", model, tokenizer, dataset, output_path)
    elif args.constrained == "outlines":
        run_baseline("outlines", model, tokenizer, dataset, output_path)
    else:
        raise ValueError("Unsupported constraint")

    print("Finished:", args.constrained)


if __name__ == "__main__":
    main()