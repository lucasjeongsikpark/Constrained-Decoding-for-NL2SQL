import json
import os
import re
import sqlite3
from collections import Counter
import matplotlib.pyplot as plt

ROOT = "evaluation/output"


def fix_sql(sql, r):
    tbl = "table_" + r["db"].replace("-", "_")
    sql = sql.replace("FROM table", f"FROM [{tbl}]")

    cols = [c[1] for c in r["schema"]["column_names_original"]]
    cols_sorted = sorted(enumerate(cols), key=lambda x: len(x[1]), reverse=True)

    for i, c in cols_sorted:
        sql = sql.replace(c, f"col{i}")

    sql = re.sub(r"('(?:[^']|\\')*')", r"\1 COLLATE NOCASE", sql)
    return sql


def make_dummy_db(record):
    conn = sqlite3.connect(":memory:")
    tbl = "table_" + record["db"].replace("-", "_")
    cols = [f"col{i} TEXT" for i, _ in enumerate(record["schema"]["column_names_original"])]
    conn.execute(f"CREATE TABLE [{tbl}] ({', '.join(cols)})")
    return conn


def check_valid_sql(record):
    raw_sql = record["predicted_sql"]
    fixed_sql = fix_sql(raw_sql, record)

    conn = make_dummy_db(record)
    try:
        conn.execute("EXPLAIN QUERY PLAN " + fixed_sql)
        return True, None, fixed_sql
    except Exception as e:
        return False, str(e), fixed_sql
    finally:
        conn.close()


def classify_error(raw_sql, error_msg):
    lower = raw_sql.lower()
    err = (error_msg or "").lower()

    if not raw_sql.strip():
        return "empty output"

    if re.search(r"limit\s+\d{20,}", lower):
        return "runaway long LIMIT integer"

    if len(raw_sql) > 250:
        return "runaway repeated conditions / overgeneration"

    if raw_sql.count("'") % 2 == 1 or raw_sql.count('"') % 2 == 1:
        return "unbalanced quotes"

    if raw_sql.count("(") != raw_sql.count(")"):
        return "unbalanced parentheses"

    if "incomplete input" in err:
        return "incomplete input"

    if "syntax error" in err:
        return "syntax error"

    if "unrecognized token" in err:
        return "unrecognized token"

    if "overflow" in err:
        return "numeric overflow"

    if "no such column" in err:
        return "bad column reference"

    return "other parse failure"


def parse_setting(filename):
    m = re.match(r"eval_wikisql_(.*)_(none|outlines|xgrammar|LMFE)\.json", filename)
    thinking = m.group(1)
    method = m.group(2)
    return thinking, method


def main():
    os.makedirs("evaluation/output/analysis", exist_ok=True)

    summary_rows = []
    error_rows = []
    case_rows = []

    for fname in sorted(os.listdir(ROOT)):
        if not fname.endswith(".json"):
            continue

        path = os.path.join(ROOT, fname)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        thinking, method = parse_setting(fname)

        valid_count = 0
        invalid_examples = []

        for r in data:
            is_valid, err, fixed_sql = check_valid_sql(r)

            if is_valid:
                valid_count += 1
            else:
                category = classify_error(r["predicted_sql"], err)
                invalid_examples.append({
                    "file": fname,
                    "thinking": thinking,
                    "method": method,
                    "instance_id": r["instance_id"],
                    "question": r["question"],
                    "gold_sql": r["gold_sql"],
                    "predicted_sql": r["predicted_sql"],
                    "fixed_sql": fixed_sql,
                    "execution_match": r.get("execution_match", False),
                    "error_message": err,
                    "error_category": category,
                })

        total = len(data)
        invalid_count = total - valid_count
        valid_rate = 100 * valid_count / total

        summary_rows.append({
            "file": fname,
            "thinking": thinking,
            "method": method,
            "total": total,
            "valid": valid_count,
            "invalid": invalid_count,
            "valid_sql_rate": valid_rate,
        })

        counter = Counter(x["error_category"] for x in invalid_examples)
        for category, count in counter.items():
            error_rows.append({
                "file": fname,
                "thinking": thinking,
                "method": method,
                "error_category": category,
                "count": count,
            })

        case_rows.extend(invalid_examples)

    import csv

    with open("evaluation/output/analysis/valid_sql_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    if error_rows:
        with open("evaluation/output/analysis/error_breakdown.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=error_rows[0].keys())
            writer.writeheader()
            writer.writerows(error_rows)

    if case_rows:
        with open("evaluation/output/analysis/invalid_case_studies.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=case_rows[0].keys())
            writer.writeheader()
            writer.writerows(case_rows)

    labels = [f"{r['thinking']} + {r['method']}" for r in summary_rows]
    values = [r["valid_sql_rate"] for r in summary_rows]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.ylabel("Valid SQL Rate (%)")
    plt.title("Valid SQL Rate Across Decoding Methods")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig("evaluation/output/analysis/valid_sql_rate.png", dpi=200)
    plt.close()

    invalid_values = [r["invalid"] for r in summary_rows]
    plt.figure(figsize=(10, 5))
    plt.bar(labels, invalid_values)
    plt.ylabel("Invalid SQL Count")
    plt.title("Invalid SQL Outputs Across Decoding Methods")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig("evaluation/output/analysis/invalid_sql_count.png", dpi=200)
    plt.close()

    print("Saved:")
    print(" - evaluation/output/analysis/valid_sql_summary.csv")
    print(" - evaluation/output/analysis/error_breakdown.csv")
    print(" - evaluation/output/analysis/invalid_case_studies.csv")
    print(" - evaluation/output/analysis/valid_sql_rate.png")
    print(" - evaluation/output/analysis/invalid_sql_count.png")


if __name__ == "__main__":
    main()
