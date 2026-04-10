"""
Usage:
    for f in output/spider/eval_*.json; do
        echo "========== $f ==========" >> error_report.txt
        python3 errors_spider.py --input "$f" --valid_exec_gap >> error_report.txt
    done
"""

import json
import os
import re
import argparse
from collections import defaultdict

def norm(sql):
    return re.sub(r'\s+', ' ', sql.strip().lower())

def from_table(sql):
    m = re.search(r'from\s+(\S+)', sql, re.IGNORECASE)
    return m.group(1).lower() if m else ''

def select_cols(sql):
    m = re.search(r'select\s+(.*?)\s+from', sql, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip().lower() if m else ''

def where_clause(sql):
    m = re.search(r'where\s+(.*)', sql, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip().lower() if m else ''

def categorise(gold, pred):
    if not pred.strip():
        return 'EMPTY'
    if norm(gold) == norm(pred):
        return 'CORRECT'
    if len(pred) > 500:
        return 'OVERGENERATION'
    if pred.count('SELECT') > 3:
        return 'OVERCOMPLEX'
    if 'JOIN' in pred.upper() and 'JOIN' not in gold.upper():
        return 'EXTRA_JOIN'
    if 'JOIN' not in pred.upper() and 'JOIN' in gold.upper():
        return 'MISSING_JOIN'
    return 'OTHER'

def is_syntactically_valid(sql):
    """Return True if sqlparse can parse the SQL without errors."""
    try:
        import sqlparse
        flat = sql.strip().lower()
        if not flat or flat in ('', ';'):
            return False
        parsed = sqlparse.parse(sql.strip())
        return bool(parsed and parsed[0].tokens)
    except Exception:
        return False

def guess_crash_reason(pred_sql, gold_sql):
    pred = pred_sql.lower()
    if not pred.strip():
        return 'EMPTY'
    if len(pred_sql) > 500:
        return 'OVERGENERATION'
    if 'join' in pred and 'join' not in gold_sql.lower():
        return 'BAD_JOIN'
    if 'no such table' in pred:
        return 'WRONG_TABLE_REF'
    if 'no such column' in pred:
        return 'BAD_COLUMN_REF'
    if pred.count('select') > 3:
        return 'OVERCOMPLEX'
    return 'UNKNOWN'

def run_default_or_exec(data, exec_only):
    total     = len(data)
    buckets   = defaultdict(list)
    n_correct = 0
    for r in data:
        if exec_only and r.get('predicted_answer') is None:
            continue
        cat = categorise(r['gold_sql'], r['predicted_sql'])
        if cat == 'CORRECT':
            n_correct += 1
        else:
            buckets[cat].append(r)
    return total, n_correct, buckets

def run_valid_gap(data):
    """
    Find cases where predicted SQL is syntactically valid but crashed at runtime.
    Requires eval_*.json (has predicted_answer field from er.py).
    """
    total     = len(data)
    n_valid   = 0
    n_exec    = 0
    gap_cases = []

    for r in data:
        pred    = r.get('predicted_sql', '').strip()
        valid   = is_syntactically_valid(pred)
        crashed = r.get('predicted_answer') is None

        if valid:   n_valid += 1
        if not crashed: n_exec += 1

        if valid and crashed:
            reason = guess_crash_reason(pred, r['gold_sql'])
            r2 = dict(r)
            r2['_crash_reason'] = reason
            gap_cases.append(r2)

    return total, n_valid, n_exec, gap_cases

def print_default(fname, total, n_correct, buckets, exec_only, limit):
    n_errors = sum(len(v) for v in buckets.values())
    scope    = "(exec-only)" if exec_only else "(all)"
    print(f"{'File':<50} {'Total':>5} {'Correct':>7} {'Errors':>7}")
    print("-" * 75)
    print(f"{fname:<50} {total:>5} {n_correct:>7} {n_errors:>7}  {scope}")
    print()
    ORDER = ['EMPTY', 'WRONG_TABLE', 'WRONG_COL', 'WRONG_COND', 'OTHER']
    print(f"{'Category':<14} {'Count':>6} {'% of errors':>12}")
    print("-" * 35)
    for cat in ORDER:
        cnt = len(buckets[cat])
        pct = cnt / n_errors * 100 if n_errors else 0
        print(f"{cat:<14} {cnt:>6} {pct:>11.1f}%")
    for cat in ORDER:
        cases = buckets[cat]
        if not cases:
            continue
        print(f"\n{'=' * 65}")
        print(f"  {cat}  ({len(cases)} cases, showing up to {limit})")
        print(f"{'=' * 65}")
        for r in cases[:limit]:
            print(f"  id       : {r['instance_id']}")
            print(f"  question : {r['question']}")
            print(f"  gold     : {r['gold_sql']}")
            print(f"  pred     : {r['predicted_sql']}")
            print()

def print_valid_gap(fname, total, n_valid, n_exec, gap_cases, limit):
    n_gap = len(gap_cases)
    print(f"{'File':<50} {'Total':>5} {'Valid':>6} {'Exec':>5} {'Gap':>5}")
    print("-" * 75)
    print(f"{fname:<50} {total:>5} {n_valid:>6} {n_exec:>5} {n_gap:>5}")
    print()
    print(f"  Valid SQL rate   : {n_valid/total*100:.1f}%")
    print(f"  Execution rate   : {n_exec/total*100:.1f}%")
    print(f"  Gap (valid→exec) : {n_gap} cases  ({n_gap/total*100:.1f}% of total)")
    print()

    if not gap_cases:
        print("  No gap cases found.")
        return

    reason_counts = defaultdict(list)
    for r in gap_cases:
        reason_counts[r['_crash_reason']].append(r)

    REASONS = ['WRONG_TABLE_REF', 'BAD_COLUMN_REF', 'TYPE_MISMATCH', 'OVERGENERATION', 'UNKNOWN']
    print(f"{'Crash Reason':<20} {'Count':>6} {'% of gap':>10}")
    print("-" * 40)
    for reason in REASONS:
        cnt = len(reason_counts[reason])
        pct = cnt / n_gap * 100 if n_gap else 0
        print(f"{reason:<20} {cnt:>6} {pct:>9.1f}%")

    for reason in REASONS:
        cases = reason_counts[reason]
        if not cases:
            continue
        print(f"\n{'=' * 65}")
        print(f"  {reason}  ({len(cases)} cases, showing up to {limit})")
        print(f"{'=' * 65}")
        for r in cases[:limit]:
            print(f"  id       : {r['instance_id']}")
            print(f"  question : {r['question']}")
            print(f"  gold     : {r['gold_sql']}")
            print(f"  pred     : {r['predicted_sql']}")
            print()

def main(input_path, exec_only, valid_exec_gap, limit):
    data  = json.load(open(input_path, encoding='utf-8'))
    fname = os.path.basename(input_path)

    if valid_exec_gap:
        if 'predicted_answer' not in data[0]:
            print("ERROR: --valid_exec_gap requires eval_*.json files (run er.py first)")
            return
        try:
            import sqlparse
        except ImportError:
            print("ERROR: --valid_exec_gap requires sqlparse.  Run: pip install sqlparse")
            return
        total, n_valid, n_exec, gap_cases = run_valid_gap(data)
        print_valid_gap(fname, total, n_valid, n_exec, gap_cases, limit)
    else:
        total, n_correct, buckets = run_default_or_exec(data, exec_only)
        print_default(fname, total, n_correct, buckets, exec_only, limit)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--exec_only', action='store_true')
    parser.add_argument('--valid_exec_gap', action='store_true')
    parser.add_argument('--limit', type=int, default=5)
    args = parser.parse_args()
    main(args.input, args.exec_only, args.valid_exec_gap, args.limit)
