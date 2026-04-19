"""
Usage:
===============================================================================
thinking/non-thinking

    for shot in few zero; do
        a="output/spider/eval_spider_non-thinking_${shot}_none.json"
        b="output/spider/eval_spider_thinking_${shot}_none.json"
        python compare_spider.py --a "$a" --b "$b" --limit 7 >> compare_report_t_nt.txt
    done
===============================================================================  
few/zero

    for mode in thinking non-thinking; do
        a="output/spider/eval_spider_${mode}_zero_none.json"
        b="output/spider/eval_spider_${mode}_few_none.json"
        python compare_spider.py --a "$a" --b "$b" --limit 7 >> compare_report_few_zero.txt
    done
===============================================================================
none/xgrammar

    for mode in thinking non-thinking; do
        for shot in few zero; do
              a="output/spider/eval_spider_${mode}_${shot}_none.json"
              b="output/spider/eval_spider_${mode}_${shot}_xgrammar.json"
              python compare_spider.py --a "$a" --b "$b" --limit 7 >> compare_report_none_xg.txt
        done
    done
===============================================================================
none/outlines

    for mode in thinking non-thinking; do
        for shot in few zero; do
            a="output/spider/eval_spider_${mode}_${shot}_none.json"
            b="output/spider/eval_spider_${mode}_${shot}_outlines.json"
            python compare_spider.py --a "$a" --b "$b" --limit 7 >> compare_report_none_out.txt
        done
    done
===============================================================================
"""

import json
import os
import argparse

def load(path):
    data = json.load(open(path, encoding='utf-8'))
    return {r['instance_id']: r for r in data}

def sql_match(r):
    return r.get('execution_match', False)

def is_exec(r):
    return r.get('predicted_answer') is not None

def main(path_a, path_b, limit):
    data_a  = load(path_a)
    data_b  = load(path_b)
    shared  = sorted(set(data_a) & set(data_b))
    n       = len(shared)
    label_a = os.path.basename(path_a)
    label_b = os.path.basename(path_b)

    print(f"{'Shared':>8}: {n}   Only-A: {len(set(data_a)-set(data_b))}   Only-B: {len(set(data_b)-set(data_a))}")
    print()

    match_a = sum(sql_match(data_a[i]) for i in shared)
    match_b = sum(sql_match(data_b[i]) for i in shared)
    print(f"{'File':<50} {'Match':>6} {'Rate':>7}")
    print("-" * 65)
    print(f"{label_a:<50} {match_a:>6} {match_a/n*100:>6.1f}%")
    print(f"{label_b:<50} {match_b:>6} {match_b/n*100:>6.1f}%")
    print()

    a_only   = [i for i in shared if     sql_match(data_a[i]) and not sql_match(data_b[i])]
    b_only   = [i for i in shared if not sql_match(data_a[i]) and     sql_match(data_b[i])]
    both_ok  = [i for i in shared if     sql_match(data_a[i]) and     sql_match(data_b[i])]
    both_bad = [i for i in shared if not sql_match(data_a[i]) and not sql_match(data_b[i])]

    print(f"{'Both correct':<20} {len(both_ok):>4}  ({len(both_ok)/n*100:.1f}%)")
    print(f"{'Both wrong':<20} {len(both_bad):>4}  ({len(both_bad)/n*100:.1f}%)")
    print(f"{'Only A correct':<20} {len(a_only):>4}  ({len(a_only)/n*100:.1f}%)")
    print(f"{'Only B correct':<20} {len(b_only):>4}  ({len(b_only)/n*100:.1f}%)")
    
    exec_a = sum(is_exec(data_a[i]) for i in shared)
    exec_b = sum(is_exec(data_b[i]) for i in shared) 
    print(f"\n{'Executable comparison':<50}")
    print("-" * 65)
    print(f"{label_a:<50} {exec_a:>6} {exec_a/n*100:>6.1f}%")
    print(f"{label_b:<50} {exec_b:>6} {exec_b/n*100:>6.1f}%")

    def show(ids, title):
        if not ids:
            return
        print(f"\n{'=' * 68}")
        print(f"  {title}  (showing up to {limit})")
        print(f"{'=' * 68}")
        for iid in ids[:limit]:
            ra, rb = data_a[iid], data_b[iid]
            print(f"  id       : {iid}")
            print(f"  question : {ra['question']}")
            print(f"  gold     : {ra['gold_sql']}")
            print(f"  A pred   : {ra['predicted_sql']}")
            print(f"  B pred   : {rb['predicted_sql']}")
            print()

    show(a_only, f"Only A correct  —  {label_a}")
    show(b_only, f"Only B correct  —  {label_b}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', required=True)
    parser.add_argument('--b', required=True)
    parser.add_argument('--limit', type=int, default=5)
    args = parser.parse_args()
    main(args.a, args.b, args.limit)