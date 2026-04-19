import json
import os
import statistics

def is_exec(r):
    return r.get('predicted_answer') is not None

def is_correct(r):
    return r.get('execution_match', False)

def main():
    print(f"{'File':<55} {'Total':>5} {'Exec':>6} {'ER':>7} {'Match':>6} {'EA':>7} {'Avg Lat(s)':>12} {'Avg Tok':>8}")
    print("-" * 110)

    for fname in sorted(os.listdir('output/spider')):
        if fname.endswith('.json'):
            path = os.path.join('output/spider', fname)
            data = json.load(open(path, encoding='utf-8'))
            total = len(data)
            exec_n = sum(1 for r in data if is_exec(r))
            match  = sum(1 for r in data if is_correct(r))
            er = exec_n / total * 100 if total else 0
            ea = match  / total * 100 if total else 0
            lats = [r['latency_ms'] for r in data if 'latency_ms' in r]
            toks = [r['num_tokens_generated'] for r in data if 'num_tokens_generated' in r]
            avg_lat = f"{statistics.mean(lats)/1000:>10.1f}" if lats else f"{'N/A':>10}"
            avg_tok = f"{statistics.mean(toks):>8.1f}"       if toks else f"{'N/A':>8}"
            print(f"{fname:<55} {total:>5} {exec_n:>6} {er:>6.1f}% {match:>6} {ea:>6.1f}% {avg_lat} {avg_tok}")

if __name__ == '__main__':
    main()