import json
import os
import statistics

def main():
    print(f"{'File':<50} {'Total':>5} {'SQL Match':>9} {'Rate':>7} {'Avg Lat(s)':>12} {'Avg Tok':>8}")
    print("-" * 95)

    for fname in sorted(os.listdir('results/wiki')):
        if fname.endswith('.json'):
            data  = json.load(open(os.path.join('results/wiki', fname), encoding='utf-8'))
            total = len(data)
            match = sum(1 for r in data if r['gold_sql'].strip() == r['predicted_sql'].strip())
            lats  = [r['latency_ms'] for r in data if 'latency_ms' in r]
            toks  = [r['num_tokens_generated'] for r in data if 'num_tokens_generated' in r]
            avg_lat = f"{statistics.mean(lats)/1000:>10.1f}" if lats else f"{'N/A':>10}"
            avg_tok = f"{statistics.mean(toks):>8.1f}"       if toks else f"{'N/A':>8}"
            print(f"{fname:<50} {total:>5} {match:>9} {match/total*100:>6.1f}% {avg_lat} {avg_tok}")

if __name__ == '__main__':
    main()
