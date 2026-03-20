import json
import os

print(f"{'File':<50} {'Total':>5} {'SQL Match':>9} {'Rate':>7}")
print("-" * 75)

for fname in sorted(os.listdir('results')):
    if fname.endswith('.json'):
        data = json.load(open(os.path.join('results', fname), encoding='utf-8'))
        total = len(data)
        match = sum(1 for r in data if r['gold_sql'].strip() == r['predicted_sql'].strip())
        print(f"{fname:<50} {total:>5} {match:>9} {match/total*100:>6.1f}%")