import json

none = json.load(open('output/eval_wikisql_thinking_zero_none.json', encoding='utf-8'))
outlines = json.load(open('output/eval_wikisql_thinking_zero_outlines.json', encoding='utf-8'))

none_map = {r['instance_id']: r for r in none}
outlines_map = {r['instance_id']: r for r in outlines}

print("=== Thinking: None CORRECT, Outlines WRONG ===\n")
count = 0
for iid in none_map:
    n = none_map[iid]
    o = outlines_map.get(iid)
    if o and n['execution_match'] and not o['execution_match']:
        count += 1
        print(f"[{count}] {iid}")
        print(f"  Q: {n['question']}")
        print(f"  Gold SQL:     {n['gold_sql']}")
        print(f"  None SQL:     {n['predicted_sql']}")
        print(f"  Outlines SQL: {o['predicted_sql']}")
        print(f"  Gold answer:     {n['gold_answer']}")
        print(f"  None answer:     {n['predicted_answer']}")
        print(f"  Outlines answer: {o['predicted_answer']}")
        print()

print(f"Total: {count} cases")