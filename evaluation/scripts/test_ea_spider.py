import sqlite3
import json
import os

# map question -> db_id
spider_data = json.load(open('data/spider_data/spider_test.json', encoding='utf-8'))
q_to_db = {d['input']['question']: d['input']['db_id'] for d in spider_data}

DB_DIR = 'data/spider_data/test_database'
RESULTS_DIR = 'results_spider'
OUTPUT_DIR = 'output'

def get_conn(db_id):
    db_path = os.path.join(DB_DIR, db_id, f'{db_id}.sqlite')
    if not os.path.exists(db_path):
        return None
    return sqlite3.connect(db_path)

def evaluate(input_path, output_path):
    data = json.load(open(input_path, encoding='utf-8'))
    results = []
    match_count = 0
    gold_errors = 0
    not_found = 0

    for r in data:
        db_id = q_to_db.get(r['question'])
        if db_id is None:
            not_found += 1
            record = dict(r)
            record['db_id'] = None
            record['gold_answer'] = None
            record['predicted_answer'] = None
            record['execution_match'] = False
            results.append(record)
            continue

        conn = get_conn(db_id)
        if conn is None:
            not_found += 1
            record = dict(r)
            record['db_id'] = db_id
            record['gold_answer'] = None
            record['predicted_answer'] = None
            record['execution_match'] = False
            results.append(record)
            continue

        try:
            gold_answer = [list(row) for row in conn.execute(r['gold_sql']).fetchall()]
        except Exception as e:
            gold_answer = None
            gold_errors += 1

        try:
            pred_answer = [list(row) for row in conn.execute(r['predicted_sql']).fetchall()]
        except:
            pred_answer = None

        if gold_answer is not None and pred_answer is not None:
            match = sorted(str(x) for x in gold_answer) == sorted(str(x) for x in pred_answer)
        else:
            match = False

        if match:
            match_count += 1

        record = dict(r)
        record['db_id'] = db_id
        record['gold_answer'] = gold_answer
        record['predicted_answer'] = pred_answer
        record['execution_match'] = match
        results.append(record)

        conn.close()

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    total = len(data)
    ea = match_count / total * 100 if total > 0 else 0
    return total, match_count, ea, gold_errors, not_found

# run spider evaluation
print(f"{'File':<52} {'Total':>5} {'Match':>5} {'EA':>7} {'GErr':>5} {'NotF':>5}")
print("-" * 85)

for fname in sorted(os.listdir(RESULTS_DIR)):
    if fname.endswith('.json'):
        input_path = os.path.join(RESULTS_DIR, fname)
        output_path = os.path.join(OUTPUT_DIR, 'eval_' + fname.replace('result_', ''))
        total, match, ea, gold_err, not_found = evaluate(input_path, output_path)
        print(f"{fname:<52} {total:>5} {match:>5} {ea:>6.1f}% {gold_err:>5} {not_found:>5}")