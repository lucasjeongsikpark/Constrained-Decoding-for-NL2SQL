import sqlite3
import json
import os

spider_data = json.load(open('data/raw/spider_test.json', encoding='utf-8'))
q_to_db = {d['input']['question']: d['input']['db_id'] for d in spider_data}

DB_DIR = 'data/spider_data/test_database'

def get_conn(db_id):
    db_path = os.path.join(DB_DIR, db_id, f'{db_id}.sqlite')
    if not os.path.exists(db_path):
        return None
    return sqlite3.connect(db_path)

def evaluate(input_path, output_path):
    data = json.load(open(input_path, encoding='utf-8'))
    results = []
    exec_count = 0
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
        except Exception:
            gold_answer = None
            gold_errors += 1

        try:
            pred_answer = [list(row) for row in conn.execute(r['predicted_sql']).fetchall()]
            exec_count += 1
        except Exception:
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

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    total = len(data)
    effective_total = total - not_found if total > not_found else 1
    ea = match_count / effective_total * 100
    er = exec_count / effective_total * 100
    return total, exec_count, er, match_count, ea, gold_errors, not_found


def main():
    os.makedirs('output', exist_ok=True)

    print(f"{'File':<52} {'Total':>5} {'Match':>5} {'EA':>7} {'GErr':>5} {'Exec':>5} {'ER':>6} {'NotF':>5}")
    print("-" * 100)

    for fname in sorted(os.listdir('results/spider')):
        if 'spider' in fname and fname.endswith('.json'):
            input_path = os.path.join('results/spider', fname)
            output_path = os.path.join('output/spider', 'eval_' + fname.replace('result_', ''))
            total, exec_n, er, match, ea, gold_err, not_found = evaluate(input_path, output_path)
            print(f"{fname:<52} {total:>5} {match:>5} {ea:>6.1f}% {gold_err:>5} {exec_n:>5} {er:>5.1f}% {not_found:>5}")

if __name__ == '__main__':
    main()