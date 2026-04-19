import sqlite3
import json
import re
import os

def fix_sql(sql, record):
    tbl  = 'table_' + record['db'].replace('-', '_')
    sql  = sql.replace('FROM table', f'FROM [{tbl}]')
    cols = [c[1] for c in record['schema']['column_names_original']]
    for i, col in sorted(enumerate(cols), key=lambda x: len(x[1]), reverse=True):
        sql = sql.replace(col, f'col{i}')
    sql = re.sub(r"('(?:[^']|\\')*')", r"\1 COLLATE NOCASE", sql)
    return sql

def evaluate(conn, input_path, output_path):
    data        = json.load(open(input_path, encoding='utf-8'))
    results     = []
    exec_count  = 0
    match_count = 0
    gold_errors = 0

    for r in data:
        gold_sql = fix_sql(r['gold_sql'],      r)
        pred_sql = fix_sql(r['predicted_sql'], r)

        try:
            gold_answer = [list(row) for row in conn.execute(gold_sql).fetchall()]
        except:
            gold_answer  = None
            gold_errors += 1

        try:
            pred_answer = [list(row) for row in conn.execute(pred_sql).fetchall()]
            exec_count += 1
        except:
            pred_answer = None

        if gold_answer is not None and pred_answer is not None:
            match = sorted(str(x) for x in gold_answer) == sorted(str(x) for x in pred_answer)
        else:
            match = False

        if match:
            match_count += 1

        record = dict(r)
        record['gold_answer'] = gold_answer
        record['predicted_answer'] = pred_answer
        record['execution_match'] = match
        results.append(record)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    total = len(data)
    ea = match_count / total * 100
    er = exec_count / total * 100
    return total, exec_count, er, match_count, ea, gold_errors

def main():
    conn = sqlite3.connect('data/data/train.db')
    os.makedirs('output', exist_ok=True)

    print(f"{'File':<50} {'Total':>5} {'Match':>5} {'EA':>7} {'Gold Err':>8} {'Exec':>5} {'ER':>6}")
    print("-" * 95)

    for fname in sorted(os.listdir('results/wiki')):
        if fname.endswith('.json'):
            input_path  = os.path.join('results/wiki', fname)
            output_path = os.path.join('output/wiki', 'eval_' + fname.replace('result_', ''))
            total, exec_n, er, match, ea, gold_err = evaluate(conn, input_path, output_path)
            print(f"{fname:<50} {total:>5} {match:>5} {ea:>6.1f}% {gold_err:>8} {exec_n:>5} {er:>5.1f}%")

    conn.close()

if __name__ == '__main__':
    main()
