import csv
import os

path = "evaluation/output/analysis/invalid_case_studies.csv"
out_path = "evaluation/output/analysis/case_study_preview.txt"

with open(path, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

targets = [
    ("thinking_zero", "xgrammar", "empty output"),
    ("thinking_zero", "xgrammar", "runaway repeated conditions / overgeneration"),
    ("non-thinking_zero", "xgrammar", "runaway repeated conditions / overgeneration"),
    ("thinking_zero", "none", "bad column reference"),
]

os.makedirs(os.path.dirname(out_path), exist_ok=True)

lines = []

for thinking, method, category in targets:
    lines.append("\n" + "#" * 100)
    lines.append(f"{thinking} | {method} | {category}")
    matches = [
        r for r in rows
        if r["thinking"] == thinking and r["method"] == method and r["error_category"] == category
    ]
    for r in matches[:3]:
        lines.append("-" * 80)
        lines.append(f"instance_id: {r['instance_id']}")
        lines.append(f"question: {r['question']}")
        lines.append(f"predicted_sql: {r['predicted_sql']}")
        lines.append(f"error_message: {r['error_message']}")

text = "\n".join(lines)

print(text)

with open(out_path, "w", encoding="utf-8") as f:
    f.write(text)

print(f"\nSaved to: {out_path}")
