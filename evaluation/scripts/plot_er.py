"""
3 evaluation charts for the constrained decoding project
Figures saved to output/ as PNG files.
Usage: python plot_er.py
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import json

def load_from_json(dir):
    methods = ['None', 'Outlines', 'XGrammar', 'LMFE']
    tag_map = {'none': 'None', 'outlines': 'Outlines', 'xgrammar': 'XGrammar', 'lmfe': 'LMFE'}
    er_scores, sql_scores = {}, {}

    for fname in os.listdir(dir):
        if not fname.endswith('.json'):
            continue
        nl = fname.lower()
        mode   = 'non_thinking' if ('non-thinking' in nl or 'non_thinking' in nl) else \
                 'thinking'     if 'thinking' in nl else None
        method = next((label for tag, label in tag_map.items() if tag in nl), None)
        if not mode or not method:
            continue

        data  = json.load(open(os.path.join(dir, fname), encoding='utf-8'))
        total = len(data)
        if total == 0:
            continue

        er_n  = sum(1 for r in data if r.get('predicted_answer') is not None)
        sql_n = sum(1 for r in data
                    if r['gold_sql'].strip().lower() == r['predicted_sql'].strip().lower())
        er_scores [(mode, method)] = er_n  / total * 100
        sql_scores[(mode, method)] = sql_n / total * 100

    def extract(scores, mode):
        return [scores.get((mode, m), 0.0) for m in methods]

    return {
        'methods'          : methods,
        'non_thinking_er'  : extract(er_scores,  'non_thinking'),
        'thinking_er'      : extract(er_scores,  'thinking'),
        'non_thinking_sql' : extract(sql_scores, 'non_thinking'),
        'thinking_sql'     : extract(sql_scores, 'thinking'),
    }

def annotate(ax, bars, fontsize=10):
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.annotate(f'{h:.0f}%',
                        xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 4), textcoords='offset points',
                        ha='center', fontsize=fontsize)

# ============================================================
# Figure 1: ER by decoding method
# ============================================================
def fig1(data, output_dir):
    methods = data['methods']
    x, w    = np.arange(len(methods)), 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w/2, data['non_thinking_er'], w, label='Non-thinking', color='#4472C4')
    b2 = ax.bar(x + w/2, data['thinking_er'],     w, label='Thinking',     color='#ED7D31')
    ax.set_ylabel('Execution Rate (%)', fontsize=12)
    ax.set_title('Figure 1: Execution Rate by Decoding Method', fontsize=13, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylim(0, 100); ax.legend(fontsize=11); ax.grid(axis='y', alpha=0.3)
    annotate(ax, b1); annotate(ax, b2)
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig1_er_comparison.png')
    plt.savefig(path, dpi=200); plt.close(); print(f"Saved: {path}")

# ============================================================
# Figure 2: SQL Match vs ER
# ============================================================
def fig2(data, output_dir):
    methods  = data['methods']
    settings = [f'NT+{m}' for m in methods] + [f'T+{m}' for m in methods]
    sql_vals = data['non_thinking_sql'] + data['thinking_sql']
    er_vals  = data['non_thinking_er']  + data['thinking_er']
    x, w     = np.arange(len(settings)), 0.35
    fig, ax  = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w/2, sql_vals, w, label='SQL Exact Match',  color='#A5A5A5')
    b2 = ax.bar(x + w/2, er_vals,  w, label='Execution Rate',   color='#4472C4')
    ax.set_ylabel('Rate (%)', fontsize=12)
    ax.set_title('Figure 2: SQL Exact Match vs Execution Rate', fontsize=13, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(settings, fontsize=9, rotation=15)
    ax.set_ylim(0, 100); ax.legend(fontsize=11); ax.grid(axis='y', alpha=0.3)
    annotate(ax, b1, fontsize=9); annotate(ax, b2, fontsize=9)
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig2_sql_match_vs_er.png')
    plt.savefig(path, dpi=200); plt.close(); print(f"Saved: {path}")

# ============================================================
# Figure 3: ER breakdown (stacked)
# ============================================================
def fig3(data, output_dir):
    methods  = data['methods']
    settings = [f'NT+{m}' for m in methods] + [f'T+{m}' for m in methods]
    sql_vals = data['non_thinking_sql'] + data['thinking_sql']
    er_vals  = data['non_thinking_er']  + data['thinking_er']
    gap      = [er - sql for er, sql in zip(er_vals, sql_vals)]
    x        = np.arange(len(settings))
    fig, ax  = plt.subplots(figsize=(10, 5))
    ax.bar(x, sql_vals, 0.5, label='SQL Exact Match',          color='#4472C4')
    ax.bar(x, gap,      0.5, bottom=sql_vals,
           label='ER beyond SQL Match', color='#70AD47')
    ax.set_ylabel('Rate (%)', fontsize=12)
    ax.set_title('Figure 3: ER Breakdown — SQL Match + Additional Correct Executions',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(settings, fontsize=9, rotation=15)
    ax.set_ylim(0, 100); ax.legend(fontsize=11); ax.grid(axis='y', alpha=0.3)
    for i, total in enumerate(er_vals):
        if total > 0:
            ax.annotate(f'{total:.0f}%', xy=(i, total),
                        xytext=(0, 4), textcoords='offset points', ha='center', fontsize=10)
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig3_er_gap_analysis.png')
    plt.savefig(path, dpi=200); plt.close(); print(f"Saved: {path}")

def main():
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    data = load_from_json('output')

    fig1(data, output_dir)
    fig2(data, output_dir)
    fig3(data, output_dir)

if __name__ == '__main__':
    main()
