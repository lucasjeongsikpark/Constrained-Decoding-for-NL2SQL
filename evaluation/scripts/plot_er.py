import matplotlib.pyplot as plt
import numpy as np
import os
import json

INPUT_DIR = 'output/spider'
OUTPUT_DIR = 'output'

METHODS = ['none', 'outlines', 'xgrammar']
MODES   = ['non-thinking', 'thinking']
SHOTS   = ['zero', 'few']


def load_data():
    data = {}

    for fname in os.listdir(INPUT_DIR):
        if not fname.endswith('.json'):
            continue

        path = os.path.join(INPUT_DIR, fname)
        records = json.load(open(path, encoding='utf-8'))
        total = len(records)
        if total == 0:
            continue

        # parse filename
        name = fname.replace('eval_spider_', '').replace('.json', '')
        mode, shot, method = name.split('_')

        exec_n = sum(1 for r in records if r.get('predicted_answer') is not None)
        match  = sum(1 for r in records if r.get('execution_match', False))

        er = exec_n / total * 100
        ea = match  / total * 100

        data[(mode, shot, method)] = (er, ea)

    return data


def get_vals(data, mode, shot):
    return [data.get((mode, shot, m), (0, 0))[0] for m in METHODS]


def get_ea_vals(data, mode, shot):
    return [data.get((mode, shot, m), (0, 0))[1] for m in METHODS]


def annotate(ax, bars):
    for b in bars:
        h = b.get_height()
        if h > 0:
            ax.annotate(f'{h:.0f}%',
                        (b.get_x() + b.get_width()/2, h),
                        textcoords="offset points",
                        xytext=(0, 4),
                        ha='center',
                        fontsize=9)


# ============================================================
# Figure 1: ER comparison
# ============================================================
def plot_er(data):
    x = np.arange(len(METHODS))
    w = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (mode, shot) in enumerate([
        ('non-thinking', 'zero'),
        ('non-thinking', 'few'),
        ('thinking', 'zero'),
        ('thinking', 'few'),
    ]):
        vals = get_vals(data, mode, shot)
        offset = (i - 1.5) * w
        bars = ax.bar(x + offset, vals, w, label=f'{mode}-{shot}')
        annotate(ax, bars)

    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in METHODS])
    ax.set_ylabel('Execution Rate (%)')
    ax.set_title('Execution Rate across Settings')
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'spider_er_comparison.png')
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# Figure 2: EA vs ER
# ============================================================
def plot_ea_vs_er(data):
    labels = []
    er_vals = []
    ea_vals = []

    for mode in MODES:
        for shot in SHOTS:
            for method in METHODS:
                er, ea = data.get((mode, shot, method), (0, 0))
                labels.append(f'{mode[:2]}-{shot[:1]}-{method[:1]}')
                er_vals.append(er)
                ea_vals.append(ea)

    x = np.arange(len(labels))
    w = 0.4

    fig, ax = plt.subplots(figsize=(12, 5))

    b1 = ax.bar(x - w/2, ea_vals, w, label='Execution Accuracy (EA)')
    b2 = ax.bar(x + w/2, er_vals, w, label='Execution Rate (ER)')

    annotate(ax, b1)
    annotate(ax, b2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30)
    ax.set_ylabel('Rate (%)')
    ax.set_title('Execution Accuracy vs Execution Rate (Spider)')
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'spider_ea_vs_er.png')
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = load_data()

    plot_er(data)
    plot_ea_vs_er(data)


if __name__ == '__main__':
    main()