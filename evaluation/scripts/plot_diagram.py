"""
EA Evaluation Charts
Run: pip install matplotlib
Then: python generate_charts.py
Output: 3 PNG files in output/ folder
"""
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('output', exist_ok=True)

methods = ['None', 'Outlines', 'XGrammar', 'LMFE']
non_thinking_ea = [30.0, 28.0, 34.0, 26.0]
thinking_ea     = [44.0, 16.0, 0.0, 14.0]
non_thinking_sql = [16.0, 16.0, 18.0, 14.0]
thinking_sql     = [22.0, 4.0, 0.0, 4.0]

# ============================================================
# Figure 1: Grouped bar chart — EA by decoding method
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(methods))
w = 0.35

bars1 = ax.bar(x - w/2, non_thinking_ea, w, label='Non-thinking', color='#4472C4')
bars2 = ax.bar(x + w/2, thinking_ea, w, label='Thinking', color='#ED7D31')

ax.set_ylabel('Execution Accuracy (%)', fontsize=12)
ax.set_title('Figure 1: Execution Accuracy by Decoding Method', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=11)
ax.set_ylim(0, 55)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.0f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 4), textcoords="offset points", ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('output/fig1_ea_comparison.png', dpi=200)
plt.close()
print("Saved: output/fig1_ea_comparison.png")

# ============================================================
# Figure 2: SQL Match vs EA side-by-side
# ============================================================
settings = [
    'NT+None', 'NT+Outlines', 'NT+XGrammar', 'NT+LMFE',
    'T+None', 'T+Outlines', 'T+XGrammar', 'T+LMFE'
]
sql_match = non_thinking_sql + thinking_sql
ea_vals   = non_thinking_ea + thinking_ea

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(settings))
w = 0.35

bars1 = ax.bar(x - w/2, sql_match, w, label='SQL Exact Match', color='#A5A5A5')
bars2 = ax.bar(x + w/2, ea_vals, w, label='Execution Accuracy', color='#4472C4')

ax.set_ylabel('Rate (%)', fontsize=12)
ax.set_title('Figure 2: SQL Exact Match vs Execution Accuracy', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(settings, fontsize=9, rotation=15)
ax.set_ylim(0, 55)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.annotate(f'{h:.0f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 4), textcoords="offset points", ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('output/fig2_sql_match_vs_ea.png', dpi=200)
plt.close()
print("Saved: output/fig2_sql_match_vs_ea.png")

# ============================================================
# Figure 3: EA Gap analysis (stacked: SQL match + extra EA)
# ============================================================
gap = [ea - sql for ea, sql in zip(ea_vals, sql_match)]

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(settings))

ax.bar(x, sql_match, 0.5, label='SQL Exact Match', color='#4472C4')
ax.bar(x, gap, 0.5, bottom=sql_match, label='EA beyond SQL Match', color='#70AD47')

ax.set_ylabel('Rate (%)', fontsize=12)
ax.set_title('Figure 3: EA Breakdown — SQL Match + Additional Correct Executions', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(settings, fontsize=9, rotation=15)
ax.set_ylim(0, 55)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

for i in range(len(settings)):
    total = ea_vals[i]
    if total > 0:
        ax.annotate(f'{total:.0f}%', xy=(i, total), xytext=(0, 4),
                    textcoords="offset points", ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('output/fig3_ea_gap_analysis.png', dpi=200)
plt.close()
print("Saved: output/fig3_ea_gap_analysis.png")

print("\nAll charts generated in output/ folder.")