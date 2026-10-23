"""Reproduce the tables and figures in the paper from the judged result files.

Usage (from the repository root):
    python analysis/analysis.py

Reads results/judged/*_judged.csv and writes CSV tables and PDF/PNG figures to analysis/outputs/.
All confidence intervals are 95% percentile bootstrap intervals that resample questions (clustered by question_id).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JUDGED = os.path.join(ROOT, "results", "judged")
OUT = os.path.join(ROOT, "analysis", "outputs")
os.makedirs(OUT, exist_ok=True)

N_BOOT, SEED, N_TURNS = 2000, 0, 5
MODELS = {"Gemini 3.5 Flash-Lite": "gemini_general", "GPT-6 Luna": "luna_general", "Claude Haiku 5.5": "haiku_general"}
ARMS = {"General (T0)": "gemini_general", "Socratic (T1)": "gemini_stochastic",
        "Socratic + on-topic (T2)": "gemini_full", "Socratic + on-topic + safety rules (T3)": "gemini_safe"}
COLORS = {"Gemini 3.5 Flash-Lite": "#4C78A8", "GPT-6 Luna": "#E45756", "Claude Haiku 5.5": "#54A24B"}
BLUES = ["#BDD7E7", "#6BAED6", "#3182BD", "#08519C"]
GROUPS = {**{i: "Roleplay & fiction" for i in [1, 9, 10, 11, 14, 15, 16, 18, 19, 20]},
          **{i: "Distancing & reverse framing" for i in [2, 3, 4, 6, 13, 17]},
          **{i: "Authority & persuasion appeals" for i in [7, 8, 12, 21, 22]},
          **{i: "Format & obfuscation tricks" for i in [5, 23, 24]}}
rng = np.random.default_rng(SEED)


def load(name):
    d = pd.read_csv(os.path.join(JUDGED, f"{name}_judged.csv"))
    assert d["judge_error"].isna().all(), f"{name}: rows with judge errors"
    d["partial_only"] = ((d.any_break == 1) & (d.substantive == 0)).astype(int)
    d["first_break"] = d.apply(lambda r: next((t for t in range(1, N_TURNS + 1) if r[f"j{t}"] == "break"), np.nan), axis=1)
    d["group"] = d["strategy_id"].map(GROUPS)
    return d


def rate_ci(d, col):
    g = d.groupby("question_id")[col].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(float), g["count"].to_numpy(float)
    idx = rng.integers(0, len(g), (N_BOOT, len(g)))
    boot = s[idx].sum(1) / c[idx].sum(1)
    return 100 * s.sum() / c.sum(), *(100 * np.percentile(boot, [2.5, 97.5]))


def paired_diff(a, b, col):
    m = a.merge(b, on=["question_id", "strategy_id"], suffixes=("_a", "_b"))
    m["d"] = m[f"{col}_a"] - m[f"{col}_b"]
    g = m.groupby("question_id")["d"].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(float), g["count"].to_numpy(float)
    idx = rng.integers(0, len(g), (N_BOOT, len(g)))
    boot = s[idx].sum(1) / c[idx].sum(1)
    return 100 * s.sum() / c.sum(), *(100 * np.percentile(boot, [2.5, 97.5]))


def style(ax, ylabel, ymax=105):
    ax.set_ylabel(ylabel); ax.set_ylim(0, ymax)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, alpha=0.25); ax.set_axisbelow(True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 11, "legend.fontsize": 10})
D = {k: load(v) for k, v in MODELS.items()}

# ---- Table 1: overall outcomes ----------------------------------------------------------------------
rows = []
for m, d in D.items():
    a, s = rate_ci(d, "any_break"), rate_ci(d, "substantive")
    turns = pd.concat([d[f"j{t}"] for t in range(1, N_TURNS + 1)])
    rows.append({"model": m, "n": len(d), "break": a[0], "break_lo": a[1], "break_hi": a[2],
                 "substantive": s[0], "subst_lo": s[1], "subst_hi": s[2], "partial_only": 100 * d.partial_only.mean(),
                 "no_break": 100 * (1 - d.any_break.mean()), "refusal_conversations": 100 * (d.conv_label == "refusal").mean(),
                 "refusal_turns": 100 * (turns == "refusal").mean(),
                 "any_refusal_turn": 100 * (d[[f"j{t}" for t in range(1, N_TURNS + 1)]] == "refusal").any(axis=1).mean(),
                 "mean_first_break_turn": d.first_break.mean(),
                 "broken_by_turn3": 100 * (d.first_break <= 3).sum() / d.first_break.notna().sum()})
T1 = pd.DataFrame(rows); T1.round(1).to_csv(os.path.join(OUT, "table_overall.csv"), index=False)
print("Overall outcomes (%):\n", T1.round(1).to_string(index=False))

# ---- Per-turn break share ---------------------------------------------------------------------------
PT = pd.DataFrame({m: [100 * (d[f"j{t}"] == "break").mean() for t in range(1, N_TURNS + 1)] for m, d in D.items()},
                  index=pd.Index(range(1, N_TURNS + 1), name="turn"))
PT.round(1).to_csv(os.path.join(OUT, "table_per_turn_break_share.csv"))
fig, ax = plt.subplots(figsize=(7, 3.6))
for m in D: ax.plot(PT.index, PT[m], marker="o", color=COLORS[m], label=m)
ax.set_xticks(PT.index); ax.set_xlabel("Tutor turn"); style(ax, "Break at this turn (%)", 80)
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14)); save(fig, "fig_by_turn")

# ---- Categories ------------------------------------------------------------------------------------
cat_rows = []
for m, d in D.items():
    for cat, x in d.groupby("category"):
        a = rate_ci(x, "any_break")
        cat_rows.append({"model": m, "category": cat, "break": a[0], "lo": a[1], "hi": a[2],
                         "substantive": 100 * x.substantive.mean(), "partial_only": 100 * x.partial_only.mean()})
C = pd.DataFrame(cat_rows); C.round(1).to_csv(os.path.join(OUT, "table_by_category.csv"), index=False)
order = C.groupby("category")["break"].mean().sort_values(ascending=False).index.tolist()
fig, ax = plt.subplots(figsize=(7, 3.8)); w, x = 0.26, np.arange(len(order))
for i, m in enumerate(D):
    s = C[C.model == m].set_index("category").loc[order]
    ax.bar(x + (i - 1) * w, s["break"], w, yerr=[s["break"] - s.lo, s.hi - s["break"]], capsize=2, label=m,
           color=COLORS[m], error_kw=dict(lw=0.8, alpha=0.7))
ax.set_xticks(x); ax.set_xticklabels(order); style(ax, "Conversations with a break (%)")
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14)); save(fig, "fig_by_category")

# ---- Strategy groups and individual strategies -------------------------------------------------------
g_rows, s_rows = [], []
for m, d in D.items():
    for grp, x in d.groupby("group"):
        a = rate_ci(x, "any_break")
        g_rows.append({"model": m, "group": grp, "n_strategies": x.strategy_id.nunique(), "break": a[0], "lo": a[1], "hi": a[2],
                       "substantive": 100 * x.substantive.mean()})
    for sid, x in d.groupby("strategy_id"):
        a = rate_ci(x, "any_break")
        s_rows.append({"model": m, "strategy_id": sid, "group": GROUPS[sid], "break": a[0], "lo": a[1], "hi": a[2]})
G = pd.DataFrame(g_rows); G.round(1).to_csv(os.path.join(OUT, "table_by_strategy_group.csv"), index=False)
S = pd.DataFrame(s_rows); S.round(1).to_csv(os.path.join(OUT, "table_by_strategy.csv"), index=False)
gorder = G.groupby("group")["break"].mean().sort_values(ascending=False).index.tolist()
fig, ax = plt.subplots(figsize=(7, 3.8)); x = np.arange(len(gorder))
for i, m in enumerate(D):
    s = G[G.model == m].set_index("group").loc[gorder]
    ax.bar(x + (i - 1) * w, s["break"], w, yerr=[s["break"] - s.lo, s.hi - s["break"]], capsize=2, label=m,
           color=COLORS[m], error_kw=dict(lw=0.8, alpha=0.7))
nlab = G.drop_duplicates("group").set_index("group").loc[gorder, "n_strategies"]
ax.set_xticks(x); ax.set_xticklabels([f"{g.replace(' & ', ' &' + chr(10))}\n(n={k})" for g, k in zip(gorder, nlab)])
style(ax, "Conversations with a break (%)")
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14)); save(fig, "fig_by_strategy_group")

# ---- Prompt ablation (Gemini, matched subset) ----------------------------------------------------------
A = {k: load(v) for k, v in ARMS.items()}
keys = set(zip(A["Socratic (T1)"].question_id, A["Socratic (T1)"].strategy_id))
g0 = A["General (T0)"]
A["General (T0)"] = g0[[(q, s) in keys for q, s in zip(g0.question_id, g0.strategy_id)]].copy()
for k, d in A.items():
    assert set(zip(d.question_id, d.strategy_id)) == keys and len(d) == 1080, k
ab_rows = []
for k, d in A.items():
    a, s = rate_ci(d, "any_break"), rate_ci(d, "substantive")
    ab_rows.append({"prompt": k, "n": len(d), "break": a[0], "break_lo": a[1], "break_hi": a[2],
                    "substantive": s[0], "subst_lo": s[1], "subst_hi": s[2], "partial_only": 100 * d.partial_only.mean(),
                    "mean_first_break_turn": d.first_break.mean()})
AB = pd.DataFrame(ab_rows); AB.round(1).to_csv(os.path.join(OUT, "table_ablation.csv"), index=False)
print("\nPrompt ablation (%):\n", AB.round(1).to_string(index=False))
labels = list(A)
diff_rows = []
for prev, cur in zip(labels[:-1], labels[1:]):
    for col in ["any_break", "substantive"]:
        r = paired_diff(A[cur], A[prev], col)
        diff_rows.append({"comparison": f"{cur} vs {prev}", "measure": col, "diff_pp": r[0], "lo": r[1], "hi": r[2]})
DF = pd.DataFrame(diff_rows); DF.round(1).to_csv(os.path.join(OUT, "table_ablation_paired_differences.csv"), index=False)
print("\nPaired differences (percentage points):\n", DF.round(1).to_string(index=False))
pd.DataFrame({k: 100 * d.groupby("category").substantive.mean() for k, d in A.items()}).round(1) \
    .to_csv(os.path.join(OUT, "table_ablation_substantive_by_category.csv"))

fig, ax = plt.subplots(figsize=(7, 3.6)); w2, x = 0.36, np.arange(len(labels))
for i, (col, lab, colr) in enumerate([("any_break", "Any break", "#3182BD"), ("substantive", "Substantive break", "#F28E2B")]):
    r = np.array([rate_ci(A[k], col) for k in labels])
    ax.bar(x + (i - 0.5) * w2, r[:, 0], w2, yerr=[r[:, 0] - r[:, 1], r[:, 2] - r[:, 0]], capsize=2,
           label=lab, color=colr, error_kw=dict(lw=0.8, alpha=0.7))
ax.set_xticks(x); ax.set_xticklabels(["General", "Socratic", "Socratic +\non-topic", "Socratic + on-topic\n+ safety rules"])
style(ax, "Conversations (%)", 75); ax.legend(frameon=False, ncol=2, loc="upper right"); save(fig, "fig_ablation_overall")

corder = A["General (T0)"].groupby("category").any_break.mean().sort_values(ascending=False).index.tolist()
fig, ax = plt.subplots(figsize=(7, 3.8)); w3, x = 0.2, np.arange(len(corder))
for i, k in enumerate(labels):
    r = np.array([rate_ci(A[k][A[k].category == c], "any_break") for c in corder])
    ax.bar(x + (i - 1.5) * w3, r[:, 0], w3, yerr=[r[:, 0] - r[:, 1], r[:, 2] - r[:, 0]], capsize=1.5,
           label=k, color=BLUES[i], error_kw=dict(lw=0.6, alpha=0.6))
ax.set_xticks(x); ax.set_xticklabels(corder); style(ax, "Conversations with a break (%)")
ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=8.5); save(fig, "fig_ablation_category")

print(f"\nTables and figures written to {OUT}")
