"""Stratified random sample of question-strategy pairs for the Urdu experiment.
Strata: category x safe_n (0-3). 8 pairs per stratum, <=2 pairs per question per stratum.
Substance use / safe_n=3 has only 5 pairs: take all 5, draw the missing 3 from Substance use / safe_n=2.
Input: data/english_safety_by_pair.csv (one row per English benchmark pair; safe_<model> = no break in the original English run, safe_n = number of the three tutors that were safe).
Usage: python sample_urdu_pairs.py [input.csv] [output.csv]
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np, pandas as pd

SEED = 2027
PER_STRATUM = 8
CAP_PER_Q = 2
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "data", "english_safety_by_pair.csv")
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "data", "urdu_pairs_192.csv")
w = pd.read_csv(src)
assert len(w) == 2016
rng = np.random.default_rng(SEED)

# target counts per stratum
target = {(c, s): PER_STRATUM for c in sorted(w.category.unique()) for s in range(4)}
short = {}
for (c, s), n in target.items():
    avail = ((w.category == c) & (w.safe_n == s)).sum()
    if avail < n:
        short[(c, s)] = n - avail
        target[(c, s)] = avail
for (c, s), k in short.items():          # top up from the next-lower safe_n level
    target[(c, s - 1)] += k

def draw(cell, n):
    order = cell.iloc[rng.permutation(len(cell))]
    picked, per_q = [], {}
    for idx, r in order.iterrows():
        if per_q.get(r.question_id, 0) < CAP_PER_Q:
            picked.append(idx); per_q[r.question_id] = per_q.get(r.question_id, 0) + 1
        if len(picked) == n:
            break
    if len(picked) < n:                    # cap infeasible: relax it for this cell only
        rest = [i for i in order.index if i not in picked]
        picked += rest[: n - len(picked)]
        print(f"  note: cap relaxed in {cell.category.iloc[0]} / safe_n={cell.safe_n.iloc[0]}")
    return picked

rows = []
for (c, s), n in target.items():
    cell = w[(w.category == c) & (w.safe_n == s)]
    idx = draw(cell, n)
    sub = w.loc[idx].copy()
    sub["stratum_size"] = len(cell)
    sub["n_drawn"] = n
    sub["weight"] = len(cell) / n          # inverse inclusion probability
    rows.append(sub)

S = pd.concat(rows).sort_values(["category", "question_id", "strategy_id"]).reset_index(drop=True)
S.insert(0, "pair_id", range(1, len(S) + 1))
keep = ["pair_id", "category", "question_id", "strategy_id", "safe_n",
        "safe_gemini", "safe_luna", "safe_haiku", "stratum_size", "n_drawn", "weight"]
S[keep].to_csv(out, index=False)

# ---- summary + representativeness check ----
print(f"pairs: {len(S)} | questions: {S.question_id.nunique()}/84 | strategies: {S.strategy_id.nunique()}/24")
print(pd.crosstab(S.category, S.safe_n, margins=True))
print("max pairs from one question (overall):", S.question_id.value_counts().max())
print("\nOriginal English break rate: full benchmark vs weighted sample vs unweighted sample")
for m in ["gemini", "luna", "haiku"]:
    full = 1 - w[f"safe_{m}"].astype(float).mean()
    wt = 1 - np.average(S[f"safe_{m}"].astype(float), weights=S.weight)
    un = 1 - S[f"safe_{m}"].astype(float).mean()
    print(f"  {m:6s}  full {full:6.1%}   weighted {wt:6.1%}   unweighted {un:6.1%}")
