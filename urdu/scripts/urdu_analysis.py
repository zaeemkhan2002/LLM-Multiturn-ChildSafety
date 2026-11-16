"""
Urdu short paper: full analysis of the judged conversations.

Usage (defaults point at the repo layout, so from the repo root this is enough):
    python scripts/urdu_analysis.py [--judged-dir judged] [--pairs data/urdu_pairs_192.csv] \
                                    [--results-dir conversations] [--out results]

--judged-dir   folder with gemini_urdu_judged.csv, luna_urdu_judged.csv, haiku_urdu_judged.csv
--pairs        urdu_pairs_192.csv (gives safe_n and the stratum weights)
--results-dir  optional: folder with the raw results_*_urdu*.csv files, used only for the
               provider-block counts and the language-drift check
--out          output folder (tables as CSV + LaTeX, figures as PDF + PNG, summary.md)

Intervals: 95% percentile bootstrap, resampling QUESTIONS (pairs and repeats stay with their
question), 4,000 draws, fixed seed - the same clustering as the main paper.
Differences are paired: every pair runs 3x in each condition, so (Urdu - English) is computed
on identical question-strategy pairs.
"""
import argparse, glob, os, re
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--judged-dir", default=os.path.join(ROOT, "judged"))
p.add_argument("--pairs", default=os.path.join(ROOT, "data", "urdu_pairs_192.csv"))
p.add_argument("--results-dir", default=os.path.join(ROOT, "conversations"))
p.add_argument("--out", default=os.path.join(ROOT, "results"))
p.add_argument("--boot", type=int, default=4000)
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
FIG = os.path.join(args.out, "figures"); TAB = os.path.join(args.out, "tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
rng = np.random.default_rng(2027)

MODELS = {"gemini": "Gemini 3.5 Flash-Lite", "luna": "GPT-6 Luna", "haiku": "Claude Haiku 5.5"}
COND = ["en", "ur", "roman"]
COND_NAME = {"en": "English", "ur": "Urdu script", "roman": "Roman Urdu"}
COLOR = {"en": "#2a78d6", "ur": "#eb6834", "roman": "#1baf7a"}   # validated categorical slots 1-3
METRICS = {"any_break": "Any break", "substantive": "Substantive break"}

# ------------------------------------------------------------------ load
frames = []
for key in MODELS:
    hits = glob.glob(os.path.join(args.judged_dir, f"*{key}*urdu*_judged.csv"))
    assert len(hits) == 1, f"expected one judged file for {key} in {args.judged_dir}, found {hits}"
    d = pd.read_csv(hits[0]); d["model"] = key; frames.append(d)
D = pd.concat(frames, ignore_index=True)
P = pd.read_csv(args.pairs)[["pair_id", "safe_n", "weight", "stratum_size"]]
D = D.merge(P, on="pair_id", how="left", validate="many_to_one")
assert D["safe_n"].notna().all(), "some pair_ids are not in the pairs file"

check = D.groupby(["model", "condition"]).agg(conversations=("id", "size"),
                                              judge_errors=("judge_error", lambda s: s.notna().sum()),
                                              pairs=("pair_id", "nunique"))
check.to_csv(os.path.join(TAB, "00_completeness.csv"))
D = D[D["judge_error"].isna() & D["any_break"].notna()].copy()
for m in METRICS: D[m] = D[m].astype(int)
JC = [f"j{t}" for t in range(1, 6)]
D["first_break_turn"] = D[JC].apply(lambda r: next((t for t in range(1, 6) if r[f"j{t}"] == "break"), np.nan), axis=1)

QIDS = np.sort(D["question_id"].unique())
qidx = {q: i for i, q in enumerate(QIDS)}
BOOT = rng.integers(0, len(QIDS), size=(args.boot, len(QIDS)))     # one shared set of question draws


def qsums(g, col, weighted=False):
    """per-question numerator and denominator (optionally weighted)"""
    w = g["weight"] if weighted else pd.Series(1.0, index=g.index)
    num = np.zeros(len(QIDS)); den = np.zeros(len(QIDS))
    s = pd.DataFrame({"q": g["question_id"].map(qidx), "n": g[col] * w, "d": w}).groupby("q").sum()
    num[s.index] = s["n"]; den[s.index] = s["d"]
    return num, den


def rate_ci(g, col, weighted=False):
    num, den = qsums(g, col, weighted)
    est = num.sum() / den.sum()
    bs = num[BOOT].sum(1) / den[BOOT].sum(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return est, lo, hi


def diff_ci(g, col, a, b, weighted=False):
    """rate(a) - rate(b) on the same pairs, question bootstrap"""
    na, da = qsums(g[g.condition == a], col, weighted)
    nb, db = qsums(g[g.condition == b], col, weighted)
    est = na.sum() / da.sum() - nb.sum() / db.sum()
    bs = na[BOOT].sum(1) / da[BOOT].sum(1) - nb[BOOT].sum(1) / db[BOOT].sum(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p_two = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return est, lo, hi, p_two


def groups():
    for m in MODELS:
        yield MODELS[m], D[D.model == m]
    yield "Pooled", D


pct = lambda x: 100 * x

# ------------------------------------------------------------------ 1. rates and paired differences
rows, drows = [], []
for weighted in (False, True):
    for name, g in groups():
        for col in METRICS:
            r = {"weighting": "benchmark-weighted" if weighted else "sample", "model": name, "metric": col}
            for c in COND:
                e, lo, hi = rate_ci(g[g.condition == c], col, weighted)
                r[c] = pct(e); r[c + "_lo"] = pct(lo); r[c + "_hi"] = pct(hi)
            rows.append(r)
            for a, b, lab in [("ur", "en", "Urdu script - English"), ("roman", "en", "Roman Urdu - English"),
                              ("ur", "roman", "Urdu script - Roman Urdu")]:
                e, lo, hi, pv = diff_ci(g, col, a, b, weighted)
                drows.append({"weighting": r["weighting"], "model": name, "metric": col, "contrast": lab,
                              "diff_pts": pct(e), "lo": pct(lo), "hi": pct(hi), "boot_p": pv})
R = pd.DataFrame(rows).round(1); DF = pd.DataFrame(drows)
DF[["diff_pts", "lo", "hi"]] = DF[["diff_pts", "lo", "hi"]].round(1); DF["boot_p"] = DF["boot_p"].round(4)
R.to_csv(os.path.join(TAB, "01_rates.csv"), index=False)
DF.to_csv(os.path.join(TAB, "02_paired_differences.csv"), index=False)

# ------------------------------------------------------------------ 2. by English safety level (safe_n)
srows = []
for name, g in groups():
    for s in range(4):
        gs = g[g.safe_n == s]
        r = {"model": name, "safe_n": s, "pairs": gs.pair_id.nunique()}
        for c in COND:
            e, lo, hi = rate_ci(gs[gs.condition == c], "any_break")
            r[c] = pct(e); r[c + "_lo"] = pct(lo); r[c + "_hi"] = pct(hi)
        for a, lab in [("ur", "ur-en"), ("roman", "roman-en")]:
            e, lo, hi, _ = diff_ci(gs, "any_break", a, "en")
            r[lab] = pct(e); r[lab + "_lo"] = pct(lo); r[lab + "_hi"] = pct(hi)
        srows.append(r)
S = pd.DataFrame(srows).round(1); S.to_csv(os.path.join(TAB, "03_by_safe_n.csv"), index=False)

# ------------------------------------------------------------------ 3. by category (pooled over models)
crows = []
for cat, g in D.groupby("category"):
    r = {"category": cat}
    for c in COND:
        r[c] = pct(g[g.condition == c]["any_break"].mean())
    for a, lab in [("ur", "ur-en"), ("roman", "roman-en")]:
        e, lo, hi, _ = diff_ci(g, "any_break", a, "en")
        r[lab] = pct(e); r[lab + "_lo"] = pct(lo); r[lab + "_hi"] = pct(hi)
    crows.append(r)
C = pd.DataFrame(crows).round(1); C.to_csv(os.path.join(TAB, "04_by_category_pooled.csv"), index=False)

# ------------------------------------------------------------------ 4. when the break happens
trows = []
for name, g in groups():
    for c in COND:
        gc = g[g.condition == c]
        r = {"model": name, "condition": c}
        for t in range(1, 6):
            r[f"broken_by_turn{t}"] = pct((gc.first_break_turn <= t).mean())
        r["median_first_break_turn"] = gc.first_break_turn.median()
        trows.append(r)
T = pd.DataFrame(trows).round(1); T.to_csv(os.path.join(TAB, "05_cumulative_by_turn.csv"), index=False)

# ------------------------------------------------------------------ 5. pair-level flips (3 repeats each)
frows = []
for name, g in groups():
    k = g.groupby(["model", "pair_id", "condition"])["any_break"].max().unstack()   # broke in >=1 of 3 runs
    never_en = k[k["en"] == 0]; always_en = k[k["en"] == 1]
    frows.append({"model": name, "pairs": len(k),
                  "never broke in English (3 runs)": len(never_en),
                  "  ...broke in Urdu script": int(never_en["ur"].sum()),
                  "  ...broke in Roman Urdu": int(never_en["roman"].sum()),
                  "broke in English": len(always_en),
                  "  ...never broke in Urdu script": int((always_en["ur"] == 0).sum()),
                  "  ...never broke in Roman Urdu": int((always_en["roman"] == 0).sum())})
FL = pd.DataFrame(frows); FL.to_csv(os.path.join(TAB, "06_pair_flips.csv"), index=False)

# ------------------------------------------------------------------ 6. judge label mix
L = D.assign(all_refusal=(D.conv_label == "refusal").astype(int)).groupby(["model", "condition"])[
    ["any_break", "substantive", "all_refusal"]].mean().mul(100).round(1)
L.to_csv(os.path.join(TAB, "07_label_mix.csv"))

# ------------------------------------------------------------------ 7. provider blocks and language drift (optional)
blocks_txt = ""; BLOCKS = None
if args.results_dir:
    AR = re.compile(r"[؀-ۿ]"); LA = re.compile(r"[A-Za-z]")
    UR_W = set("hai hain ka ki ke kya main mujhe aap ko se mein nahi aur yeh woh kaise kyun kar karna sakta sakte hota "
               "hoti tha thi apne mera meri hum ek koi bhi to par liye wala wali ye ho".split())
    EN_W = set("the is are you and of to what how can it that this for with please do does i".split())

    def drift(text, cond):
        s = text if isinstance(text, str) else ""
        if cond == "ur":
            a, l = len(AR.findall(s)), len(LA.findall(s))
            return (a + l) > 0 and a / (a + l) < 0.5
        if cond == "roman":
            w = re.findall(r"[a-z]+", s.lower())
            return len(w) >= 5 and sum(x in EN_W for x in w) > sum(x in UR_W for x in w)
        return False

    brows = []
    for key in MODELS:
        hits = [h for h in glob.glob(os.path.join(args.results_dir, f"results_*{ {'gemini':'gemini','luna':'luna','haiku':'haiku'}[key] }*urdu*.csv"))]
        hits = sorted(hits, key=lambda h: ("complete" not in h, h))
        if not hits:
            print(f"  (no raw results file for {key} in {args.results_dir}; skipping blocks/drift)"); continue
        r = pd.read_csv(hits[0])
        for c in COND:
            rc = r[r.condition == c]
            blk = pd.concat([(rc[f"Response Turn {t}"].isna() | (rc[f"Response Turn {t}"].astype(str).str.strip() == ""))
                             for t in range(1, 6)], axis=1)
            api_ref = pd.concat([rc[f"FinishReason{t}"].astype(str).eq("refusal") for t in range(1, 6)], axis=1)
            att = pd.concat([rc[f"Prompt Turn {t}"].map(lambda s: drift(s, c)) for t in range(1, 6)], axis=1)
            tut = pd.concat([rc[f"Response Turn {t}"].map(lambda s: drift(s, c)) for t in range(1, 6)], axis=1)
            fr = pd.concat([rc[f"FinishReason{t}"].astype(str) + "|" + rc[f"BlockReason{t}"].astype(str) for t in range(1, 6)], axis=1)
            prov = (blk.to_numpy() | api_ref.to_numpy()
                    | fr.apply(lambda col: col.str.contains("PROHIBITED|refusal", regex=True)).to_numpy())
            brows.append({"model": MODELS[key], "condition": c, "conversations": len(rc),
                          "convs with >=1 provider block %": round(100 * prov.any(axis=1).mean(), 1),
                          "convs with >=1 empty/blocked reply": int(blk.any(axis=1).sum()),
                          "convs with an API refusal (Claude)": int(api_ref.any(axis=1).sum()),
                          "attacker turns off-language %": round(100 * att.to_numpy().mean(), 1),
                          "tutor turns off-language %": round(100 * tut.to_numpy().mean(), 1)})
    B = pd.DataFrame(brows); B.to_csv(os.path.join(TAB, "08_blocks_and_drift.csv"), index=False)
    blocks_txt = B.to_markdown(index=False) if len(B) else ""
    BLOCKS = B

# ------------------------------------------------------------------ LaTeX main table
def cell(r, c):
    return f"{r[c]:.1f} {{\\scriptsize[{r[c+'_lo']:.0f}, {r[c+'_hi']:.0f}]}}"

def dcell(model, metric, contrast, weighting="sample"):
    x = DF[(DF.model == model) & (DF.metric == metric) & (DF.contrast == contrast) & (DF.weighting == weighting)].iloc[0]
    star = "$^{*}$" if (x.lo > 0 or x.hi < 0) else ""
    f0 = lambda v: "0" if round(v) == 0 else f"{v:+.0f}"
    return f"{x.diff_pts:+.1f}{star} {{\\scriptsize[{f0(x.lo)}, {f0(x.hi)}]}}"

lines = [r"\begin{table*}[t]", r"\centering\small",
         r"\caption{Break rates (\%) by language condition on the 192 sampled pairs (3 runs per pair and condition; "
         r"576 conversations per cell). Brackets: 95\% question-bootstrap intervals. Differences are paired on identical "
         r"question--strategy pairs; $^{*}$ the interval excludes zero.}",
         r"\label{tab:urdu-main}",
         r"\begin{tabular}{llccc|cc}", r"\toprule",
         r"Metric & Model & English & Urdu script & Roman Urdu & Urdu $-$ Eng. & Roman $-$ Eng. \\", r"\midrule"]
for col, lab in METRICS.items():
    for i, (_, r) in enumerate(R[(R.metric == col) & (R.weighting == "sample")].iterrows()):
        lines.append(f"{lab if i == 0 else ''} & {r.model} & {cell(r,'en')} & {cell(r,'ur')} & {cell(r,'roman')} & "
                     f"{dcell(r.model, col, 'Urdu script - English')} & {dcell(r.model, col, 'Roman Urdu - English')} \\\\")
    lines.append(r"\midrule" if col == "any_break" else r"\bottomrule")
lines += [r"\end{tabular}", r"\end{table*}"]
open(os.path.join(TAB, "table_main.tex"), "w").write("\n".join(lines) + "\n")

# ------------------------------------------------------------------ figures
plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#888", "axes.labelcolor": "#333",
                     "xtick.color": "#555", "ytick.color": "#555", "legend.frameon": False,
                     "pdf.fonttype": 42, "savefig.dpi": 300})
GRID = dict(axis="y", color="#e6e6e6", linewidth=0.6)

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)

def legend_handles():
    return [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=COLOR[c], markersize=7, label=COND_NAME[c])
            for c in COND]

# Fig 1: break rate per model and condition (two panels: any / substantive)
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)
names = list(MODELS.values()) + ["Pooled"]
x = np.arange(len(names)); w = 0.26
for ax, (col, lab) in zip(axes, METRICS.items()):
    sub = R[(R.metric == col) & (R.weighting == "sample")].set_index("model").loc[names]
    for k, c in enumerate(COND):
        xs = x + (k - 1) * w
        ax.bar(xs, sub[c], w - 0.03, color=COLOR[c], zorder=3)
        ax.errorbar(xs, sub[c], yerr=[sub[c] - sub[c + "_lo"], sub[c + "_hi"] - sub[c]], fmt="none",
                    ecolor="#333", elinewidth=0.8, capsize=1.8, zorder=4)
    ax.set_xticks(x); ax.set_xticklabels([n.replace(" ", "\n", 1) if n != "Pooled" else n for n in names])
    ax.set_title(lab, fontsize=9, color="#222"); ax.set_ylim(0, 100); ax.grid(**GRID); ax.set_axisbelow(True)
axes[0].set_ylabel("Conversations with a break (%)")
axes[1].legend(handles=legend_handles(), loc="upper right", fontsize=7, ncol=1)
save(fig, "fig1_break_rates")

# Fig 2: paired differences vs English (forest plot)
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2), sharey=True)
for ax, (col, lab) in zip(axes, METRICS.items()):
    for k, (contrast, c) in enumerate([("Urdu script - English", "ur"), ("Roman Urdu - English", "roman")]):
        sub = DF[(DF.metric == col) & (DF.contrast == contrast) & (DF.weighting == "sample")].set_index("model").loc[names]
        ys = np.arange(len(names))[::-1] + (0.15 if k == 0 else -0.15)
        ax.errorbar(sub.diff_pts, ys, xerr=[sub.diff_pts - sub.lo, sub.hi - sub.diff_pts], fmt="o", ms=4.5,
                    color=COLOR[c], ecolor=COLOR[c], elinewidth=1.2, capsize=0, label=COND_NAME[c] + " − English")
    ax.axvline(0, color="#777", linewidth=0.8, zorder=0)
    ax.set_yticks(np.arange(len(names))[::-1]); ax.set_yticklabels(names)
    ax.axhline(0.5, color="#ddd", linewidth=0.6)
    ax.set_title(lab, fontsize=9, color="#222"); ax.set_xlabel("Difference from English (percentage points)")
    ax.grid(axis="x", color="#eee", linewidth=0.6); ax.set_axisbelow(True)
axes[0].legend(loc="lower left", fontsize=7, bbox_to_anchor=(0, 1.08), ncol=2)
save(fig, "fig2_paired_differences")

# Fig 3: by English safety level (pooled)
fig, ax = plt.subplots(figsize=(3.4, 2.4))
sp = S[S.model == "Pooled"].set_index("safe_n")
xs = np.arange(4)
for k, c in enumerate(COND):
    ax.errorbar(xs + (k - 1) * 0.12, sp[c], yerr=[sp[c] - sp[c + "_lo"], sp[c + "_hi"] - sp[c]], fmt="-o", ms=4,
                lw=1.6, color=COLOR[c], elinewidth=0.9, capsize=0, label=COND_NAME[c])
ax.set_xticks(xs); ax.set_xticklabels(["0\n(fragile)", "1", "2", "3\n(robust)"])
ax.set_xlabel("Models safe on the pair in the English benchmark"); ax.set_ylabel("Conversations with a break (%)")
ax.set_ylim(0, 100); ax.grid(**GRID); ax.set_axisbelow(True); ax.legend(fontsize=7, loc="upper right")
save(fig, "fig3_by_safe_n")

# Fig 4: change by category (pooled)
fig, ax = plt.subplots(figsize=(3.4, 2.5))
Cs = C.sort_values("ur-en")
ys = np.arange(len(Cs))
for k, (lab, c) in enumerate([("ur-en", "ur"), ("roman-en", "roman")]):
    yy = ys + (0.15 if k == 0 else -0.15)
    ax.errorbar(Cs[lab], yy, xerr=[Cs[lab] - Cs[lab + "_lo"], Cs[lab + "_hi"] - Cs[lab]], fmt="o", ms=4,
                color=COLOR[c], elinewidth=1.1, capsize=0, label=COND_NAME[c] + " − English")
ax.axvline(0, color="#777", linewidth=0.8, zorder=0)
ax.set_yticks(ys); ax.set_yticklabels(Cs.category)
ax.set_xlabel("Difference in break rate (pp), pooled over models")
ax.grid(axis="x", color="#eee", linewidth=0.6); ax.set_axisbelow(True)
ax.legend(fontsize=7, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
save(fig, "fig4_by_category")

# Fig 5: cumulative share broken by turn (pooled)
fig, ax = plt.subplots(figsize=(3.4, 2.3))
tp = T[T.model == "Pooled"].set_index("condition")
for c in COND:
    ax.plot(range(1, 6), [tp.loc[c, f"broken_by_turn{t}"] for t in range(1, 6)], "-o", ms=4, lw=1.8,
            color=COLOR[c], label=COND_NAME[c])
ax.set_xticks(range(1, 6)); ax.set_xlabel("Turn"); ax.set_ylabel("Conversations broken so far (%)")
ax.set_ylim(0, 100); ax.grid(**GRID); ax.set_axisbelow(True); ax.legend(fontsize=7, loc="upper left")
save(fig, "fig5_cumulative_by_turn")

# Fig 6: provider-level blocks (empty / filtered replies, API refusals) by model and condition
if BLOCKS is not None and len(BLOCKS):
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    mods = [m for m in MODELS.values() if m in set(BLOCKS.model)]
    x = np.arange(len(mods)); w = 0.26
    for k, c in enumerate(COND):
        v = [BLOCKS[(BLOCKS.model == m) & (BLOCKS.condition == c)]["convs with >=1 provider block %"].iloc[0] for m in mods]
        bars = ax.bar(x + (k - 1) * w, v, w - 0.03, color=COLOR[c], label=COND_NAME[c], zorder=3)
        for b, val in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, val + 0.4, f"{val:.0f}", ha="center", va="bottom", fontsize=6, color="#444")
    ax.set_xticks(x); ax.set_xticklabels([m.replace(" ", "\n", 1) for m in mods])
    ax.set_ylabel("Conversations with a provider\nblock or API refusal (%)"); ax.grid(**GRID); ax.set_axisbelow(True)
    ax.legend(fontsize=7, loc="upper left")
    save(fig, "fig6_provider_blocks")

# ------------------------------------------------------------------ summary.md
def g(model, metric, contrast, weighting="sample"):
    x = DF[(DF.model == model) & (DF.metric == metric) & (DF.contrast == contrast) & (DF.weighting == weighting)].iloc[0]
    return f"{x.diff_pts:+.1f} pts [{x.lo:+.1f}, {x.hi:+.1f}]"

def rr(model, metric, c, weighting="sample"):
    x = R[(R.model == model) & (R.metric == metric) & (R.weighting == weighting)].iloc[0]
    return f"{x[c]:.1f}% [{x[c+'_lo']:.1f}, {x[c+'_hi']:.1f}]"

md = ["# Urdu experiment: results", "",
      f"{len(D)} judged conversations ({D.pair_id.nunique()} pairs, {D.question_id.nunique()} questions, 3 conditions x 3 runs x 3 models). "
      "95% intervals: question-clustered bootstrap. Differences are paired on identical pairs.", "",
      "## Headline (pooled over the three models)", "",
      f"- Any break: English {rr('Pooled','any_break','en')}, Urdu script {rr('Pooled','any_break','ur')}, Roman Urdu {rr('Pooled','any_break','roman')}",
      f"- Urdu script - English: {g('Pooled','any_break','Urdu script - English')}; Roman Urdu - English: {g('Pooled','any_break','Roman Urdu - English')}; Urdu script - Roman Urdu: {g('Pooled','any_break','Urdu script - Roman Urdu')}",
      f"- Substantive break: Urdu script - English {g('Pooled','substantive','Urdu script - English')}; Roman Urdu - English {g('Pooled','substantive','Roman Urdu - English')}",
      f"- Benchmark-weighted (stratum weights), any break: Urdu script - English {g('Pooled','any_break','Urdu script - English','benchmark-weighted')}; Roman Urdu - English {g('Pooled','any_break','Roman Urdu - English','benchmark-weighted')}",
      "", "## Per model (any break)", ""]
for m in MODELS.values():
    md.append(f"- **{m}**: EN {rr(m,'any_break','en')} | UR {rr(m,'any_break','ur')} | Roman {rr(m,'any_break','roman')} | "
              f"UR-EN {g(m,'any_break','Urdu script - English')} | Roman-EN {g(m,'any_break','Roman Urdu - English')}")
md += ["", "## By English safety level (pooled, any break %)", "", S[S.model == "Pooled"][["safe_n", "pairs", "en", "ur", "roman", "ur-en", "roman-en"]].to_markdown(index=False),
       "", "## By category (pooled, any break %)", "", C[["category", "en", "ur", "roman", "ur-en", "roman-en"]].to_markdown(index=False),
       "", "## Pair-level flips (a pair 'broke' if any of its 3 runs broke)", "", FL.to_markdown(index=False),
       "", "## Cumulative share broken by turn (pooled)", "", T[T.model == "Pooled"].to_markdown(index=False)]
if blocks_txt:
    md += ["", "## Provider blocks and language drift", "", blocks_txt]
md += ["", "## Files", "", "- tables/: 00_completeness ... 08_blocks_and_drift (CSV), table_main.tex",
       "- figures/: fig1_break_rates, fig2_paired_differences, fig3_by_safe_n, fig4_by_category, fig5_cumulative_by_turn, fig6_provider_blocks (PDF + PNG)"]
open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
print("\n".join(md))
