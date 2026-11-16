# Urdu extension: multi-turn child-safety failures of LLM tutors in Urdu

> **Content warning.** The conversation files contain attacker prompts and tutor replies about self-harm, sexual content, hate speech, dangerous activities, substance use and profanity, written to test child-safety behaviour. They are released for research only.

This folder holds the data and code for the Urdu short paper. It reuses the benchmark, attacker and judge from the main paper and reruns 192 question-strategy pairs in English, Urdu script and Roman Urdu, three runs each, against three tutors (Gemini 3.5 Flash-Lite, GPT-6 Luna, Claude Haiku 5.5).

## Layout

| Folder | Contents |
|---|---|
| `data/` | `questions_84_final_v1.xlsx` (English questions), `questions_84_urdu.xlsx` (Urdu-script and Roman Urdu translations, with translation notes), `strategies_shortlist_v2.xlsx` (the 24 attack strategies), `english_safety_by_pair.csv` (per-pair English outcome from the main paper, used for sampling), `urdu_pairs_192.csv` (the sampled pairs and their weights) |
| `notebooks/` | `run_gemini_urdu.ipynb`, `run_gpt_urdu.ipynb`, `run_claude_urdu.ipynb`: attacker-tutor conversations (5 turns) for each tutor |
| `conversations/` | Raw conversations, one file per tutor (192 pairs x 3 conditions x 3 runs = 1,728 rows each) |
| `judge/` | `judge_prompt_final.md` (judge prompt v7.5) and `judge_full_run_urdu.ipynb` |
| `judged/` | Judge labels per turn and per conversation, one file per tutor |
| `scripts/` | `sample_urdu_pairs.py` (stratified sampling) and `urdu_analysis.py` (all tables and figures) |

## Reproducing

```bash
pip install -r requirements.txt
python scripts/sample_urdu_pairs.py    # rebuilds data/urdu_pairs_192.csv (seed 2027)
python scripts/urdu_analysis.py        # writes results/tables, results/figures, results/summary.md
```

Sampling: pairs are stratified by harm category and by how many of the three tutors were safe on the pair in the original English run (`safe_n`, 0-3), 8 pairs per stratum, at most 2 pairs per question per stratum. Weights (`stratum_size / n_drawn`) recover benchmark-level rates.

The run and judge notebooks need API keys and file paths filled in at the top; they are left blank here. The attacker (Gemma 4 31B) was served with vLLM.
