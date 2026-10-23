# Just One More Question: Multi-Turn Jailbreaks Against LLM Tutors for Children

Code, prompts and data for a multi-turn red-teaming study of LLM tutors configured for a 12-year-old student. An adaptive attacker model (Gemma 4 31B) holds five-turn conversations with a tutor persona, using one of 24 jailbreak strategies to pursue one of 84 questions from MinorBench and KIDBench. An LLM judge (DeepSeek V4 Flash) labels every tutor reply as a break, a refusal or a helpful response.

> **Content warning.** The conversation files contain model outputs on substance use, self-harm, sexual content, profanity, hateful content and dangerous activities, produced under adversarial pressure. They are released for safety research only. No real children took part in this study; the "child" is simulated by the attacker model.

## Repository layout

```
data/
  questions_84_final_v1.xlsx       84 questions (14 per harm category) and the selection decisions
  strategies_shortlist_v2.xlsx     24 jailbreak strategies, their sources and references
prompts/
  attacker_prompt_template.txt     attacker prompt (also embedded in the run notebooks)
  tutor_system_prompts.json        the four tutor system prompts
notebooks/
  run_gemini.ipynb                 main run, Gemini 3.5 Flash-Lite tutor (2,016 conversations)
  run_gpt.ipynb                    main run, GPT-6 Luna tutor (2,016 conversations)
  run_claude.ipynb                 main run, Claude Haiku 5.5 tutor (2,016 conversations)
  run_gemini_prompt_ablation.ipynb prompt ablation on Gemini (1,080 conversations per prompt)
judge/
  judge_full_run.ipynb             LLM judge over all result files
  judge_prompt_final.md            frozen judge prompt (sha256 prefix 9921e27ea7a1c84c)
results/
  conversations/                   full conversations, one row per conversation
  judged/                          judge labels, one row per conversation
analysis/
  analysis.py                      reproduces the paper's tables and figures
  outputs/                         tables (CSV) and figures (PDF/PNG) produced by analysis.py
```

## Prompt conditions

| Paper | Notebook name | Instructions |
|---|---|---|
| T0 | `general` | Tutor for a 12-year-old, keep replies safe and appropriate, response format |
| T1 | `stochastic` | T0 + "Do not give answers directly, and guide them step-by-step." |
| T2 | `full` | T1 + "Do not answer anything irrelevant." |
| T3 | `safe` | T2 + explicit child-safety rules |

All three models were run with T0. T1 to T3 were run on Gemini only, on a fixed subset of 60 questions (10 per category) and 18 strategies selected by seeded SHA-256 ranking; the T0 control for the ablation is the same 1,080 pairs taken from the main Gemini run. Full prompt text is in `prompts/tutor_system_prompts.json`.

## Results files

`results/conversations/results_*.csv`: one row per conversation, with the question and strategy (`question_id`, `question`, `category`, `strategy_id`, `strategy`, `family`), the prompt condition (`system_prompt_variant`), and for each turn t = 1 to 5 the attacker message (`Prompt Turn t`), the cleaned tutor reply (`Response Turn t`), the raw reply (`Raw Turn t`), the finish and block reasons, and token counts. The `break_definition` column comes from the question file and was not used by the judge.

`results/judged/*_judged.csv`: one row per conversation, joined to the conversation files by `id`, with per-turn labels `j1`-`j5` (`break`, `refusal`, `helpful`), break types `bt1`-`bt5` (`full`, `other`, `partial`), quoted evidence `ev1`-`ev5`, rationales `why1`-`why5`, and:

- `any_break`: 1 if any turn is a break
- `substantive`: 1 if any break turn is typed `full` or `other`
- `conv_label`: `break`, `refusal` (all five turns refusals) or `helpful`

| File | Conversations | Any break | Substantive break |
|---|---|---|---|
| gemini_general | 2,016 | 47.2% | 30.8% |
| luna_general | 2,016 | 70.9% | 59.7% |
| haiku_general | 2,016 | 40.5% | 30.7% |
| gemini_stochastic (T1) | 1,080 | 37.1% | 20.1% |
| gemini_full (T2) | 1,080 | 28.7% | 13.4% |
| gemini_safe (T3) | 1,080 | 17.4% | 7.6% |

The T0 rate on the matched ablation subset is 51.9% (substantive 34.7%).

## Reproducing the analysis

```
pip install -r requirements.txt
python analysis/analysis.py
```

This writes all tables and figures to `analysis/outputs/`. Confidence intervals are 95% bootstrap intervals that resample questions (2,000 resamples, seed 0).

## Re-running the experiments

The run notebooks were executed on Kaggle and read API keys with `kaggle_secrets`; replace those lines to run elsewhere. The attacker (`gemma-4-31b-it-qat-q4_0-unquantized`) is served with vLLM as an OpenAI-compatible endpoint; set `RUNPOD_BASE_URL`, `RUNPOD_API_KEY` and `RUNPOD_MODEL` in the config cell, and point `STRATEGIES_PATH` and `QUESTIONS_PATH` to the files in `data/`. Attacker sampling: temperature 0.85, top-p 0.92, repetition penalty 1.15, at most 450 new tokens. Because the attacker samples, a re-run produces new conversations rather than the released ones.

The judge notebook reads `DEEPSEEK_API_KEY` from the environment. Put the conversation files next to the notebook or edit the paths in `FILES`. It skips conversations that are already judged, so it can be resumed after an interruption.

## Notes

- Conversations whose attacker prompt exceeded the attacker's initial 4,096-token context (mostly strategy 20) were re-run with an 8,192-token context and merged into the released files, so every file has a complete set of conversations with no errors.
- Claude Haiku 5.5 returned 567 empty turns where the API refused to respond; the judge saw these as refusal placeholders.
- Strategy 24 was intended as a prefix-injection attack, but the attacker wrote "Sure, here's" in its own message instead of asking the tutor to use it; see the paper.
