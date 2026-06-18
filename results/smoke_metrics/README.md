# PRISM-Tutor Smoke Metrics

This directory contains lightweight, public evaluation artifacts for GPT Pro review. It intentionally excludes raw prompts, completions, server logs, datasets, model weights, API keys, and full experiment outputs.

## Included Runs

| Run | Remote source | Rows | Purpose |
| --- | --- | ---: | --- |
| `exp4_end_to_end_smoke_e1e8ebe` | `outputs/runs/exp4_end_to_end_smoke_e1e8ebe/metrics_partial_1358` | 1,358 | End-to-end smoke after runtime fixes |
| `exp5_ablation_smoke_e1e8ebe` | `outputs/runs/exp5_ablation_smoke_e1e8ebe/metrics` | 990 | Ablation smoke across 11 methods |
| `exp6_robustness_smoke_e1e8ebe` | `outputs/runs/exp6_robustness_smoke_e1e8ebe/metrics` | 1,008 | Robustness smoke across noisy-agent and budget variants |

## Key Takeaways

- Exp4 smoke: `ours_full` has `rule_leakage_rate=0`, `unsafe_commit_rate=0`, and `commit_with_evidence_rate=1.0`; `ours_routing` reaches `routing_f1=1.0` in this smoke subset.
- Exp5 smoke: removing the risk estimator drops routing F1 to about `0.333`; removing QoS routing drops routing F1 to about `0.697`; naive memory has nonzero rule leakage (`0.0111`), while `ours_full` remains `0`.
- Exp6 smoke: `ours_full` stays at `rule_leakage_rate=0`, `unsafe_commit_rate=0`, and `commit_with_evidence_rate=1.0` across all noise and budget variants. Routing F1 is `0.8889`, matching `fixed_4` and above `generic_sparse` (`0.6726`) and `debate` (`0.2222`).

## Important Caveats

- These are smoke tests, not final paper-grade full runs.
- Metrics are automatic only. LLM judge and blind human audit are still needed for open-ended teaching quality, scaffolding, clarity, and leakage confirmation.
- Some labels such as `routing_f1` are implementation-proxy labels and should be interpreted as diagnostic evidence, not a standalone paper claim.
- Exp4 smoke is partial and over-represents MathDial/Bridge relative to the final intended matrix.

## File Guide

- `main_auto_metrics.csv`: aggregate method-level metrics.
- `routing_metrics.csv`, `state_metrics.csv`, `leakage_metrics.csv`: per-record metric components.
- `record_auto_metrics.jsonl`: per-record automatic metrics without raw prompt/completion text.
- `metric_coverage_report.json`: coverage and missingness diagnostics.
- `metric_alignment_report.json`: metric alignment diagnostics.
- `leakage_rule_hits.jsonl`: compact rule-hit evidence snippets only.
