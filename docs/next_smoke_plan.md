# Next smoke plan (post T1–T6) — for Codex to run on the server

Goal: confirm that, after the T1–T6 changes, PRISM-Tutor is competitive on the
paper main-line metrics at lower deliberation cost — on a sample large enough to
read. NO judge, NO human audit, NO full run.

## Why a new smoke
The previous smoke (`main_e780244`) used only **16 unique samples/dataset**, so the
gold metrics were noise. It also predates: misconception/pedagogy routing on a
student-answer signal (T1), real `rounds` logging (T2), solver thinking + per-agent
token budgets (T4), and the `hit@1/@3` diagnostics (T3). Re-run on the new commit.

## Scope
- Datasets: MathDial test, Bridge test, Misconception Benchmark (all 220).
- Methods (Exp4 core): `single_tutor, fixed_2, fixed_4, debate, generic_sparse,
  difficulty_routing, ours_routing, ours_routing_budget, ours_full`.
- Key Exp5 ablations: `ablate_qos_routing, ablate_budget_controller, ablate_state_commit,
  ablate_misconception_risk, replace_two_phase_commit_with_naive_memory`.
- Unique samples: **≥60 per dataset** for MathDial/Bridge; all 220 for Misconception.
- Fixed seed, clean git-frozen commit, `--live-llm`, judge OFF (`dry_run` / not invoked).

## Commands (template; adapt `--limit`/shards to the runner)
```
# rebuild datasets first (final_answer + candidate_misconceptions come from build)
python scripts/01_build_datasets.py

# generation (real Qwen, 2 endpoints round-robin); repeat per experiment
python scripts/02_run_generation.py --experiment exp4_end_to_end --limit 60 \
  --output_dir outputs/runs/smoke_next --live-llm
python scripts/02_run_generation.py --experiment exp5_ablation   --limit 60 \
  --output_dir outputs/runs/smoke_next --live-llm
python scripts/02_run_generation.py --experiment exp6_robustness  --limit 30 \
  --output_dir outputs/runs/smoke_next --live-llm

# metrics only (NO judge)
python scripts/04_compute_metrics.py --generations outputs/runs/smoke_next/generations \
  --gold data/splits --out outputs/runs/smoke_next/metrics
```

## Acceptance gates (ALL must hold before considering a full run)
1. Engineering: error_rows = 0, parse_fail ≈ 0, runtime contains no gold
   (`assert_no_gold_fields` passes), reproducible (clean frozen commit).
2. Routing coverage: ours routes the misconception agent on ≥90% of MMB samples;
   low-risk Bridge turns still drop to a small agent set (cost adaptivity visible).
3. Misconception: ours `misconception_f1` ≥ fixed_4 (or within noise) AND ours
   tokens/agent_calls < fixed_4 (same/ better quality, cheaper). Report `hit@1` and
   `hit@3` — `hit@3` separates "named it" from "ranked first".
4. Solver: with solver thinking ON, solver-running methods reach non-floor MathDial
   solver correctness (>~0.3) and the value is consistent across methods (confirming
   it is a controlled variable, not a differentiator).
5. State: ours `external_state_accuracy` ≥ state-bearing baselines (two_phase / naive)
   and `incorrect_misconception_commit_rate` clearly lower than naive memory.
6. Mechanism real: `rounds` varies (not pinned to 3); the budget/verifier loop fires
   on some hard cases (rounds > 1 for a non-trivial fraction); buckets non-degenerate.
7. Cost Pareto: on ≥2/3 datasets ours total tokens ≤ fixed_4 at equal-or-better
   automatic quality; on MathDial ours is at least not materially worse.

If gates 2–7 hold, the main-line is supportable on automatic metrics; only then add
the LLM judge + 200-sample human audit (still deferred until this clears).
