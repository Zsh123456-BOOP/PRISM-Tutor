# Next smoke plan — for Codex to run on the server

## Round-4 changes (baseline FIDELITY — needed for Exp1/Exp2 to mean anything)
The round/debate/random baselines were not faithful, so the efficiency narrative had
no expensive baseline to beat and "debate" didn't debate:
- `debate` / `fixed_2/3/4_rounds` now genuinely deliberate K rounds (non-final agents
  run K times over accumulating context; the response is produced once). Cost now
  scales with rounds: single 1, fixed_4 5, debate 9, fixed_3_rounds 10, fixed_4_rounds
  17 agent calls. fixed_4 (B2, 1 round) is unchanged.
- `random_routing` is now an actual seeded-random agent subset (was aliased to the
  deterministic generic-sparse planner) — the real negative control for Exp1.
=> The confirmation smoke MUST include Exp1 (routing) and Exp2 (budget), not just
Exp4/Exp5, so these contrasts are exercised. Expected: ours (1-2 rounds) is clearly
cheaper than debate / fixed_3/4_rounds at parity quality — this is where the Pareto/
efficiency claim lives (NOT ours < fixed_4).

## Round-3 changes (since main_bbb8aac; what THIS run validates)
bbb8aac passed gates 1 (parse 0.08%), 3 (misconception F1 parity: ours 0.37-0.38 ≥
fixed_4 0.34), 4 (solver ~0.75). It failed 5 (state) and 7 (cost). Root causes found
and fixed:
- COST: `candidate_misconceptions` (55 long labels, ~1658 tok) was injected into
  EVERY agent prompt → ~40% of MMB tokens. Now scoped to the misconception agent
  only; runtime_state passed to agents is trimmed to latest-output-per-agent. Expect
  MMB tokens ~halved for ALL methods (solver prompt ~2000→~290), so ours_routing ≈
  fixed_4 and the absolute numbers become sane.
- STATE: the verifier flagged `state_conflict` on ~32% of single-turn MMB samples →
  `_tentative_all` withheld ~1/3 of commits → external_state < naive. A conflict with
  EMPTY prior state is a false positive; commit now proceeds by confidence on turn 1
  (the two-phase conflict gate only applies when prior state exists). Expect ours MMB
  external_state ≈ naive (parity), keeping the commit-safety properties.

## Changes since the main_73de01e smoke
- Solver parse failures (all 146 were the solver, thinking-truncation): solver budget
  4096 + deterministic answer-salvage fallback. Target parse_fail < 0.5%.
- Misconception F1 < fixed_4 (ours diagnosed without the reference solution): the
  solver is now routed whenever the misconception agent is, and runs FIRST
  (canonical order solver→misconception→…). Validated: solver-before-misconception
  40/40 on all datasets.
- State < naive (two-phase was over-conservative): commit threshold 0.70→0.55
  (still gated by the verifier conflict check).
- Cost framing corrected: **M3 is NOT the efficiency variant** — it adds state
  management, so it legitimately costs more than fixed_4. The Pareto/efficiency
  claim is carried by **M1 (ours_routing) / M2 (ours_routing_budget)** vs fixed_4 /
  debate; M3 is compared against the STATE baselines (no_memory / naive / single_writer
  / two_phase) in Exp3, where its cost is comparable.



Goal: confirm that, after the T1–T6 changes, PRISM-Tutor is competitive on the
paper main-line metrics at lower deliberation cost — on a sample large enough to
read. NO judge, NO human audit, NO full run.

## Why a new smoke
The previous smoke (`main_e780244`) used only **16 unique samples/dataset**, so the
gold metrics were noise. It also predates: misconception/pedagogy routing on a
student-answer signal (T1), real `rounds` logging (T2), solver thinking + per-agent
token budgets (T4), and the `hit@1/@3` diagnostics (T3). Re-run on the new commit.

## Scope (cover ALL experiments, not just Exp4/5 — each narrative needs its contrast)
- Datasets: MathDial test, Bridge test, Misconception Benchmark (all 220).
- Exp4 core: `single_tutor, fixed_2, fixed_4, debate, generic_sparse,
  difficulty_routing, ours_routing, ours_routing_budget, ours_full`.
- Exp1 routing: `random_routing, fixed_all_agents, difficulty_routing, generic_sparse,
  oracle_routing, ours_routing` (random is now a real negative control).
- Exp2 budget: `single_round, fixed_2_rounds, fixed_3_rounds, fixed_4_rounds, debate,
  generic_early_stopping, ours_routing_budget` (round baselines now truly iterate →
  the token-quality Pareto / risk-bucket curve lives here).
- Exp3 state: `no_memory, naive_shared_memory, single_writer, two_phase_commit, ours_full`.
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
3. Misconception (target = PARITY, not superiority — fixed_4 is the always-all-agents
   diagnosis ceiling): ours_full `misconception_f1`/`hit@3` within noise of fixed_4
   (solver now runs first, so the gap should close from ~0.10). Verify solver is
   routed before misconception on ~100% of diagnosed samples. The efficiency win is
   carried by M1/M2, not by beating fixed_4 on F1.
4. Solver: with solver thinking ON, solver-running methods reach non-floor MathDial
   solver correctness (>~0.3) and the value is consistent across methods (confirming
   it is a controlled variable, not a differentiator).
5. State (reframed — the ONLY gold student-state data is single-turn MMB, where
   two-phase commit ≈ naive by design; the multi-turn reconciliation advantage has
   no gold-state dataset to measure on): target ours `external_state_accuracy` ≈
   naive (parity, no longer below it after the turn-1 commit fix), WITH the commit
   safety the baselines lack — `unsafe_commit_rate` = 0, `commit_with_evidence` = 1,
   and `final_state_contradiction` ≤ naive. State is a reliability claim, not an
   accuracy-superiority claim; frame it as such in the paper.
6. Mechanism real: `rounds` varies (not pinned to 3); the budget/verifier loop fires
   on some hard cases (rounds > 1 for a non-trivial fraction); buckets non-degenerate.
7. Cost Pareto (carried by M1/M2, NOT M3): on ≥2/3 datasets `ours_routing` /
   `ours_routing_budget` total tokens ≤ fixed_4 at parity diagnosis quality, and
   below debate/fixed-N-rounds via fewer rounds. M3 (ours_full) is compared to the
   state baselines (no_memory/naive/single_writer/two_phase) in Exp3 — comparable
   cost there, NOT expected to undercut fixed_4 (it adds state management).

If gates 2–7 hold, the main-line is supportable on automatic metrics; only then add
the LLM judge + 200-sample human audit (still deferred until this clears).
