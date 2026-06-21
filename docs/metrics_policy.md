# PRISM-Tutor metric policy (judge-free interim)

Roles are emitted in the aggregate metrics so tables/figures can label them.

## Primary (judge-free phase)
- **Cost axis** (co-headline): `total_tokens`, `agent_calls`, `rounds`, `latency`.
  Report relative to a named baseline (fixed_4). latency is hardware-dependent →
  auxiliary only.
- **Misconception F1** — Misconception Benchmark ONLY (fixed 55-label taxonomy,
  constrained classification). Pair with `hit@1` / `hit@3` diagnostics.
- **External student-state correctness** — primary: `external_state_accuracy` (on MMB,
  ours_full = 0.368, significantly above the memory-strategy baselines ≤0.19 —
  bootstrap diff CI excludes 0 — but statistically tied with the best ablation variant
  replace_two_phase_with_naive = 0.386, diff CI [−0.07, +0.11] contains 0; so NOT
  "highest". unsafe_commit=0 / commit_with_evidence=1.0 hold for ALL committing methods
  on single-turn MMB, so they are not differentiators here; the two-phase commit's
  multi-turn value is unmeasured) and `misconception_commit_precision`
  (= 1 − incorrect-commit). Safety invariants: `unsafe_commit_rate`=0,
  `commit_with_evidence`≈1. `final_state_contradiction` is a precision/COVERAGE
  TRADE-OFF (it rises when a method commits more labels), NOT a pass/fail gate — do
  NOT claim "ours has lower contradiction" (committing more is why ours is more
  accurate). NOTE: the only gold student-state data is single-turn MMB, where the
  accuracy win comes from the full PRISM state architecture (Exp3) while the commit
  policy alone (Exp5) is ~parity; true multi-turn reconciliation is unmeasured
  (no multi-turn gold) — frame as limitation / future work.

## Diagnostic (process / not a standalone claim)
- `routing_f1` (`routing_metric_role = "diagnostic"`): pseudo-gold is circular.
- `rule_leakage_rate` alone (real leakage needs rule⊕judge later).
- Runtime invariants: `unsafe_commit_rate`, `commit_with_evidence_rate`,
  `tentative_when_conflict_rate`, `state_conflict_rate`.
- `risk_bucket` distribution, `parse_success`, `termination_reason`.

## Controlled variable (NOT a method differentiator)
- `solver_correctness` / `internal_correctness`
  (`solver_correctness_role = "controlled_variable"`): all methods share the same
  solver, so this measures shared model capability, not the orchestration. Report
  it to show the capability floor; do not claim ours improves it.

## Deferred until the automatic main-line clears
- LLM-judge quality (pedagogical / scaffolding / clarity / student-facing),
  judge-confirmed leakage (`final_leakage = rule OR judge`), next-turn quality,
  and the 200-sample blind human audit (κ / Spearman / preference win-rate).

## Division of labor (when judge/human are added)
runtime leakage guard (generation-time, no gold) · offline rule detector (eval,
independent, evidence spans) · LLM judge (DeepSeek; final_response only;
order-swapped, report Conflict Rate; gold/reference for objective dims) · human
audit (blind, calibrates the judge; never edits thresholds or results).
