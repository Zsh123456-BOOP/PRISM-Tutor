# PRISM-Tutor metric policy (judge-free interim)

Roles are emitted in the aggregate metrics so tables/figures can label them.

## Primary (judge-free phase)
- **Cost axis** (co-headline): `total_tokens`, `agent_calls`, `rounds`, `latency`.
  Report relative to a named baseline (fixed_4). latency is hardware-dependent →
  auxiliary only.
- **Misconception F1** — Misconception Benchmark ONLY (fixed 55-label taxonomy,
  constrained classification). Pair with `hit@1` / `hit@3` diagnostics.
- **External student-state correctness** — `external_state_accuracy`,
  `incorrect_misconception_commit_rate`, `final_state_contradiction_rate`. NOTE: the
  only dataset with gold student state is single-turn MMB, where two-phase commit
  reduces to naive (no prior state to reconcile). So accuracy here is a PARITY claim
  (≈ naive), and the two-phase contribution is a RELIABILITY claim — `unsafe_commit_rate`=0,
  `commit_with_evidence`=1, lower `final_state_contradiction` — NOT accuracy superiority.
  The multi-turn reconciliation benefit is unmeasured (no multi-turn gold state).

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
