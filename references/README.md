# PRISM-Tutor reference papers

The PDFs in this directory are downloaded locally for reading and are **git-ignored**
(large binaries). This README is the tracked artifact: it records each paper's
verified venue and how it maps onto a PRISM-Tutor baseline / component, for direct
use in Related Work.

## Framework / baseline-anchor papers (paradigm → baseline)

Controlled-comparison principle: every PRISM-Tutor baseline is a faithful
instantiation of a published communication/coordination paradigm under ONE harness
(same Qwen3-8B, same agent pool, same student-facing final prompt) — only the
orchestration differs.

| Paper | Verified venue (confidence) | Communication paradigm | PRISM baseline |
|---|---|---|---|
| ReAct (Yao et al.) | ICLR 2023 (confirmed) | single-agent think→act→observe loop (agent↔tool) | Single Tutor (execution substrate) |
| Reflexion (Shinn et al.) | NeurIPS 2023 (confirmed) | act→evaluate→verbal-reflect→retry self-loop | Fixed Reflection (B1) |
| CAMEL (Li et al.) | NeurIPS 2023 (confirmed) | fixed two-role inception-prompted dialogue | Fixed multi-agent workflow (B2) |
| Generative Agents (Park et al.) | UIST 2023 (confirmed; ACM DOI 10.1145/3586183.3606763) | memory-stream → retrieve → reflect → plan | "Ours" State Manager component |
| Exchange-of-Thought (Yin et al.) | EMNLP 2023 Main (confirmed; 2023.emnlp-main.936) | fixed topology cross-model CoT exchange (Memory/Report/Relay/Debate) | Multi-agent Debate (B3) |
| SWE-agent (Yang et al.) | NeurIPS 2024 poster (confirmed; OpenReview mXpq6ut8J3) | single-agent via agent-computer interface | Single Tutor (N=1; runtime/interface matters) |
| MALLM (Becker et al.) | **EMNLP 2025 System Demonstrations** (confirmed; 2025.emnlp-demos.29) — NOT "ACL 2025" | configurable multi-agent debate (persona/generator/paradigm/decision protocol) | Multi-agent Debate (B3) |

Already-downloaded CCF-A anchors from the earlier batch: du2023 Multiagent Debate
(ICML 2024 → Debate/B3), wu2023 AutoGen (COLM 2024 → Fixed multi-agent workflow/B2),
li2024 Sparse Debate Topology (EMNLP 2024 Findings → Generic Sparse).

## Metric / evaluation papers (earlier batch)

macina2023 MathDial (EMNLP'23 Findings — Success@k / Telling@k), wang2023 Bridge
(NAACL'24 — No/Expert/Self/Random decision ablation), daheim2024 Stepwise Verification
(EMNLP'24 — Targeted/Correct/Actionable + verification-correctness conditional),
liu2023 G-Eval (EMNLP'23), zheng2023 MT-Bench LLM-judge (NeurIPS'23), wang2023 Unfair
Evaluator (ACL'24 — position bias / Conflict Rate), cemri2025 MAST (failure taxonomy),
liu2023 AgentBench (ICLR'24 — normalized composite + finish-reason diagnostics).

## Venue verification flags (recheck BibTeX before submission)

- **MALLM** is EMNLP 2025 **System Demonstrations** (demo track), NOT ACL 2025 — phrase as a framework/system, not a research finding (arXiv 2509.11656v3).
- **SWE-agent** is NeurIPS 2024 (poster) — verify camera-ready via OpenReview mXpq6ut8J3.
- ReAct/Reflexion/CAMEL/SWE-agent have no ACL-style page numbers — prefer official proceedings / OpenReview BibTeX over arXiv auto-entries. EoT and Generative Agents have authoritative Anthology/ACM entries.
