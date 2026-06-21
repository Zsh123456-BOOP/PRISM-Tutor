from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


MethodCallable = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class MethodSpec:
    name: str
    family: str
    description: str
    selected_agents: tuple[str, ...]
    rounds: int = 1
    variant: dict[str, Any] = field(default_factory=dict)
    callable: MethodCallable | None = None

    def run(self, sample: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        if self.callable is not None:
            return self.callable(sample, context)
        prompt = str(sample.get("problem") or sample.get("question") or sample.get("text") or "")
        final_response = (
            f"[dry-run:{self.name}] dataset={sample.get('dataset')} split={sample.get('split')} "
            f"sample_id={sample.get('sample_id')} prompt_chars={len(prompt)}"
        )
        return {
            "selected_agents": list(self.selected_agents),
            "rounds": self.rounds,
            "risk_scores": {},
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": final_response},
            ],
            "state": {"dry_run": True, "method_variant": self.variant},
            "raw_completion": final_response,
            "final_response": final_response,
            "parse_success": True,
            "errors": [],
        }


class MethodRegistry:
    def __init__(self) -> None:
        self._methods: dict[str, MethodSpec] = {}

    def register(self, spec: MethodSpec) -> None:
        if spec.name in self._methods:
            raise ValueError(f"Duplicate method registration: {spec.name}")
        self._methods[spec.name] = spec

    def get(self, name: str) -> MethodSpec:
        try:
            return self._methods[name]
        except KeyError as exc:
            raise KeyError(f"Unknown method: {name}") from exc

    def names(self) -> list[str]:
        return sorted(self._methods)

    def resolve(self, names: list[str]) -> list[MethodSpec]:
        return [self.get(name) for name in names]


def default_method_registry() -> MethodRegistry:
    registry = MethodRegistry()
    specs = [
        MethodSpec("single_tutor", "B0", "Single tutor baseline", ("final_tutor",)),
        MethodSpec("fixed_2", "B1", "Tutor critic reflection", ("tutor", "critic", "final_tutor"), 2),
        MethodSpec(
            "fixed_4",
            "B2",
            "Fixed full communication",
            ("solver", "misconception", "pedagogy", "verifier", "final_tutor"),
            1,
        ),
        MethodSpec("debate", "B3", "Multi-agent debate", ("solver_a", "solver_b", "solver_c", "judge", "final_tutor"), 2),
        MethodSpec("generic_sparse", "B4", "Generic sparse MAS", ("generic_router", "selected_agents", "final_tutor")),
        MethodSpec("difficulty_routing", "B5", "Difficulty-only routing", ("difficulty_router", "final_tutor")),
        MethodSpec("oracle_routing", "oracle", "Upper-bound oracle routing", ("oracle_router", "final_tutor")),
        MethodSpec("random_routing", "routing", "Random routing baseline", ("random_router", "final_tutor")),
        MethodSpec("fixed_all_agents", "routing", "Always call all agents", ("solver", "misconception", "pedagogy", "verifier", "state", "final_tutor")),
        MethodSpec("ours_routing", "M1", "PRISM routing only", ("risk_estimator", "qos_router", "final_tutor")),
        MethodSpec("ours_routing_budget", "M2", "PRISM routing and budget", ("risk_estimator", "qos_router", "budget_controller", "final_tutor"), 2),
        MethodSpec("ours_full", "M3", "Full PRISM-Tutor runtime", ("risk_estimator", "qos_router", "budget_controller", "state_commit", "final_tutor"), 3),
        MethodSpec("single_round", "budget", "Single deliberation round", ("final_tutor",), 1),
        MethodSpec("fixed_2_rounds", "budget", "Fixed two rounds", ("tutor", "verifier", "final_tutor"), 2),
        MethodSpec("fixed_3_rounds", "budget", "Fixed three rounds", ("tutor", "verifier", "pedagogy", "final_tutor"), 3),
        MethodSpec("fixed_4_rounds", "budget", "Fixed four rounds", ("tutor", "verifier", "pedagogy", "misconception", "final_tutor"), 4),
        MethodSpec("generic_early_stopping", "budget", "Generic early stopping", ("generic_controller", "final_tutor"), 2),
        MethodSpec("no_memory", "state", "No state memory", ("final_tutor",)),
        MethodSpec("naive_shared_memory", "state", "Naive shared memory", ("shared_memory", "final_tutor")),
        MethodSpec("single_writer", "state", "Single-writer memory", ("single_writer", "final_tutor")),
        MethodSpec("two_phase_commit", "state", "Two-phase state commit", ("state_proposer", "verifier", "state_commit", "final_tutor")),
    ]
    for spec in specs:
        registry.register(spec)

    ablations = [
        "ablate_risk_estimator",
        "ablate_qos_routing",
        "ablate_budget_controller",
        "ablate_leakage_risk",
        "ablate_misconception_risk",
        "ablate_state_conflict_risk",
        "ablate_state_commit",
        "ablate_leakage_guard",
        "ablate_confidence_weighted_commit",
        "replace_pedagogical_risk_with_difficulty",
        "replace_two_phase_commit_with_naive_memory",
    ]
    for name in ablations:
        registry.register(
            MethodSpec(
                name=name,
                family="ablation",
                description=f"Full PRISM with {name.replace('_', ' ')}",
                selected_agents=("risk_estimator", "qos_router", "budget_controller", "state_commit", "final_tutor"),
                rounds=3,
                variant={"ablation": name},
            )
        )
    return registry
