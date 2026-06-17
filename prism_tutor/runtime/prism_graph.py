"""Callable PRISM-Tutor graph skeleton for M1/M2/M3."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from prism_tutor.agents.base_client import BaseLLMClient
from prism_tutor.agents.final_tutor import FinalTutorAgent
from prism_tutor.agents.hint import HintAgent
from prism_tutor.agents.misconception import MisconceptionAgent
from prism_tutor.agents.pedagogy import PedagogyAgent
from prism_tutor.agents.solver import SolverAgent
from prism_tutor.agents.state_manager import StateManagerAgent
from prism_tutor.agents.verifier import VerifierAgent
from prism_tutor.eval.leakage_detector import detect_leakage

from .budget_controller import BudgetConfig, BudgetController
from .graph_state import TutorGraphState
from .qos_router import QoSRouter, RouterConfig
from .risk_estimator import RiskConfig, estimate_risk
from .state_commit import CommitConfig, StateCommitter


AGENT_REGISTRY = {
    "solver": SolverAgent(),
    "misconception": MisconceptionAgent(),
    "pedagogy": PedagogyAgent(),
    "hint": HintAgent(),
    "verifier": VerifierAgent(),
    "state_manager": StateManagerAgent(),
    "final_tutor": FinalTutorAgent(),
}


class PrismGraphConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk: RiskConfig = Field(default_factory=RiskConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    commit: CommitConfig = Field(default_factory=CommitConfig)
    disabled_modules: list[str] = Field(default_factory=list)
    disabled_risks: list[str] = Field(default_factory=list)
    force_difficulty_only_risk: bool = False
    naive_memory_commit: bool = False
    variant: dict[str, Any] = Field(default_factory=dict)
    noisy_agent_probability: float = Field(default=0.0, ge=0, le=1)
    noisy_agent_seed: int = 42
    leakage_guard_max_retries: int = Field(default=1, ge=0, le=2)
    noisy_agent_names: list[str] = Field(
        default_factory=lambda: ["solver", "misconception", "pedagogy", "hint", "verifier", "state_manager"]
    )


class PrismGraph:
    """Graph-like callable implementing M1, M2, and M3 behavior.

    M1: routing only.
    M2: routing plus budgeted deliberation.
    M3: routing, budgeted deliberation, and state commit.
    """

    def __init__(
        self,
        *,
        method: str = "M3",
        client: BaseLLMClient | None = None,
        config: PrismGraphConfig | None = None,
    ) -> None:
        if method not in {"M1", "M2", "M3"}:
            raise ValueError("method must be one of M1, M2, M3")
        self.method = method
        self.client = client or BaseLLMClient()
        self.config = config or PrismGraphConfig()
        self.router = QoSRouter(self.config.router)
        self.budget = BudgetController(self.config.budget)
        self.committer = StateCommitter(self.config.commit)

    def invoke(self, state: TutorGraphState | dict[str, Any]) -> TutorGraphState:
        graph_state = state if isinstance(state, TutorGraphState) else TutorGraphState.model_validate(state)
        graph_state.method = self.method
        graph_state.agent_outputs.setdefault("runtime_variant", []).append(self.config.variant)
        if self._module_disabled("risk_estimator"):
            risk = self._disabled_risk(graph_state)
        else:
            risk = estimate_risk(graph_state, self.config.risk)
            risk = self._apply_risk_controls(risk)
        graph_state.risk_scores.append(risk.model_dump(mode="json"))
        if self._module_disabled("qos_routing"):
            selected = list(self.config.router.high_agents)
        else:
            selected = self.router.select_agents(risk)
        if self.method == "M3" and not self._module_disabled("state_commit"):
            selected = self._ensure_state_manager_for_commit(selected)
        defer_final_tutor = self._should_defer_final_tutor()
        needs_final_tutor = "final_tutor" in selected
        if defer_final_tutor:
            selected = self._without_final_tutor(selected)
        graph_state.selected_agents.extend(selected)
        self._run_agents(graph_state, selected)

        if self.method in {"M2", "M3"} and not self._module_disabled("budget_controller"):
            while self.budget.should_continue(graph_state):
                next_agents = self.budget.next_agents(graph_state)
                if not next_agents:
                    break
                if defer_final_tutor:
                    needs_final_tutor = needs_final_tutor or "final_tutor" in next_agents
                    next_agents = self._without_final_tutor(next_agents)
                    if not next_agents:
                        break
                graph_state.rounds += 1
                graph_state.selected_agents.extend(next_agents)
                self._run_agents(graph_state, next_agents)

        if self.method == "M3" and not self._module_disabled("state_commit"):
            if self.config.naive_memory_commit:
                decision = self.committer.commit_naive(graph_state)
            else:
                decision = self.committer.commit(graph_state)
            graph_state.agent_outputs.setdefault("state_commit", []).append(decision.model_dump(mode="json"))

        if defer_final_tutor and needs_final_tutor:
            self._run_final_tutor_once(graph_state)

        if graph_state.termination_reason is None:
            graph_state.termination_reason = "completed"
        return graph_state

    __call__ = invoke

    def _run_agents(self, state: TutorGraphState, agents: list[str]) -> None:
        for agent_name in agents:
            agent = AGENT_REGISTRY.get(agent_name)
            if agent is None:
                continue
            record = agent.invoke(
                sample=state.sample,
                state=self._agent_context(state),
                client=self.client,
                method=self.method,
            )
            self._maybe_inject_noisy_output(state, record)
            state.add_call(record)

    @staticmethod
    def _agent_context(state: TutorGraphState) -> dict[str, Any]:
        return {
            "method": state.method,
            "rounds": state.rounds,
            "student_state": state.student_state.model_dump(mode="json"),
            "agent_outputs": state.agent_outputs,
            "risk_scores": state.risk_scores,
            "selected_agents": state.selected_agents,
            "total_tokens": state.total_tokens,
        }

    def _module_disabled(self, module: str) -> bool:
        return module in set(self.config.disabled_modules)

    def _should_defer_final_tutor(self) -> bool:
        return self.method == "M2" or (self.method == "M3" and not self._module_disabled("state_commit"))

    @staticmethod
    def _without_final_tutor(selected: list[str]) -> list[str]:
        return [agent for agent in selected if agent != "final_tutor"]

    def _run_final_tutor_once(self, state: TutorGraphState) -> None:
        if any(call.get("agent_name") == "final_tutor" for call in state.llm_calls):
            return
        state.selected_agents.append("final_tutor")
        self._run_agents(state, ["final_tutor"])
        self._retry_final_tutor_on_leakage(state)

    def _retry_final_tutor_on_leakage(self, state: TutorGraphState) -> None:
        if self.config.leakage_guard_max_retries <= 0:
            return
        for retry_index in range(self.config.leakage_guard_max_retries):
            response = self._latest_final_response(state)
            leakage = detect_leakage(response, gold=self._leakage_gold(state.sample), sample_id=state.sample.get("sample_id"))
            if not leakage["rule_leakage"]:
                return
            state.agent_outputs.setdefault("leakage_guard", []).append(
                {
                    "retry_index": retry_index,
                    "matched_rules": leakage["matched_rules"],
                    "instruction": "Regenerate the final response without answer leakage or key solution steps.",
                }
            )
            state.selected_agents.append("final_tutor")
            self._run_agents(state, ["final_tutor"])

    @staticmethod
    def _latest_final_response(state: TutorGraphState) -> str:
        for call in reversed(state.llm_calls):
            if call.get("agent_name") != "final_tutor":
                continue
            parsed = call.get("parsed_output") if isinstance(call.get("parsed_output"), dict) else {}
            if parsed.get("response"):
                return str(parsed["response"])
            return str(call.get("stripped_output") or call.get("raw_completion") or "")
        return ""

    @staticmethod
    def _leakage_gold(sample: dict[str, Any]) -> dict[str, Any]:
        metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
        return {
            "answer": sample.get("answer") or sample.get("gold_answer") or sample.get("correct_answer"),
            "ground_truth": sample.get("ground_truth") or metadata.get("ground_truth") or metadata.get("correct_answer"),
            "final_answer": sample.get("final_answer") or metadata.get("final_answer"),
        }

    @staticmethod
    def _ensure_state_manager_for_commit(selected: list[str]) -> list[str]:
        if "state_manager" in selected:
            return selected
        ordered = list(selected)
        try:
            final_index = ordered.index("final_tutor")
        except ValueError:
            ordered.append("state_manager")
        else:
            ordered.insert(final_index, "state_manager")
        return ordered

    def _disabled_risk(self, state: TutorGraphState) -> Any:
        difficulty = estimate_risk(state, self.config.risk).estimated_difficulty
        return self._risk_from_values(
            {
                "answer_uncertainty": 0.0,
                "misconception_risk": 0.0,
                "pedagogy_risk": 0.0,
                "leakage_risk": 0.0,
                "state_conflict_risk": 0.0,
                "estimated_difficulty": difficulty,
            },
            weights={key: 1.0 for key in self.config.risk.weights},
        )

    def _apply_risk_controls(self, risk: Any) -> Any:
        values = {
            "answer_uncertainty": risk.answer_uncertainty,
            "misconception_risk": risk.misconception_risk,
            "pedagogy_risk": risk.pedagogy_risk,
            "leakage_risk": risk.leakage_risk,
            "state_conflict_risk": risk.state_conflict_risk,
            "estimated_difficulty": risk.estimated_difficulty,
        }
        weights = dict(self.config.risk.weights)
        if self.config.force_difficulty_only_risk:
            values = {key: (risk.estimated_difficulty if key == "estimated_difficulty" else 0.0) for key in values}
            weights = {key: (1.0 if key == "estimated_difficulty" else 0.0) for key in values}
        for risk_name in self.config.disabled_risks:
            if risk_name in values:
                values[risk_name] = 0.0
        return self._risk_from_values(values, weights=weights)

    def _risk_from_values(self, values: dict[str, float], *, weights: dict[str, float]) -> Any:
        weight_sum = sum(max(0.0, weights.get(key, 0.0)) for key in values) or 1.0
        total = sum(values[key] * max(0.0, weights.get(key, 0.0)) for key in values) / weight_sum
        if total >= self.config.risk.high_threshold:
            bucket = "high"
            mode = "deliberative"
        elif total >= self.config.risk.low_threshold:
            bucket = "medium"
            mode = "guided"
        else:
            bucket = "low"
            mode = "direct"
        from prism_tutor.agents.schemas import RiskEstimatorOutput

        return RiskEstimatorOutput(**values, total_risk=total, risk_bucket=bucket, recommended_mode=mode)

    def _maybe_inject_noisy_output(self, state: TutorGraphState, record: Any) -> None:
        probability = self.config.noisy_agent_probability
        if probability <= 0 or record.agent_name not in set(self.config.noisy_agent_names):
            return
        seed_material = {
            "seed": self.config.noisy_agent_seed,
            "sample_id": state.sample.get("sample_id") or state.sample.get("id"),
            "method": state.method,
            "agent": record.agent_name,
            "round": state.rounds,
        }
        digest = hashlib.sha256(json.dumps(seed_material, sort_keys=True).encode("utf-8")).hexdigest()
        if random.Random(int(digest[:16], 16)).random() >= probability:
            return
        noisy_output = self._noisy_output(record.agent_name)
        record.parsed_output = noisy_output
        record.stripped_output = json.dumps(noisy_output, ensure_ascii=False, sort_keys=True)
        record.request_payload["robustness_noise"] = {
            "probability": probability,
            "seed": self.config.noisy_agent_seed,
            "agent": record.agent_name,
        }
        record.warnings.append("noisy_agent_injected")

    @staticmethod
    def _noisy_output(agent_name: str) -> dict[str, Any]:
        if agent_name == "solver":
            return {
                "schema_version": "0.1.0",
                "answer": "NOISY_AGENT_INJECTED",
                "reasoning": ["Injected incorrect collaborator output for robustness evaluation."],
                "confidence": 0.95,
                "uncertainty": 0.05,
                "needs_more_info": False,
            }
        if agent_name == "misconception":
            return {
                "schema_version": "0.1.0",
                "misconception_detected": True,
                "misconception_labels": ["noisy_injected_misconception"],
                "evidence": ["Injected noisy misconception signal."],
                "severity": "high",
                "confidence": 0.95,
            }
        if agent_name == "pedagogy":
            return {
                "schema_version": "0.1.0",
                "strategy": "worked_example",
                "rationale": "Injected noisy pedagogy signal.",
                "target_skills": ["noisy_injected_skill"],
                "confidence": 0.95,
            }
        if agent_name == "hint":
            return {
                "schema_version": "0.1.0",
                "hint_text": "NOISY_AGENT_INJECTED: jump directly to the final answer.",
                "hint_level": 3,
                "answer_leakage_risk": 0.95,
                "confidence": 0.95,
            }
        if agent_name == "verifier":
            return {
                "schema_version": "0.1.0",
                "approved": False,
                "issues": [
                    {
                        "issue_type": "other",
                        "severity": "high",
                        "message": "Injected noisy verifier objection.",
                        "recommended_agent": "pedagogy",
                    }
                ],
                "leakage_detected": True,
                "state_conflict_detected": True,
                "confidence": 0.95,
            }
        if agent_name == "state_manager":
            return {
                "schema_version": "0.1.0",
                "proposed_updates": [
                    {
                        "field": "weak_skills",
                        "operation": "add",
                        "value": "noisy_injected_skill",
                        "confidence": 0.95,
                        "evidence": "Injected noisy state update.",
                    }
                ],
                "conflicts": ["noisy_injected_conflict"],
                "confidence": 0.95,
            }
        return {"schema_version": "0.1.0", "ok": False, "noise": True}


def build_prism_graph(
    method: str = "M3",
    *,
    client: BaseLLMClient | None = None,
    config: PrismGraphConfig | None = None,
) -> PrismGraph:
    return PrismGraph(method=method, client=client, config=config)
