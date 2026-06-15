from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prism_tutor.serving.endpoints import EndpointRegistry, strip_think_blocks
from prism_tutor.utils.config import load_config


HEALTH_PROMPT = "Return a compact JSON object with key status and value ok."


def _generation_config(config: dict[str, Any]) -> dict[str, Any]:
    generation = config.get("generation", {})
    model = config.get("model", {})
    return {
        "temperature": generation.get("temperature", 0.2),
        "top_p": generation.get("top_p", 0.8),
        "max_tokens": generation.get("max_tokens", {}).get("verifier", 128),
        "extra_body": {"enable_thinking": bool(model.get("enable_thinking", False))},
    }


def _dry_run_endpoint(endpoint: Any, config: dict[str, Any]) -> dict[str, Any]:
    raw_completion = '<think>dry-run internal scratchpad</think>{"status":"ok"}'
    return {
        "endpoint": endpoint.as_dict(),
        "ok": True,
        "mode": "dry-run",
        "served_model": endpoint.model,
        "latency_seconds": 0.0,
        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "raw_completion": raw_completion,
        "stripped_completion": strip_think_blocks(raw_completion),
        "request": {
            "model": endpoint.model,
            "messages": [{"role": "user", "content": HEALTH_PROMPT}],
            **_generation_config(config),
        },
    }


def _live_endpoint(endpoint: Any, config: dict[str, Any]) -> dict[str, Any]:
    request_body = {
        "model": endpoint.model,
        "messages": [{"role": "user", "content": HEALTH_PROMPT}],
        **_generation_config(config),
    }
    payload = json.dumps(request_body).encode("utf-8")
    started = time.perf_counter()
    try:
        request = urllib.request.Request(
            f"{endpoint.base_url}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=endpoint.timeout_seconds) as response:
            response_data = json.loads(response.read().decode("utf-8"))
        latency = time.perf_counter() - started
        raw_completion = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return {
            "endpoint": endpoint.as_dict(),
            "ok": True,
            "mode": "live",
            "served_model": response_data.get("model", endpoint.model),
            "latency_seconds": latency,
            "token_usage": response_data.get("usage", {}),
            "raw_completion": raw_completion,
            "stripped_completion": strip_think_blocks(raw_completion),
            "response_id": response_data.get("id"),
        }
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
        return {
            "endpoint": endpoint.as_dict(),
            "ok": False,
            "mode": "live",
            "latency_seconds": time.perf_counter() - started,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }


def run_health_check(config: dict[str, Any], *, live: bool = False) -> dict[str, Any]:
    registry = EndpointRegistry.from_config(config)
    checker = _live_endpoint if live else _dry_run_endpoint
    results = [checker(endpoint, config) for endpoint in registry.endpoints]
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "live" if live else "dry-run",
        "enable_thinking": bool(config.get("model", {}).get("enable_thinking", False)),
        "all_ok": all(item.get("ok") for item in results),
        "endpoints": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Health check configured Qwen3 OpenAI-compatible endpoints.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="outputs/logs/model_health_check.json")
    parser.add_argument("--live", action="store_true", help="Call endpoints. Default is dry-run only.")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    report = run_health_check(config, live=args.live)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(str(output))
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
