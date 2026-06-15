from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "00_prepare_env_check.py"
SPEC = importlib.util.spec_from_file_location("prepare_env_check_script", SCRIPT_PATH)
env_check = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(env_check)


def test_env_check_reports_cuda_conflict_and_install_suggestions(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "default.yaml"
    output = tmp_path / "env_check.json"
    config.write_text(
        """
seed: 42
cuda:
  expected_gpu_count: 2
  preferred_devices: "2,3"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(env_check, "package_versions", lambda: {"transformers": "4.51.0"})
    monkeypatch.setattr(env_check, "run_nvidia_smi", lambda: {"available": False, "error": "missing", "cuda_version": None, "gpus": []})
    monkeypatch.setattr(
        env_check,
        "check_imports",
        lambda names: [{"module": name, "ok": name != "vllm", "error": None if name != "vllm" else "missing"} for name in names],
    )

    rc = env_check.main(["--config", str(config), "--output", str(output), "--dry-run"])

    report = json.loads(output.read_text(encoding="utf-8"))
    assert rc == 0
    assert report["status"] == "degraded"
    assert report["checks"]["cuda_visible_devices"]["ok"] is False
    assert "CUDA_VISIBLE_DEVICES='0,1'; expected '2,3'" in report["warnings"]
    assert report["checks"]["gpu_count"]["expected"] == 2
    assert any(item["module"] == "vllm" and "Install vLLM" in item["suggestion"] for item in report["fallback_suggestions"])
    assert report["gpu_summary"]["cuda_version"] is None
