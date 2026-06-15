# Task Card 10: 生成实验 Runner 与日志规范

## 1. 任务目标

实现统一 generation runner，按方法、数据集和 split 执行 baseline 与 PRISM-Tutor，并保存可复现、可聚合、可失败追踪的 JSONL raw logs。

## 2. 背景与设计约束

方案要求每个样本、每个方法保存一个 JSONL entry，包含 sample_id、dataset、split、method、base_model、endpoint、generation_config、selected_agents、rounds、risk_scores、messages、state、token usage、latency、final_response、parse_success 和 errors。所有实验必须能从 raw logs 重新生成结果。

## 3. 前置依赖

- [ ] 依赖 Task Card 08 baseline 方法。
- [ ] 依赖 Task Card 09 PRISM-Tutor 方法。
- [ ] 依赖 Task Card 04 数据 split。
- [ ] 依赖 Task Card 03 Qwen3 endpoint 可用。

## 4. 需要新增或修改的文件

```text
scripts/02_run_generation.py
prism_tutor/experiments/runner.py
prism_tutor/experiments/method_registry.py
prism_tutor/logging/jsonl_logger.py
prism_tutor/logging/manifest.py
outputs/generations/*.jsonl
outputs/logs/*.jsonl
```

## 5. 具体执行步骤

- [ ] Step 1: 实现 method registry，统一注册 B0-B5、M1-M3 和实验专用变体。
- [ ] Step 2: 实现 CLI 参数：config、methods、datasets、split、experiment、limit、resume、output_dir。
- [ ] Step 3: 每次运行开始时保存 config snapshot、git commit hash、package versions 和 experiment manifest。
- [ ] Step 4: 每个样本执行一个 method graph，记录所有 agent calls、raw prompts、raw completions、parsed outputs、usage、latency。
- [ ] Step 5: 实现 resume 逻辑，按 sample_id+method 去重，失败样本可重跑但保留原失败日志。
- [ ] Step 6: 对 OOM、timeout、JSON parse error、endpoint failure 统一写入 `errors`，不得丢样本。

## 6. 边界情况与失败处理

- [ ] 运行中断：已完成 JSONL 保留，manifest 标记 `status=interrupted`。
- [ ] 单样本失败：输出 failure entry，继续后续样本，除非配置为 fail-fast。
- [ ] endpoint 全部不可用：停止运行并写入 health failure。
- [ ] 日志写入失败：立即终止，避免产生不可复现结果。
- [ ] 重复运行：resume 只能追加缺失样本，不覆盖已有 raw logs。

## 7. 验收标准

- [ ] 可以用 `python scripts/02_run_generation.py --limit 2 --methods single_tutor --datasets mathdial --split test` 生成 smoke JSONL。
- [ ] 每条 JSONL 包含方案规定的必需字段。
- [ ] raw prompt、completion、token usage、latency、parse_success 全部保存。
- [ ] failure entry 也包含 sample_id、method、dataset 和错误原因。
- [ ] experiment manifest 可定位输入数据、配置、模型、参数和输出路径。

## 8. 不允许做的事情

- [ ] 不允许手动编辑 generation raw logs。
- [ ] 不允许失败样本不落盘。
- [ ] 不允许不同方法使用不同样本列表。
- [ ] 不允许把 judge 分数写入 generation log 作为生成结果的一部分。

## 9. 完成后产物

```text
scripts/02_run_generation.py
outputs/generations/*.jsonl
outputs/logs/experiment_manifest_*.json
outputs/logs/generation_errors_*.jsonl
```
