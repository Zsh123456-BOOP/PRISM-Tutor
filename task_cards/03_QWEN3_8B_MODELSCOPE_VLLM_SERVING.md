# Task Card 03: Qwen3-8B ModelScope 下载与 vLLM 服务配置

## 1. 任务目标

配置 Qwen/Qwen3-8B 的 ModelScope 下载与 vLLM OpenAI-compatible 服务，优先使用两张 RTX 4090 的单卡双副本 serving，并提供双卡 tensor parallel fallback。

## 2. 背景与设计约束

主实验生成模型固定为 Qwen3-8B full BF16/FP16，不使用 AWQ、FP8 或 4-bit 量化作为主结果。所有方法必须共享同一 generation config，并设置 `enable_thinking=false`。如果输出仍包含 `<think>...</think>`，后处理要剥离，但 token usage 必须保留原始统计。

## 3. 前置依赖

- [ ] 依赖 Task Card 02 环境检查通过。
- [ ] 依赖可访问 ModelScope 或已有模型缓存。
- [ ] 依赖 `configs/model.yaml` 中声明 GPU 编号、端口、dtype 和 max model length。
- [ ] 不依赖数据集或实验 runner。

## 4. 需要新增或修改的文件

```text
serving/start_vllm_gpu0.sh
serving/start_vllm_gpu1.sh
serving/start_vllm_tp2.sh
serving/health_check.py
configs/model.yaml
prism_tutor/serving/endpoints.py
outputs/logs/model_health_check.json
```

## 5. 具体执行步骤

- [ ] Step 1: 在启动脚本中设置 `VLLM_USE_MODELSCOPE=true`，并从配置读取实际 GPU 编号。
- [ ] Step 2: 编写双副本脚本，分别启动两个 vLLM server，例如端口 8000 和 8001，dtype 使用 BF16，`--max-model-len 8192`。
- [ ] Step 3: 编写 tensor parallel fallback 脚本，使用 `--tensor-parallel-size 2`，仅在单卡 OOM 或长上下文需要时启用。
- [ ] Step 4: 实现 endpoint registry 与 round-robin 分发，确保 sample 级别可复现分配。
- [ ] Step 5: 实现 health check，测试两个 endpoint 返回、JSON 输出、token usage、latency 和 `enable_thinking=false`。
- [ ] Step 6: 实现 `<think>...</think>` 剥离工具，保存 raw output、stripped output 和原始 token usage。

## 6. 边界情况与失败处理

- [ ] OOM：降低 batch/concurrency 或切换 TP2；记录 OOM 日志和 GPU 显存状态。
- [ ] Timeout：按配置 retry 有限次数，超过后记录 endpoint、sample_id、timeout_seconds。
- [ ] Endpoint unhealthy：临时从 round-robin 池移除，所有移除行为写入日志。
- [ ] JSON parse error：交给 Task Card 06 的 retry/repair 流程，不在 serving 层丢弃样本。
- [ ] ModelScope 不可达：允许使用预先缓存路径，但必须写入 manifest。

## 7. 验收标准

- [ ] `bash serving/start_vllm_gpu0.sh` 和 `bash serving/start_vllm_gpu1.sh` 可按配置启动。
- [ ] `python serving/health_check.py` 生成 `outputs/logs/model_health_check.json`。
- [ ] health check 包含两个 endpoint 的状态、实际 served model name、latency 和 token usage。
- [ ] 主配置明确禁用量化，并记录 `enable_thinking=false`。
- [ ] fallback TP2 脚本存在，但不会默认替代双副本主方案。

## 8. 不允许做的事情

- [ ] 不允许把 AWQ、FP8 或 4-bit 量化作为主实验配置。
- [ ] 不允许混用其他 generator 模型。
- [ ] 不允许在日志中丢弃 raw completion 或 token usage。
- [ ] 不允许把 health check 失败样本静默忽略。

## 9. 完成后产物

```text
serving/start_vllm_gpu0.sh
serving/start_vllm_gpu1.sh
serving/start_vllm_tp2.sh
serving/health_check.py
outputs/logs/model_health_check.json
```
