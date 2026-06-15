# Task Card 02: 环境检查与依赖安装

## 1. 任务目标

建立可复现的 Python/conda 环境检查流程，确认服务器 GPU、CUDA、vLLM、ModelScope、Transformers、LangGraph 和评估依赖满足 PRISM-Tutor 实验要求。

## 2. 背景与设计约束

方案要求在服务器上使用 conda 虚拟环境，不能直接使用 base 环境；Qwen3 需要 `transformers>=4.51.0`；主实验使用 2 张 RTX 4090，优先检测 2 张可用 GPU，并保留物理 GPU 编号配置，例如服务器指定的 2、3 号卡。

## 3. 前置依赖

- [ ] 依赖 Task Card 01 的配置和目录规范。
- [ ] 依赖可登录目标服务器或本地开发环境。
- [ ] 依赖 `configs/default.yaml` 中的 `seed`、`cuda_devices` 和依赖版本要求。
- [ ] 不依赖已下载模型或数据集。

## 4. 需要新增或修改的文件

```text
environment.yml
requirements.txt
scripts/00_prepare_env_check.py
prism_tutor/utils/env.py
outputs/logs/env_check.json
```

## 5. 具体执行步骤

- [ ] Step 1: 编写 `environment.yml`，指定 `python=3.11` 和环境名 `prism_tutor`。
- [ ] Step 2: 编写 `requirements.txt`，包含 `vllm`、`modelscope`、`transformers>=4.51.0`、`langgraph`、`openai`、`pydantic`、`pandas`、`numpy`、`scipy`、`scikit-learn`、`jsonlines`、`pyyaml`、`matplotlib`、`rouge-score`。
- [ ] Step 3: 扩展 `scripts/00_prepare_env_check.py`，检查当前是否在 conda env，且不是 `base`。
- [ ] Step 4: 检查 CUDA、`nvidia-smi`、可见 GPU 数、GPU 名称和显存；要求至少两张 24GB 级 GPU，记录实际编号。
- [ ] Step 5: 检查核心包可 import，并保存版本号。
- [ ] Step 6: 输出 `outputs/logs/env_check.json`，包含 success、warnings、errors、package_versions、gpu_summary。

## 6. 边界情况与失败处理

- [ ] 如果只检测到 1 张 GPU，标记 `status=degraded`，允许后续小规模 smoke，但禁止 full experiment。
- [ ] 如果 GPU 不是 RTX 4090，但显存满足要求，记录 warning，由执行者确认是否继续。
- [ ] 如果 `transformers<4.51.0`，直接失败，避免 Qwen3 tokenizer/config 不兼容。
- [ ] 如果 `vllm` import 失败，输出安装建议，但不自动重新安装。
- [ ] 如果 `CUDA_VISIBLE_DEVICES` 与配置不一致，记录冲突并要求显式修正。

## 7. 验收标准

- [ ] `conda env create -f environment.yml` 或等价安装文档可执行。
- [ ] `python scripts/00_prepare_env_check.py` 生成 `outputs/logs/env_check.json`。
- [ ] 报告包含 2 张 GPU 检测结果、CUDA 版本、Python 版本和关键包版本。
- [ ] 所有失败项都有清晰错误消息和 fallback 建议。
- [ ] 没有下载模型、数据集或启动实验。

## 8. 不允许做的事情

- [ ] 不允许使用 base conda 环境作为正式实验环境。
- [ ] 不允许修改服务器系统级 CUDA 或驱动。
- [ ] 不允许执行训练、下载模型或下载数据。
- [ ] 不允许把 API key 写入 `environment.yml`、`requirements.txt` 或日志。

## 9. 完成后产物

```text
environment.yml
requirements.txt
outputs/logs/env_check.json
```
