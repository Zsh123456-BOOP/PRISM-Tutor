# Task Card 01: 项目初始化与复现规范

## 1. 任务目标

建立 PRISM-Tutor 的最小项目骨架、全局配置入口和复现记录规范，为后续模型服务、数据处理、运行时、实验和论文 artifact 提供统一约束。

## 2. 背景与设计约束

`指导方案.md` 要求所有实验固定 Qwen3-8B、固定 random seed、保存 config snapshot、git commit hash、Python/package 版本，并确保所有实验能从 raw logs 重新生成结果。本任务只定义工程骨架和复现协议，不实现实验逻辑。

## 3. 前置依赖

- [ ] 无前序 task card。
- [ ] 依赖根目录 `指导方案.md` 作为需求来源。
- [ ] 依赖 Git 仓库可用；如果当前目录尚未初始化 Git，需要在执行前确认是否初始化。
- [ ] 不依赖外部模型、数据集或 API。

## 4. 需要新增或修改的文件

```text
configs/default.yaml
configs/model.yaml
configs/datasets.yaml
configs/judge.yaml
configs/experiments.yaml
prism_tutor/__init__.py
prism_tutor/utils/reproducibility.py
prism_tutor/utils/config.py
scripts/00_prepare_env_check.py
outputs/README.md
.gitignore
README.md
```

## 5. 具体执行步骤

- [ ] Step 1: 读取 `指导方案.md`，抽取全局 seed、模型、generation config、risk thresholds、budget 和输出路径，写入 `configs/default.yaml`。
- [ ] Step 2: 创建 `prism_tutor/`、`configs/`、`scripts/`、`outputs/` 等目录骨架，但不下载数据和模型。
- [ ] Step 3: 实现配置加载工具，要求支持 YAML、环境变量覆盖和 config snapshot 导出。
- [ ] Step 4: 实现复现信息采集函数，记录 git commit hash、dirty status、Python 版本、包版本、CUDA 可见设备和运行时间。
- [ ] Step 5: 定义 experiment manifest schema，字段包括 experiment、dataset、split、method、model、generation_config、input_path、output_path。
- [ ] Step 6: 更新 `.gitignore`，排除 `outputs/`、`data/raw/`、模型权重、日志、缓存、API key、`.env` 和大型 artifacts。

## 6. 边界情况与失败处理

- [ ] 如果 Git 不可用，manifest 中写入 `git_commit=null` 和错误原因，不中断配置生成。
- [ ] 如果配置字段缺失，启动前报出具体 key 路径，不使用隐式默认值覆盖关键实验设置。
- [ ] 如果工作区 dirty，记录 dirty 文件列表，但不得自动 revert 或删除。
- [ ] 如果发现密钥出现在配置或日志中，立即失败并提示迁移到环境变量。

## 7. 验收标准

- [ ] 可以运行 `python scripts/00_prepare_env_check.py --config configs/default.yaml --dry-run`。
- [ ] 生成 config snapshot 示例，包含 seed、模型、generation config、thresholds 和 budget。
- [ ] manifest schema 文档化，后续脚本可复用。
- [ ] `.gitignore` 不会误排除源代码，但会排除 outputs、raw data、模型权重和 secrets。
- [ ] 没有修改 `指导方案.md`。

## 8. 不允许做的事情

- [ ] 不允许下载数据集或模型。
- [ ] 不允许启动 vLLM、调用 DeepSeek 或运行实验。
- [ ] 不允许把明文 API key 写入任何文件。
- [ ] 不允许手动创建结果表或伪造 experiment manifest。

## 9. 完成后产物

```text
configs/*.yaml
prism_tutor/utils/config.py
prism_tutor/utils/reproducibility.py
scripts/00_prepare_env_check.py
outputs/README.md
```
