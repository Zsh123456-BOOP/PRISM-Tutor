# Task Card Completion Status

## 当前状态

本仓库已经具备 Task Card 01-16 的可运行代码骨架与 mock/dry-run 端到端链路。当前本机验证没有下载数据、没有下载模型、没有启动 vLLM、没有调用外部 judge API，也没有运行真实实验。

## 已完成的代码路径

- [x] 01 项目初始化与复现规范：`configs/`、`environment.yml`、`requirements.txt`、`pyproject.toml`、复现 metadata、`.gitignore`。
- [x] 02 环境检查：`scripts/00_prepare_env_check.py` 可生成 env check report。
- [x] 03 Qwen3-8B vLLM 服务配置：`serving/start_vllm_*.sh`、`serving/health_check.py` dry-run 可用。
- [x] 04 数据 schema 与 split：`scripts/01_build_datasets.py`、`prism_tutor/data/` 支持本地 raw JSON/JSONL/CSV。
- [x] 05 Agent schema 与 base client：`prism_tutor/agents/schemas.py`、mock-safe OpenAI-compatible client。
- [x] 06 Agent prompt 与 JSON repair：prompt、`<think>` 剥离、retry/repair parser。
- [x] 07 Runtime state：`prism_tutor/runtime/` graph state、checkpoint、node interface。
- [x] 08 Baseline 方法：method registry 覆盖 B0-B5 与实验变体。
- [x] 09 PRISM 模块：risk estimator、QoS router、budget controller、state commit、M1/M2/M3 graph。
- [x] 10 Runner 与日志：`scripts/02_run_generation.py` 生成 JSONL raw logs 与 manifest。
- [x] 11 自动指标：`scripts/04_compute_metrics.py` 生成 record/aggregate metrics 和 coverage report。
- [x] 12 LLM judge：`scripts/03_run_judge.py` 默认 mock，真实 DeepSeek 需显式环境变量。
- [x] 13 实验矩阵：`configs/experiments.yaml` 与 `scripts/run_exp*.sh`。
- [x] 14 统计、表格和图：`scripts/05_make_tables.py`、`scripts/06_make_figures.py`。
- [x] 15 Human audit：`scripts/07_sample_human_audit.py`、`scripts/08_human_agreement.py`。
- [x] 16 Paper artifacts：`scripts/09_export_paper_artifacts.py`。

## 已验证命令

- [x] `python -m compileall prism_tutor scripts data serving tests`
- [x] `python -m pytest -q`，结果：36 passed。
- [x] `python scripts/00_prepare_env_check.py --config configs/default.yaml --dry-run`
- [x] `python scripts/01_build_datasets.py --help`
- [x] `bash serving/start_vllm_gpu0.sh`
- [x] `python serving/health_check.py --output outputs/logs/model_health_check.json`
- [x] mock generation -> judge -> metrics -> tables -> figures -> human audit -> paper export 链路。

## 仍需服务器真实执行的项目

- [ ] 在 `10.154.22.11` 的 `zsh` 账户中创建 conda 环境。
- [ ] 在 2、3 号 GPU 上启动 Qwen3-8B full BF16/FP16 vLLM。
- [ ] 将真实 MathDial、Bridge、Misconception raw 数据放入 `data/raw/` 并运行 dataset build。
- [ ] 使用真实 Qwen3-8B endpoint 跑 Exp0-Exp6 generation。
- [ ] 使用真实 DeepSeek judge API 跑 judge，并保存 actual model id、日期与 raw response。
- [ ] 全部自动实验完成后执行 200 条 blind human audit。
- [ ] 用真实 raw logs 重新生成正式论文 tables、figures、paper artifacts。

## 安全边界

- [x] 未把 `指导方案.md` 中的明文 key 复制到代码、配置或任务卡。
- [x] `outputs/` 中除 `outputs/README.md` 外的 runtime 产物被 `.gitignore` 忽略。
- [x] judge 默认 dry-run；真实 API 调用必须同时设置 `DEEPSEEK_API_KEY` 和 `PRISM_TUTOR_ENABLE_REAL_JUDGE=1`。
