# Task Card Completion Status

## 当前状态

本仓库已经具备 Task Card 01-16 的可运行代码骨架、mock/dry-run 端到端链路，以及服务器上的 Qwen3-8B vLLM live smoke 链路。MathDial、Bridge 与 MaE Math Misconceptions 三个真实数据集均已下载、传输到服务器并完成 schema build。Exp0-Exp6 已完成真实 Qwen live smoke（limit=1）与 downstream metrics/tables/figures/artifact smoke；真实 DeepSeek judge 已对 live smoke 123 条 generation 完成评分。runner 已支持 sample-level sharding，用于拆分全量正式实验，并已支持 `maintain --target-running N` 与 `supervise` 可恢复补齐后台并发。全量完成后可用 `scripts/12_finalize_full_run.py` gated finalization 生成 metrics/tables/figures/human audit sample/paper artifacts；`scripts/11_plan_or_run_shards.py progress` 可从 supervisor log 输出完成率、近期吞吐、ETA 与 health summary，并可识别 running 超过 supervisor target 的情况。当前全量 Exp0-Exp6 正式实验已经开始运行并由服务器 supervisor 接管，但尚未完成；也尚未执行正式人工标注版 200 条 blind human audit。

## 已完成的代码路径

- [x] 01 项目初始化与复现规范：`configs/`、`environment.yml`、`requirements.txt`、`pyproject.toml`、复现 metadata、`.gitignore`。
- [x] 02 环境检查：`scripts/00_prepare_env_check.py` 可生成 env check report。
- [x] 03 Qwen3-8B vLLM 服务配置：`serving/start_vllm_*.sh`、`serving/health_check.py` dry-run 与服务器 live health check 可用。
- [x] 04 数据 schema 与 split：`scripts/01_build_datasets.py`、`prism_tutor/data/` 支持本地 raw JSON/JSONL/CSV；MathDial、Bridge、MaE Misconception loader 已按真实源格式验证。
- [x] 05 Agent schema 与 base client：`prism_tutor/agents/schemas.py`、mock-safe OpenAI-compatible client，支持 endpoint-specific served model name。
- [x] 06 Agent prompt 与 JSON repair：prompt、`<think>` 剥离、retry/repair parser。
- [x] 07 Runtime state：`prism_tutor/runtime/` graph state、checkpoint、node interface。
- [x] 08 Baseline 方法：method registry 覆盖 B0-B5 与实验变体。
- [x] 09 PRISM 模块：risk estimator、QoS router、budget controller、state commit、M1/M2/M3 graph。
- [x] 10 Runner 与日志：`scripts/02_run_generation.py` 生成 JSONL raw logs 与 manifest，支持 `--live-llm` 真实 vLLM endpoint 调用和 `--num-shards/--shard-index` 样本级分片。
- [x] 11 自动指标：`scripts/04_compute_metrics.py` 生成 record/aggregate metrics 和 coverage report，并已支持 unified schema gold 字段映射。
- [x] 12 LLM judge：`scripts/03_run_judge.py` 默认 mock；真实 DeepSeek 需显式环境变量，已验证 `thinking_type=disabled` 后可稳定解析。
- [x] 13 实验矩阵：`configs/experiments.yaml` 与 `scripts/run_exp*.sh`。
- [x] 14 统计、表格和图：`scripts/05_make_tables.py`、`scripts/06_make_figures.py`。
- [x] 15 Human audit：`scripts/07_sample_human_audit.py`、`scripts/08_human_agreement.py`。
- [x] 16 Paper artifacts：`scripts/09_export_paper_artifacts.py`。

## 已验证命令

- [x] 本机 `python -m compileall prism_tutor scripts data serving tests`
- [x] 本机 `python -m pytest -q`，结果：48 passed。
- [x] 服务器 `python -m pytest -q`，结果：63 passed。
- [x] `python scripts/00_prepare_env_check.py --config configs/default.yaml --dry-run`
- [x] `python scripts/01_build_datasets.py --help`
- [x] `bash serving/start_vllm_gpu0.sh`
- [x] `python serving/health_check.py --output outputs/logs/model_health_check.json`
- [x] mock generation -> judge -> metrics -> tables -> figures -> human audit -> paper export 链路。
- [x] 服务器 live health check：`python serving/health_check.py --config configs/default.yaml --live`，两个 endpoint 均 ok。
- [x] 服务器 live generation smoke：`python scripts/02_run_generation.py --limit 1 --methods single_tutor,ours_full --datasets mathdial --split test --run-id live_qwen_smoke_agent_calls --output_dir outputs --live-llm`，2 attempted / 2 succeeded / 0 failed。
- [x] live smoke 日志确认：`single_tutor` 1 次 agent call，`ours_full` 10 次 agent call，token usage 来源为 API，两个 endpoint 均被调用，raw/stripped completion 无 `<think>` 残留。
- [x] 本机下载并传输 MathDial：`eth-nlped/mathdial`，服务器 build 后 `processed_count=18609`，官方 test split `3699`。
- [x] 本机下载并传输 Bridge：`rosewang2008/bridge`，服务器 build 后 `processed_count=700`，test split `565`，student error/remediation/teacher intention 完整率均为 `1.0`。
- [x] 本机下载并传输 MaE Math Misconceptions：`nancyotero-projects/math-misconceptions`，服务器 build 后 `processed_count=220`，test split `220`，unique misconception labels `55`。
- [x] 真实 MathDial test live smoke：`live_qwen_mathdial_real_smoke`，2 attempted / 2 succeeded / 0 failed，API token usage 正常，无 `<think>`。
- [x] 真实 Bridge test live smoke：`live_qwen_bridge_real_smoke`，2 attempted / 2 succeeded / 0 failed，API token usage 正常，两个 endpoint 被调用。
- [x] 真实 Misconception test live smoke：`live_qwen_misconception_real_smoke`，2 attempted / 2 succeeded / 0 failed，API token usage 正常。
- [x] Exp0-Exp6 dry-run matrix smoke：123 generation rows，judge/metrics/tables/figures/human-audit sample/paper artifacts 全链路生成。
- [x] Exp0-Exp6 Qwen live matrix smoke：`outputs/exp_matrix_live_smoke`，123 generation rows，123 success，123 parse_success，31 methods，3 datasets，两个 endpoint 均被调用，无 `<think>`。
- [x] live smoke 自动指标：123 rows，aggregate 81 rows，routing missing gold `0`，orphan_generation_count `0`。
- [x] live smoke 真实 DeepSeek judge：`outputs/exp_matrix_live_smoke/judge_scores_real_full`，123 rows，123 parsed，0 errors，metadata 记录 actual_model、api_date、temperature、top_p、max_tokens、thinking_type。
- [x] live smoke blind audit 文件：`outputs/exp_matrix_live_smoke/human_audit_200`，目标 200，实际 123，shortage 已写入 sampling manifest。
- [x] 全量规模估算：`outputs/exp_matrix_live_smoke/full_run_estimate.json`，约 204,373 generation records、953,088 agent calls、1.6B tokens。
- [x] Sharded runner dry-run 验证：`outputs/shard_smoke`，2 个 shard 无 sample overlap。
- [x] Sharded runner live 验证：`outputs/shard_live_smoke`，Exp0 shard0 limit=1，15/15 success，metrics orphan_generation_count `0`。
- [x] 全量 shard plan：`outputs/full_run/shard_plan.json`，1792 个 job，estimated_records `204373`，初始状态 pending。
- [x] Shard plan/status/launch 工具验证：`outputs/shard_tool_smoke`，Exp0 8-shard dry-run plan，`launch --next` 后 1 completed / 7 pending，15 generation rows，0 error rows。
- [x] Shard concurrency maintainer：`python scripts/11_plan_or_run_shards.py maintain --plan outputs/full_run/shard_plan.json --target-running 2 --max-launches 2`，当前 running 已达目标时不会重复启动 job。
- [x] Shard supervisor：`python scripts/11_plan_or_run_shards.py supervise --plan outputs/full_run/shard_plan.json --target-running 18 --interval-seconds 120 --log-path outputs/full_run/logs/shards/supervisor_compact.jsonl`，服务器 PID `2472200`。
- [x] Full-run finalization gate：`scripts/12_finalize_full_run.py --allow-incomplete --dry-run` 已在服务器真实 shard plan 上验证，默认不调用 judge，计划步骤为 auto metrics、tables、figures、human audit sample、paper artifacts。
- [x] Shard progress report：`python scripts/11_plan_or_run_shards.py progress --plan outputs/full_run/shard_plan.json --supervisor-log outputs/full_run/logs/shards/supervisor_compact.jsonl --rate-window 5` 已在服务器验证，health summary 当前为 `ok`，并记录 `target_running`。
- [x] 正式 full_run 后台运行：已完成 40 个 shard，18 个 shard 正在 running；最近检查 generation_rows `4619`、error_rows `0`，近期吞吐约 `88.67` rows/min，粗略 ETA 约 `37.55` hours，GPU2/GPU3 均 100% utilization，health summary 为 `ok`。

## 仍需服务器真实执行的项目

- [x] 在 `10.154.22.11` 的 `zsh` 账户中创建 conda 环境 `prism_tutor`。
- [x] 在 2、3 号 GPU 上启动 Qwen3-8B full BF16 vLLM，不使用 AWQ/FP8/4-bit 主实验量化。
- [x] 将真实 MathDial、Bridge、MaE Misconception raw 数据放入 `data/raw/` 并运行 dataset build。
- [x] 使用真实 Qwen3-8B endpoint 跑 Exp0-Exp6 live smoke generation。
- [x] 使用真实 DeepSeek judge API 跑 live smoke judge，并保存 actual model id、日期与 raw response。
- [x] 用 live smoke raw logs 重新生成 smoke 版 tables、figures、paper artifacts。
- [x] 提供全量正式实验前的分片执行能力与规模估算 gate。
- [x] 提供全量正式实验的 shard manifest、status、launch 和 maintain 工具：`scripts/11_plan_or_run_shards.py`。
- [ ] 使用真实 Qwen3-8B endpoint 跑完全量 Exp0-Exp6 generation。目前已完成 Exp0 前 40 个 shard，并由 supervisor 维持 18 个 shard 并发继续运行；尚未完成全量 1792 个 job。
- [ ] 全量实验完成后执行正式 200 条 blind human audit，并填入人工标签后计算 agreement。
- [ ] 用全量真实 raw logs 重新生成正式论文 tables、figures、paper artifacts。

## 安全边界

- [x] 未把 `指导方案.md` 中的明文 key 复制到代码、配置或任务卡。
- [x] `outputs/` 中除 `outputs/README.md` 外的 runtime 产物被 `.gitignore` 忽略。
- [x] judge 默认 dry-run；真实 API 调用必须同时设置 `DEEPSEEK_API_KEY` 和 `PRISM_TUTOR_ENABLE_REAL_JUDGE=1`。
