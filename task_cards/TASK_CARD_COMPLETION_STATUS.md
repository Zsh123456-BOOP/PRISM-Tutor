# Task Card Completion Status

## 当前状态

本仓库已经具备 Task Card 01-16 的可运行代码骨架、mock/dry-run 端到端链路，以及服务器上的 Qwen3-8B vLLM live smoke 链路。MathDial、Bridge 与 MaE Math Misconceptions 三个真实数据集均已下载、传输到服务器并完成 schema build。Exp0-Exp6 已完成真实 Qwen live smoke（limit=1）与 downstream metrics/tables/figures/artifact smoke；真实 DeepSeek judge 已对 live smoke 123 条 generation 完成评分。runner 已支持 sample-level sharding，用于拆分全量正式实验，并已支持 `maintain --target-running N` 与 `supervise` 可恢复补齐后台并发。PRISM runtime 现在会从 `configs/default.yaml` 注入 risk weights、thresholds、budget 与 commit threshold；Exp5 ablation 会真实禁用或替换对应模块/风险信号；Exp6 robustness 会展开 noisy-agent probability × token budget 变体，并把 deterministic noisy injection 参数写入 raw log。全量完成后可用 `scripts/12_finalize_full_run.py` gated finalization 生成 metrics/tables/figures/human audit sample/paper artifacts；finalization manifest/stdout 会输出 completed/total/can_finalize/planned_steps，并会把每个后处理 step 的 stdout/stderr 保存到 `outputs/full_run/logs/finalization`；formal finalization 会拒绝 incomplete shards 和 generation error rows，非 dry-run 时某一步失败会 fail-fast 并将后续步骤标记为 skipped；paper artifact exporter 会检查 `outputs/full_run/*` run-local 产物；human audit sampler 会在正式模式下要求 metrics、generation logs、judge scores 和 tables 全部存在，并用 generation logs 回填 blind CSV 所需 response/context 字段；finalization 支持人工标注完成后用 `--run-human-agreement` 在 paper artifact 前生成 run-local agreement report；table builder 会从 record metrics 自动导出 Table 1-6 及 significance JSON；paper artifact exporter 可用 full-run shard plan 生成覆盖 Exp0-Exp6 的 experiment manifest，并记录 datasets、split、methods、job count、estimated records 和输出路径；reproducibility checklist 会记录 seed/config/model/generation、GPU、judge metadata、data/log paths、git 和 package versions；`scripts/11_plan_or_run_shards.py progress` 可从 supervisor log 输出完成率、近期吞吐、ETA 与 health summary，并可识别 running 超过 supervisor target 的情况。当前全量 Exp0-Exp6 正式实验已经开始运行并由服务器 supervisor 接管，但尚未完成；也尚未执行正式人工标注版 200 条 blind human audit。

## 已完成的代码路径

- [x] 01 项目初始化与复现规范：`configs/`、`environment.yml`、`requirements.txt`、`pyproject.toml`、复现 metadata、`.gitignore`；experiment manifest 带 `schema_version`、audit block、config snapshot、input/output paths、duration 和 CUDA/VLLM env metadata。
- [x] 02 环境检查：`scripts/00_prepare_env_check.py` 可生成 env check report，包含 CUDA version、GPU count、CUDA_VISIBLE_DEVICES conflict、warnings/errors 和缺失依赖 fallback suggestions。
- [x] 03 Qwen3-8B vLLM 服务配置：`serving/start_vllm_*.sh`、`serving/health_check.py` dry-run 与服务器 live health check 可用。
- [x] 04 数据 schema 与 split：`scripts/01_build_datasets.py`、`prism_tutor/data/` 支持本地 raw JSON/JSONL/CSV；MathDial、Bridge、MaE Misconception loader 已按真实源格式验证。
- [x] 05 Agent schema 与 base client：`prism_tutor/agents/schemas.py`、mock-safe OpenAI-compatible client，支持 endpoint-specific served model name 与 retryable HTTP/timeout request retry；测试覆盖全部 agent schema 的有效 payload、缺失必填、confidence 越界、额外字段、关键 Literal/嵌套边界，以及 `configs/default.yaml` 到 live LLM client 的 endpoint/model/generation config 注入。
- [x] 06 Agent prompt 与 JSON repair：prompt、`<think>` 剥离、retry/repair parser；测试覆盖 parse failure 后 retry 与 final_tutor `<think>` 不进入 student-visible output。
- [x] 07 Runtime state：`prism_tutor/runtime/` graph state、checkpoint、node interface；`GraphBuilder` 优先使用 LangGraph backend，当前环境缺少 LangGraph 时使用同一 `invoke` 接口的 `simple_fallback`，测试覆盖 checkpoint audit fields、max rounds 与 token budget。
- [x] 08 Baseline 方法：method registry 覆盖 B0-B5 与实验变体；`prism_tutor/baselines/` 提供 Single Tutor、Fixed 2、Fixed 4、Debate、Generic Sparse、Difficulty Routing 与 Oracle Routing planner；runner live baseline 路径会记录 `baseline_plan`，Generic Sparse 与 Difficulty Routing 有禁止读取教育风险字段的 fairness tests。
- [x] 09 PRISM 模块：risk estimator、QoS router、budget controller、state commit、M1/M2/M3 graph。
- [x] 10 Runner 与日志：`scripts/02_run_generation.py` 生成 JSONL raw logs 与 schema-versioned manifest，支持 `--live-llm` 真实 vLLM endpoint 调用和 `--num-shards/--shard-index` 样本级分片。
- [x] 11 自动指标：`scripts/04_compute_metrics.py` 生成 record/aggregate metrics、`routing_metrics.csv`、`state_metrics.csv`、`leakage_rule_hits.jsonl`、`leakage_metrics.csv` 和 coverage/alignment report，并已支持 unified schema gold 字段映射；`parse_success=false` 样本会保留在记录和 parse_success_rate 中，但 internal correctness / misconception 等结构化指标按 missing 处理；judge row 只有在 `parsed_score` 有效且无 `error` 时才计入 judge leakage coverage，失败 judge 不会被当作 non-leakage。
- [x] 12 LLM judge：`scripts/03_run_judge.py` 默认 mock；真实 DeepSeek 需显式环境变量，已验证 `thinking_type=disabled` 后可稳定解析；judge runner 会为单候选/多候选保存 deterministic display order、seed、唯一 candidate_label，降低 position bias；judge score schema 会正确解析 `answer_leakage` 布尔值，避免字符串 `"false"` 被 truthiness 误判，并拒绝 bool 型数值分数；metrics/finalization 支持合并 rule leakage 与 judge leakage，输出 `judge_leakage`、`final_leakage` 和 `leakage_conflict`；run-level judge metadata 汇总 actual_models、parsed_count、error_count 和 raw_response_count；formal finalization 的 judge step 默认传 `--require-real`，只有显式 `--allow-mock-judge` 才允许 smoke mock judge。
- [x] 13 实验矩阵：`configs/experiments.yaml` 与 `scripts/run_exp*.sh`。
- [x] 14 统计、表格和图：`scripts/05_make_tables.py`、`scripts/06_make_figures.py`；tables/significance 使用合并后的 `final_leakage`、`judge_leakage`、`leakage_conflict`，并能把 Exp6 的 `base__noise...__budget...` 方法变体纳入 robustness 表；table builder 会写出 `table_manifest.json`，正式模式拒绝空任务表或核心比较没有 paired samples，`--allow-incomplete-tables` 仅用于 smoke；figure builder 会在缺少 `internal_correctness`、`total_tokens`、`risk_bucket`、`agent_calls` 或 `state_conflict_rate` 等关键列时 fail-fast，并写出 `figure_manifest.json` 记录每张图的数据摘要。
- [x] 15 Human audit：`scripts/07_sample_human_audit.py`、`scripts/08_human_agreement.py`；blind audit CSV 不包含 method/selected agents/risk 字段，最终展示顺序由 seed 全局随机化，并在 sampling manifest 记录 display order seed 与 sample id 顺序；formal sampler 要求具体 Table 1-6、`table_manifest.json`、judge scores 和 generation logs 存在，并拒绝缺少 `problem` 或 `candidate_response` 的不可标注 blind rows；sampler 会为同一样本的 ours 与 baseline 生成 A/B blind preference 字段，并把 method 映射单独写入不发给标注者的 `preference_mapping.json`；human agreement 正式模式要求 `sample_id`、`annotator_id`、`human_quality_score`、`human_leakage_label`、`human_preference` 核心列，缺列时输出 schema error 并返回非 0；正式 agreement gate 会拒绝空标签、无双标 overlap 或 preference 缺失，`--allow-unlabeled` 仅用于 smoke。
- [x] 16 Paper artifacts：`scripts/09_export_paper_artifacts.py`；paper artifact summary 会在 missing experiments 或 checklist failure 时标记 final artifact status failed，避免 incomplete artifacts 显示为 passed；paper artifact CLI 默认在 final artifact status failed 时返回非 0，只有 `--allow-failed-checklist` 才允许 smoke；显式传入的 `--shard-plan` 缺失或 JSON 损坏会直接失败，避免丢失 full-run traceability；experiment manifest 会记录 config snapshot path/hash、generator model、generation config、judge config 摘要，并把这些参数复制到每个实验条目；artifact index 会逐项列出核心 metrics、Table 1-6、Figure 1-5、human audit 和 judge 文件的状态、来源脚本和输入；formal finalization 会拒绝 generation error rows，并在后处理步骤失败时 fail-fast 而不是继续生成后续 artifact；reproducibility checklist 会要求 judge raw response、完整且非 mock/dry-run 的 judge metadata、`table_manifest.json`、具体 Table 1-6 CSV/TeX、Figure 1-5 PDF、`figure_manifest.json`、`human_audit_blind.csv`、`sampling_manifest.json`、`preference_mapping.json` 和非空关键文件，并递归扫描 required artifact 目录中的明文 secret。

## 已验证命令

- [x] 本机 `python -m compileall prism_tutor scripts data serving tests`
- [x] 本机 `python -m pytest -q`，结果：173 passed。
- [x] 服务器 `python -m pytest -q`，结果：173 passed。
- [x] `python scripts/00_prepare_env_check.py --config configs/default.yaml --dry-run`
- [x] 服务器 env check dry-run：`CUDA_VISIBLE_DEVICES=2,3 python scripts/00_prepare_env_check.py --config configs/default.yaml --output /tmp/prism_env_check_latest.json --dry-run`，结果 `status=ok`，检测到 4 张 GPU，CUDA_VISIBLE_DEVICES 与配置一致。
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
- [x] Human audit formal gate：`scripts/07_sample_human_audit.py` 正式模式缺少 judge scores 或 tables 时拒绝抽样；`--allow-incomplete` 仅用于 smoke，并会在 sampling manifest 记录 prerequisite 状态。
- [x] Human agreement finalization gate：`scripts/12_finalize_full_run.py --run-human-agreement` 会在 `paper_artifacts` 前运行 `scripts/08_human_agreement.py`，并使用 full-run 本地 `human_audit_labeled.csv` / `human_agreement_report.json` 路径。
- [x] 全量规模估算：初版 `outputs/exp_matrix_live_smoke/full_run_estimate.json` 约 204,373 generation records、953,088 agent calls、1.6B tokens；修正 Exp6 robustness 的 noise × budget 展开后，服务器 `outputs/full_run/shard_plan.json` 当前 estimated_records 为 `294053`。
- [x] Sharded runner dry-run 验证：`outputs/shard_smoke`，2 个 shard 无 sample overlap。
- [x] Sharded runner live 验证：`outputs/shard_live_smoke`，Exp0 shard0 limit=1，15/15 success，metrics orphan_generation_count `0`。
- [x] 全量 shard plan：`outputs/full_run/shard_plan.json`，1792 个 job，estimated_records `294053`；旧估算已在服务器备份为 `outputs/full_run/shard_plan.pre_exp6_variants_1694d10.json`。
- [x] Shard plan/status/launch 工具验证：`outputs/shard_tool_smoke`，Exp0 8-shard dry-run plan，`launch --next` 后 1 completed / 7 pending，15 generation rows，0 error rows。
- [x] Shard concurrency maintainer：`python scripts/11_plan_or_run_shards.py maintain --plan outputs/full_run/shard_plan.json --target-running 2 --max-launches 2`，当前 running 已达目标时不会重复启动 job。
- [x] Shard supervisor：`python scripts/11_plan_or_run_shards.py supervise --plan outputs/full_run/shard_plan.json --target-running 18 --interval-seconds 120 --log-path outputs/full_run/logs/shards/supervisor_compact.jsonl`，服务器 PID `2472200`。
- [x] Full-run finalization gate：`scripts/12_finalize_full_run.py --allow-incomplete --dry-run` 已在服务器真实 shard plan 上验证，默认不调用 judge，计划步骤为 auto metrics、tables、figures、human audit sample、paper artifacts；manifest/stdout 会输出 `completed_jobs`、`total_jobs`、`can_finalize`、`planned_steps`。
- [x] Full-run finalization latest dry-run：服务器真实 shard plan 当前 `156/1792` jobs completed，`can_finalize=false`，planned steps 仍为 auto metrics、tables、figures、human audit sample、paper artifacts，未调用 judge/API。
- [x] Finalization step logs：`scripts/12_finalize_full_run.py` 非 dry-run 时会为每个后处理 step 保存 stdout/stderr，并在 manifest 记录日志路径。
- [x] Paper artifact run-local path gate：`scripts/09_export_paper_artifacts.py --artifact-prefix outputs/full_run` 可让 reproducibility checklist 和 artifact index 检查 full-run 目录，而不是误查全局 `outputs/*`。
- [x] Reproducibility checklist coverage：`scripts/09_export_paper_artifacts.py` 导出的 checklist 包含 seed、config、model、GPU、judge metadata、data/log paths、git 和 package versions。
- [x] Experiment manifest shard-plan coverage：`scripts/09_export_paper_artifacts.py --shard-plan` 可从 full-run shard plan 生成 Exp0-Exp6 manifest 元数据，并过滤无 experiment/name 的旧日志 manifest。
- [x] Table export coverage：`scripts/05_make_tables.py` 可自动生成 `table1_main_results` 至 `table6_robustness` 的 CSV/TeX，并写出 paired significance JSON。
- [x] Exp5/Exp6 runtime variant coverage：Exp5 ablation 会真实禁用 risk estimator、QoS routing、budget controller、state commit 或对应风险项；Exp6 dry-run smoke 验证 `fixed_4/debate/generic_sparse/ours_full × noise{0.2,0.4} × budget{1000,2000,4000}` 展开为 24 个 method variants。
- [x] Shard progress report：`python scripts/11_plan_or_run_shards.py progress --plan outputs/full_run/shard_plan.json --supervisor-log outputs/full_run/logs/shards/supervisor_compact.jsonl --rate-window 5` 已在服务器验证，health summary 当前为 `ok`，并记录 `target_running`。
- [x] 正式 full_run 后台运行：已完成 195 个 shard，18 个 shard 正在 running；最近检查 generation_rows `18151`、error_rows `0`，estimated_records `294053`，completion_fraction `0.06172696758747573`；health summary 为 `ok`。

## 仍需服务器真实执行的项目

- [x] 在 `10.154.22.11` 的 `zsh` 账户中创建 conda 环境 `prism_tutor`。
- [x] 在 2、3 号 GPU 上启动 Qwen3-8B full BF16 vLLM，不使用 AWQ/FP8/4-bit 主实验量化。
- [x] 将真实 MathDial、Bridge、MaE Misconception raw 数据放入 `data/raw/` 并运行 dataset build。
- [x] 使用真实 Qwen3-8B endpoint 跑 Exp0-Exp6 live smoke generation。
- [x] 使用真实 DeepSeek judge API 跑 live smoke judge，并保存 actual model id、日期与 raw response。
- [x] 用 live smoke raw logs 重新生成 smoke 版 tables、figures、paper artifacts。
- [x] 提供全量正式实验前的分片执行能力与规模估算 gate。
- [x] 提供全量正式实验的 shard manifest、status、launch 和 maintain 工具：`scripts/11_plan_or_run_shards.py`。
- [ ] 使用真实 Qwen3-8B endpoint 跑完全量 Exp0-Exp6 generation。目前已完成 Exp0 前 195 个 shard，并由 supervisor/maintainer 维持运行；尚未完成全量 1792 个 job。
- [ ] 全量实验完成后执行正式 200 条 blind human audit，并填入人工标签后计算 agreement。
- [ ] 用全量真实 raw logs 重新生成正式论文 tables、figures、paper artifacts。

## 安全边界

- [x] 未把 `指导方案.md` 中的明文 key 复制到代码、配置或任务卡。
- [x] `outputs/` 中除 `outputs/README.md` 外的 runtime 产物被 `.gitignore` 忽略。
- [x] judge 默认 dry-run；真实 API 调用必须同时设置 `DEEPSEEK_API_KEY` 和 `PRISM_TUTOR_ENABLE_REAL_JUDGE=1`。
