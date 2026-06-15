# PRISM-Tutor Task Card Index

## 项目目标

PRISM-Tutor 目标是在同一 Qwen3-8B、同一数据集与同一运行封装下，评估教学风险感知的多 Agent 路由、预算讨论与学生状态提交机制，比较其教学质量、误解诊断、答案泄露、状态一致性和 token 成本。

## 执行顺序表

| 顺序 | 文件 | 主题 | 依赖 | 预计产物 | 完成标准摘要 |
|---:|---|---|---|---|---|
| 01 | `01_PROJECT_SETUP_AND_REPRODUCIBILITY.md` | 项目骨架与复现规范 | 无 | 配置、目录、manifest 规范 | seed、git hash、config snapshot 规范齐全 |
| 02 | `02_ENVIRONMENT_AND_DEPENDENCIES.md` | 环境检查与依赖安装 | 01 | env check 脚本与环境报告 | conda、Python、GPU、包版本可校验 |
| 03 | `03_QWEN3_8B_MODELSCOPE_VLLM_SERVING.md` | Qwen3-8B vLLM 服务 | 02 | vLLM 启动脚本、health check | 双 endpoint 或 TP fallback 可验证 |
| 04 | `04_DATASETS_SCHEMA_AND_SPLITS.md` | 数据集 schema 与 split | 01,02 | processed/splits/report | 三个数据集统一 schema 且有 dataset report |
| 05 | `05_AGENT_SCHEMAS_AND_BASE_CLIENT.md` | Agent schema 与 LLM client | 01,03 | Pydantic schema、base client | 请求、usage、latency、raw/parsed 输出可记录 |
| 06 | `06_AGENT_PROMPTS_AND_JSON_REPAIR.md` | Prompt 与 JSON repair | 05 | agent prompts、retry/repair | JSON 失败不丢样本 |
| 07 | `07_LANGGRAPH_RUNTIME_STATE.md` | LangGraph 状态与图结构 | 05,06 | graph state、runtime graph | baseline 与 ours 使用同一运行封装 |
| 08 | `08_BASELINE_METHODS.md` | Baseline 方法 | 04,07 | B0-B5 baseline | 公平性约束可检查 |
| 09 | `09_PRISM_TUTOR_RUNTIME_MODULES.md` | PRISM 核心模块 | 04,07 | risk/router/budget/commit | M1-M3 可独立运行 |
| 10 | `10_GENERATION_RUNNER_AND_LOGGING.md` | 生成 runner 与日志 | 08,09 | generation JSONL、error logs | raw logs 可复现实验 |
| 11 | `11_AUTOMATIC_METRICS.md` | 自动指标 | 10 | metrics CSV/JSON | 自动指标与 judge 指标分离 |
| 12 | `12_LLM_JUDGE_DEEPSEEK_V4_PRO.md` | LLM judge | 10 | judge scores、raw responses | 实际模型名、日期、参数和 prompt 保存 |
| 13 | `13_EXPERIMENT_MATRIX_EXP0_TO_EXP6.md` | Exp0-Exp6 实验矩阵 | 10,11,12 | experiment outputs | 每个实验有输入、方法、指标、表格 |
| 14 | `14_STATISTICS_TABLES_AND_FIGURES.md` | 统计检验、表格与图 | 11,13 | tables、figures、CI | 结果从 raw logs 自动生成 |
| 15 | `15_HUMAN_AUDIT_AFTER_ALL_EXPERIMENTS.md` | 正式人工抽样审计 | 12,13,14 | blind audit CSV、agreement report | 全部实验后 blind audit |
| 16 | `16_PAPER_ARTIFACT_EXPORT_AND_FINAL_CHECKS.md` | 论文 artifact 与最终复现检查 | 14,15 | paper artifacts、checklist | manifest 与 reproducibility checklist 完整 |

## 并行与串行关系

- [ ] 必须串行：01 -> 02 -> 03，模型服务依赖环境检查。
- [ ] 必须串行：05 -> 06 -> 07，schema/client、prompt、runtime state 逐层依赖。
- [ ] 必须串行：10 -> 11/12 -> 13 -> 14 -> 15 -> 16，实验输出、指标、统计、人工审计和论文导出依赖 raw logs。
- [ ] 可并行：04 数据构建可在 03 服务配置之外推进。
- [ ] 可并行：08 baseline 与 09 PRISM 模块可在 07 完成后并行实现。
- [ ] 可并行：11 自动指标与 12 judge client 可在 10 日志格式稳定后并行实现。

## 最小可行执行路径

- [ ] 01 项目复现规范。
- [ ] 02 环境检查。
- [ ] 03 Qwen3-8B 服务 health check。
- [ ] 04 数据 schema 与小规模 split。
- [ ] 05-07 Agent client、prompt、runtime graph。
- [ ] 08 中至少 B0、B2、B5。
- [ ] 09 中至少 M1、M2、M3。
- [ ] 10 小样本 generation runner。
- [ ] 11 核心自动指标。
- [ ] 13 先跑 Exp4 小规模 smoke。

## 完整论文实验执行路径

- [ ] 完成 01-16 全部 task card。
- [ ] Exp0-Exp6 全部从 raw logs 运行并生成 metrics、tables、figures。
- [ ] LLM judge 完整保存 raw response 与 parsed scores。
- [ ] 全部自动实验结束后执行 blind human audit。
- [ ] 导出 paper artifacts、experiment manifest 和 reproducibility checklist。

## 风险清单与 fallback plan

- [ ] 数据不可用：记录缺失来源与下载失败日志；允许本地手动下载后传入 `data/raw/`；不得用 test set 手动调阈值。
- [ ] Qwen3-8B 服务不稳定：先用单卡双副本；OOM 时切换双卡 tensor parallel；timeout 样本记录失败并可重试。
- [ ] DeepSeek judge API 不稳定：保留请求失败日志；使用同一 judge schema 切换备用 judge；报告中记录实际模型名和调用日期。
- [ ] 生成 JSON 不稳定：retry 一次，再 JSON repair；仍失败则 `parse_success=false`，不得删除样本。
- [ ] 指标无法自动计算：输出 skipped reason 与 coverage report；gold label 指标优先自动评估，judge 不替代 gold metric。
- [ ] 人工审计延迟：自动实验与 judge 结果先冻结；human audit 独立补跑，不回改主结果。

## 额外拆分说明

- [ ] 本索引保持 16 张执行卡，未额外增加编号，原因是用户建议清单已覆盖项目从初始化到论文导出的完整链条。
- [ ] 安全补充：`指导方案.md` 中出现明文 API key，后续任务不得复制密钥到代码、配置、日志或 task card，只能使用环境变量或本机 local-only 凭据来源。
